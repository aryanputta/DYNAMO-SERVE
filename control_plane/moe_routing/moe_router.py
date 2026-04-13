"""
Expert-Parallel MoE Router with per-expert KV cache partitioning.

Mixture-of-Experts models (Mixtral-8×7B, DeepSeek-V3, etc.) route each token
to K of E experts. In distributed serving this creates two challenges:

  1. Expert parallelism: different experts live on different GPUs.
     A token batch must be routed to the right GPU, processed, then all-reduced.

  2. Per-expert KV cache: in speculative or multi-step decoding the KV cache
     for expert i should live on the GPU hosting expert i to avoid transfers.

This module models both effects so the scheduler can:
  - Assign requests to the node group that minimises expert-transfer cost.
  - Partition KV blocks per expert to prevent cross-GPU eviction storms.

Design
------
  ExpertGroup: a set of GPU nodes that collectively host one expert shard.
               (For 8 experts on 8 GPUs: one expert per GPU.)

  MoERouter:   Given a request, scores each ExpertGroup and returns a ranked
               list of (expert_group_id, estimated_routing_overhead_ms).

  The routing overhead models:
    - Token scatter/gather across expert GPUs (NVLink vs PCIe cost)
    - Load imbalance when some experts are hotter than others
    - KV cache miss penalty when KV blocks must be migrated between expert GPUs

References:
  - Mixtral-8×7B: https://arxiv.org/abs/2401.04088
  - DeepSpeed-MoE expert parallelism
  - NVIDIA Dynamo MoE serving notes
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.models import GPUNode, InferenceRequest

logger = logging.getLogger(__name__)

# Default MoE architecture parameters (Mixtral-8×7B style)
DEFAULT_NUM_EXPERTS   = 8
DEFAULT_TOP_K_EXPERTS = 2      # tokens routed to top-2 experts per layer
DEFAULT_NUM_LAYERS    = 32


@dataclass
class ExpertGroup:
    """
    A logical group of GPU nodes hosting one expert shard.

    In a 8-expert / 8-GPU deployment each ExpertGroup has one node.
    In a 8-expert / 4-GPU deployment each node hosts 2 experts.
    """

    expert_id: int
    nodes: list[GPUNode]
    kv_blocks_reserved: int = 0   # KV blocks reserved for this expert's cache

    @property
    def total_kv_free(self) -> int:
        return sum(n.kv_blocks_free for n in self.nodes)

    @property
    def mean_memory_pressure(self) -> float:
        if not self.nodes:
            return 1.0
        return sum(n.memory_pressure for n in self.nodes) / len(self.nodes)

    @property
    def nvlink_capable(self) -> bool:
        return all(n.nvlink_enabled for n in self.nodes)

    @property
    def peak_bandwidth_gbps(self) -> float:
        return sum(n.memory_bandwidth_gbps for n in self.nodes)


@dataclass
class RoutingDecision:
    """Output of MoERouter.route() for a single request."""

    request_id: str
    selected_experts: list[int]          # Which expert IDs were selected (top-k)
    expert_groups: list[ExpertGroup]     # Corresponding ExpertGroup objects
    routing_overhead_ms: float           # Estimated token scatter/gather cost
    kv_partition_plan: dict[int, int]    # expert_id → kv_blocks_reserved
    load_imbalance: float                # 0.0 = perfectly balanced, 1.0 = hot expert


class MoERouter:
    """
    Routes MoE inference requests to the appropriate expert GPU groups.

    Integrates with the KVAwareSLAScheduler: when model_id indicates a MoE
    model, the scheduler delegates expert placement to MoERouter before
    calling the standard PlacementEngine for the chosen primary node.
    """

    MOE_MODELS = {"mixtral-8x7b", "deepseek-v2", "deepseek-v3", "grok-1", "qwen-moe"}

    def __init__(
        self,
        expert_groups: list[ExpertGroup],
        num_experts: int = DEFAULT_NUM_EXPERTS,
        top_k: int = DEFAULT_TOP_K_EXPERTS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        seed: int = 42,
    ) -> None:
        self._groups    = {eg.expert_id: eg for eg in expert_groups}
        self.num_experts = num_experts
        self.top_k       = top_k
        self.num_layers  = num_layers
        self._rng        = random.Random(seed)

        # Per-expert token counters for load tracking
        self._expert_token_counts: dict[int, int] = {i: 0 for i in range(num_experts)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, request: InferenceRequest) -> RoutingDecision:
        """
        Select top-k experts for *request* and estimate routing overhead.

        Uses a learned gate simulation: experts are scored by a combination
        of current load, KV headroom, and NVLink availability.
        """
        expert_scores = self._score_experts(request)
        # Select top-k by score
        ranked = sorted(range(self.num_experts), key=lambda i: -expert_scores[i])
        selected = ranked[:self.top_k]

        groups    = [self._groups[i] for i in selected if i in self._groups]
        overhead  = self._routing_overhead_ms(request, groups)
        kv_plan   = self._kv_partition_plan(request, groups)
        imbalance = self._load_imbalance(selected)

        # Update load counters
        for eid in selected:
            self._expert_token_counts[eid] += request.prompt_tokens

        return RoutingDecision(
            request_id=request.request_id,
            selected_experts=selected,
            expert_groups=groups,
            routing_overhead_ms=overhead,
            kv_partition_plan=kv_plan,
            load_imbalance=imbalance,
        )

    def is_moe_model(self, model_id: str) -> bool:
        return any(m in model_id.lower() for m in self.MOE_MODELS)

    def expert_load_stats(self) -> dict[int, int]:
        return dict(self._expert_token_counts)

    def reset_load_counters(self) -> None:
        self._expert_token_counts = {i: 0 for i in range(self.num_experts)}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_experts(self, request: InferenceRequest) -> list[float]:
        scores = []
        total_tokens = sum(self._expert_token_counts.values()) + 1

        for eid in range(self.num_experts):
            group = self._groups.get(eid)
            if group is None:
                scores.append(-1e9)
                continue

            # Load balance score (lower utilisation = higher score)
            load_ratio = self._expert_token_counts[eid] / total_tokens
            load_score = 1.0 - load_ratio * self.num_experts

            # KV headroom score
            kv_score = group.total_kv_free / max(1, sum(
                n.kv_blocks_total for n in group.nodes
            ))

            # Memory pressure penalty
            mem_score = 1.0 - group.mean_memory_pressure

            # NVLink bonus for long-context
            nvlink_bonus = 0.1 if (group.nvlink_capable and request.is_long_context) else 0.0

            scores.append(0.40 * load_score + 0.35 * kv_score + 0.15 * mem_score + nvlink_bonus)

        return scores

    def _routing_overhead_ms(
        self, request: InferenceRequest, groups: list[ExpertGroup]
    ) -> float:
        """
        Estimate token scatter + gather time for expert-parallel execution.

        Each token must be:
          1. Sent to the expert GPU (scatter)
          2. Processed by the expert MLP
          3. Gathered back to the primary GPU (reduce)

        Overhead = (scatter_bytes + gather_bytes) / bandwidth
        """
        if not groups:
            return 0.0

        # Token vector size per expert layer (d_model = 4096, fp16)
        bytes_per_token = 4096 * 2
        scatter_tokens  = request.prompt_tokens * self.top_k

        # Bandwidth: use minimum across groups (bottleneck)
        min_bw_gbps = min(g.peak_bandwidth_gbps for g in groups) if groups else 100.0

        # NVLink: fast path; PCIe: slow path
        effective_bw = min_bw_gbps if all(g.nvlink_capable for g in groups) else min_bw_gbps * 0.25

        scatter_ms = (
            scatter_tokens * bytes_per_token * self.num_layers
            / (effective_bw * 1e9)
            * 1000.0
        )
        # Gather is symmetric
        total_ms = scatter_ms * 2.0
        return max(0.1, total_ms)

    def _kv_partition_plan(
        self, request: InferenceRequest, groups: list[ExpertGroup]
    ) -> dict[int, int]:
        """
        Assign KV blocks to expert groups proportionally to their role.

        Each expert handles 1/top_k of the total tokens per layer,
        so each expert needs approximately kv_blocks / top_k blocks.
        """
        total_blocks = request.kv_blocks_needed
        blocks_per_expert = max(1, total_blocks // self.top_k)
        plan = {}
        for group in groups:
            plan[group.expert_id] = blocks_per_expert
        return plan

    def _load_imbalance(self, selected: list[int]) -> float:
        """
        Jain's fairness index-derived imbalance metric.
        0.0 = perfectly balanced; 1.0 = all tokens on one expert.
        """
        counts = [self._expert_token_counts[i] for i in range(self.num_experts)]
        total  = sum(counts) + 1
        max_c  = max(counts) + 1
        return (max_c / total - 1.0 / self.num_experts) / (1.0 - 1.0 / self.num_experts)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_moe_cluster(
    all_nodes: list[GPUNode],
    num_experts: int = DEFAULT_NUM_EXPERTS,
    top_k: int = DEFAULT_TOP_K_EXPERTS,
) -> MoERouter:
    """
    Partition *all_nodes* into *num_experts* expert groups and return a MoERouter.

    If there are more nodes than experts, experts share nodes.
    If there are fewer nodes than experts, nodes host multiple experts.
    """
    groups: list[ExpertGroup] = []
    n = len(all_nodes)

    for eid in range(num_experts):
        # Assign nodes round-robin
        group_nodes = [all_nodes[i] for i in range(n) if i % num_experts == eid]
        if not group_nodes:
            group_nodes = [all_nodes[eid % n]]
        groups.append(ExpertGroup(expert_id=eid, nodes=group_nodes))

    return MoERouter(groups, num_experts=num_experts, top_k=top_k)
