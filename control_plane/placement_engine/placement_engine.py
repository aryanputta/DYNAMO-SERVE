"""
Placement Engine.

Given a list of candidate GPU nodes and an InferenceRequest, the placement
engine scores each node and returns the best placement decision.

Scoring considers:
  - KV cache headroom (blocks available)
  - Prefix cache affinity (prefer nodes that already hold the prefix)
  - Memory pressure (avoid hot nodes)
  - Compute utilization (avoid overloaded nodes)
  - NVLink topology (penalise cross-node memory transfers)
  - SLA class urgency (REALTIME requests get lower latency nodes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.models import GPUNode, GPUType, InferenceRequest, SLAClass

logger = logging.getLogger(__name__)

# NVLink transfer penalty (ms) added to estimated TTFT for non-NVLink nodes
# when the request has a long prompt (> 4096 tokens)
NVLINK_LONG_CONTEXT_PENALTY_MS = 80.0

# Weights for the scoring function
W_KV_HEADROOM = 0.40
W_MEMORY_PRESSURE = 0.25
W_COMPUTE_UTIL = 0.20
W_PREFIX_AFFINITY = 0.15


@dataclass
class PlacementDecision:
    node_id: str
    score: float
    estimated_ttft_ms: float
    reason: str
    nvlink_path: bool = False


class PlacementEngine:
    """
    Scores candidate GPU nodes for an incoming inference request.

    The engine is intentionally stateless w.r.t. active placements –
    all state is read from GPUNode objects passed in at call time.
    """

    def __init__(
        self,
        kv_manager=None,   # KVCacheManager, optional dependency
        latency_predictor=None,   # Optional ML predictor
    ) -> None:
        self._kv_manager = kv_manager
        self._latency_predictor = latency_predictor

    def place(
        self,
        request: InferenceRequest,
        nodes: list[GPUNode],
    ) -> Optional[PlacementDecision]:
        """
        Return the best placement decision for *request* from *nodes*.

        Returns None if no node can fit the request.
        """
        if not nodes:
            return None

        decisions = []
        for node in nodes:
            d = self._score_node(request, node)
            if d is not None:
                decisions.append(d)

        if not decisions:
            logger.debug(
                "PlacementEngine: no feasible node for request %s (%d kv blocks needed)",
                request.request_id,
                request.kv_blocks_needed,
            )
            return None

        # Higher score = better; pick max
        best = max(decisions, key=lambda d: d.score)
        logger.debug(
            "PlacementEngine: placed %s on %s (score=%.3f, ttft=%.1fms)",
            request.request_id,
            best.node_id,
            best.score,
            best.estimated_ttft_ms,
        )
        return best

    def _score_node(
        self, request: InferenceRequest, node: GPUNode
    ) -> Optional[PlacementDecision]:
        """Score *node* for *request*. Returns None if infeasible."""
        # Feasibility check
        if not node.can_fit(request.kv_blocks_needed, request.estimated_memory_gb):
            return None

        # ── Component scores (each 0–1, higher = better) ──

        # 1. KV headroom: prefer nodes with lots of free blocks
        kv_headroom_ratio = node.kv_blocks_free / max(1, node.kv_blocks_total)

        # 2. Memory pressure: lower is better
        mem_score = 1.0 - node.memory_pressure

        # 3. Compute utilization: lower is better
        compute_score = 1.0 - node.compute_utilization

        # 4. Prefix affinity
        prefix_affinity = 0.0
        if self._kv_manager and request.prefix_hash:
            stats = self._kv_manager.get_stats(node.node_id)
            # Rough proxy: if the node has good hit rate, the prefix may be cached
            prefix_affinity = stats.hit_rate

        # 5. NVLink bonus for long-context requests
        nvlink_bonus = 0.0
        if node.nvlink_enabled and request.is_long_context:
            nvlink_bonus = 0.1

        # ── Weighted aggregate ──
        score = (
            W_KV_HEADROOM * kv_headroom_ratio
            + W_MEMORY_PRESSURE * mem_score
            + W_COMPUTE_UTIL * compute_score
            + W_PREFIX_AFFINITY * prefix_affinity
            + nvlink_bonus
        )

        # ── Estimated TTFT ──
        if self._latency_predictor is not None:
            est_ttft = self._latency_predictor.predict_ttft(request, node)
        else:
            est_ttft = self._heuristic_ttft(request, node)

        # For REALTIME requests, boost nodes with low TTFT estimates
        if request.sla_class == SLAClass.REALTIME:
            ttft_score = max(0.0, 1.0 - est_ttft / 500.0)
            score += 0.2 * ttft_score

        return PlacementDecision(
            node_id=node.node_id,
            score=score,
            estimated_ttft_ms=est_ttft,
            reason="scored",
            nvlink_path=node.nvlink_enabled,
        )

    def _heuristic_ttft(self, request: InferenceRequest, node: GPUNode) -> float:
        """
        Approximate TTFT estimate without ML predictor.

        Formula: prefill_time + queue_delay + transfer_penalty

        - prefill_time: proportional to prompt_tokens / (bandwidth * occupancy)
        - queue_delay: proportional to active_requests
        """
        # tokens / (GB/s * 1e9 / bytes_per_token)
        bytes_per_token = 2 * 4096 * 2   # fp16, d_model=4096
        transfer_time_ms = (
            request.prompt_tokens * bytes_per_token
            / (node.memory_bandwidth_gbps * 1e9)
            * 1000.0
        )
        queue_delay_ms = node.active_requests * 20.0   # 20ms per active request
        nvlink_penalty = 0.0
        if not node.nvlink_enabled and request.is_long_context:
            nvlink_penalty = NVLINK_LONG_CONTEXT_PENALTY_MS

        return transfer_time_ms + queue_delay_ms + nvlink_penalty

    def rank_nodes(
        self,
        request: InferenceRequest,
        nodes: list[GPUNode],
    ) -> list[PlacementDecision]:
        """Return all feasible nodes sorted by score descending."""
        decisions = [self._score_node(request, n) for n in nodes]
        valid = [d for d in decisions if d is not None]
        return sorted(valid, key=lambda d: d.score, reverse=True)
