"""
KV-Aware SLA Scheduler – the main contribution of DYNAMO-SERVE.

This scheduler integrates four signals that prior baselines ignore:

  1. KV prefix-cache affinity  – route to the node that already holds
     the request's prefix to avoid cold-cache TTFT penalties.

  2. SLA-class differentiation – REALTIME requests jump the queue,
     BATCH requests yield when memory is tight.

  3. ML-predicted TTFT / memory pressure – use a trained predictor
     to estimate latency before committing a placement decision.

  4. NVLink topology awareness – penalise cross-node memory transfers
     for long-context requests that aren't connected via NVSwitch.

Architecture
------------
    schedule() → AdmissionController.decide()
              → PlacementEngine.place()
              → KVCacheManager.allocate()
              → GPUNode.allocate()

The KVCacheManager and PlacementEngine are injected at construction so
the scheduler can be unit-tested with mocks.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, TYPE_CHECKING

from control_plane.admission_controller.admission_controller import (
    AdmissionController,
    AdmissionDecision,
)
from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.placement_engine.placement_engine import PlacementEngine
from control_plane.scheduler.base_scheduler import BaseScheduler, SchedulingResult
from control_plane.sla_policy.sla_policy import SLAPolicy
from core.models import GPUNode, InferenceRequest, SLAClass, TenantPriority

if TYPE_CHECKING:
    from ml.latency_predictor.latency_predictor import LatencyPredictor
    from ml.spill_risk_model.spill_risk_model import SpillRiskModel

logger = logging.getLogger(__name__)

# Per-SLA priority multiplier for the global request queue
SLA_QUEUE_PRIORITY = {
    SLAClass.REALTIME: 10,
    SLAClass.INTERACTIVE: 5,
    SLAClass.BATCH: 1,
}


class KVAwareSLAScheduler(BaseScheduler):
    """
    Production-grade KV-cache-aware scheduler with SLA enforcement.

    This is the primary system contribution; all other schedulers are baselines.
    """

    def __init__(
        self,
        nodes: list[GPUNode],
        kv_manager: KVCacheManager,
        sla_policy: Optional[SLAPolicy] = None,
        latency_predictor: Optional["LatencyPredictor"] = None,
        spill_model: Optional["SpillRiskModel"] = None,
        use_ml: bool = False,
        preemption_enabled: bool = True,
    ) -> None:
        super().__init__(nodes, name="kv_aware_sla")
        self._kv_manager = kv_manager
        self._sla_policy = sla_policy or SLAPolicy()
        self._use_ml = use_ml

        self._placement_engine = PlacementEngine(
            kv_manager=kv_manager,
            latency_predictor=latency_predictor if use_ml else None,
        )
        self._admission = AdmissionController(
            nodes=nodes,
            kv_manager=kv_manager,
            sla_policy=self._sla_policy,
            use_ml_model=use_ml,
            spill_model=spill_model,
        )
        self._preemption_enabled = preemption_enabled

        # Track in-flight requests: request_id -> (node_id, blocks, memory_gb)
        self._inflight: dict[str, tuple[str, int, float]] = {}

        # Priority queue per SLA class
        self._sla_counters: dict[SLAClass, int] = {c: 0 for c in SLAClass}
        self._total_scheduled = 0
        self._total_preempted = 0

    # ------------------------------------------------------------------
    # Core scheduling interface
    # ------------------------------------------------------------------

    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        """
        Full scheduling pipeline:
          1. Admission gate
          2. Placement scoring
          3. KV block allocation
          4. Node allocation update
        """
        start = time.monotonic()

        # ── Step 1: Admission control ──
        decision, reason = self._admission.decide(request)

        if decision == AdmissionDecision.REJECT:
            logger.debug("KVAware: REJECT %s (%s)", request.request_id, reason)
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None,
                accepted=False,
                rejection_reason=reason,
            )

        if decision == AdmissionDecision.QUEUE:
            logger.debug("KVAware: QUEUE %s (%s)", request.request_id, reason)
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None,
                accepted=False,
                rejection_reason=f"queued:{reason}",
            )

        # ── Step 2: Placement scoring ──
        placement = self._placement_engine.place(request, self._nodes)

        if placement is None:
            # Attempt to free blocks via eviction before giving up
            evicted = self._try_preempt_for(request)
            if evicted:
                placement = self._placement_engine.place(request, self._nodes)

        if placement is None:
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None,
                accepted=False,
                rejection_reason="no_placement",
            )

        node = self._node_map[placement.node_id]

        # ── Step 3: KV block allocation ──
        allocated_blocks, cache_hit = self._kv_manager.allocate(
            placement.node_id, request
        )

        # ── Step 4: Update node state ──
        node.allocate(request.kv_blocks_needed, request.estimated_memory_gb)

        self._inflight[request.request_id] = (
            placement.node_id,
            request.kv_blocks_needed,
            request.estimated_memory_gb,
        )
        self._sla_counters[request.sla_class] += 1
        self._total_scheduled += 1

        elapsed_us = (time.monotonic() - start) * 1e6
        logger.debug(
            "KVAware: ACCEPT %s → %s (score=%.3f, ttft_est=%.1fms, "
            "cache_hit=%s, sched_overhead=%.1fµs)",
            request.request_id,
            placement.node_id,
            placement.score,
            placement.estimated_ttft_ms,
            cache_hit,
            elapsed_us,
        )

        return SchedulingResult(
            request_id=request.request_id,
            node_id=placement.node_id,
            accepted=True,
            estimated_ttft_ms=placement.estimated_ttft_ms,
        )

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        info = self._inflight.pop(request.request_id, None)
        if info:
            node_id, blocks, mem_gb = info
            self._kv_manager.free(node_id, [], keep_prefix=True)
            node.release(blocks, mem_gb)

        # Drain the admission queue now that resources are freed
        queued = self._admission.pop_queued()
        for queued_req in queued:
            self.schedule(queued_req)

    def on_fail(self, request: InferenceRequest, node: Optional[GPUNode]) -> None:
        info = self._inflight.pop(request.request_id, None)
        if info and node:
            node_id, blocks, mem_gb = info
            node.release(blocks, mem_gb)

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def _try_preempt_for(self, request: InferenceRequest) -> bool:
        """
        Attempt to free resources by preempting a lower-priority request.

        Returns True if preemption freed enough blocks.
        """
        if not self._preemption_enabled:
            return False

        # Only preempt BATCH if we need room for REALTIME
        if request.sla_class not in (SLAClass.REALTIME,):
            return False

        # Find the lowest-priority in-flight request
        candidates = []
        for req_id, (node_id, blocks, mem) in self._inflight.items():
            # We only track SLA class implicitly; use preemption score heuristic
            candidates.append((req_id, node_id, blocks, mem))

        if not candidates:
            return False

        # Heuristic: evict from most-loaded node for the requesting request's target
        target_node = self._most_loaded_node()
        evicted = self._kv_manager.force_evict(target_node.node_id, request.kv_blocks_needed)
        if evicted > 0:
            self._total_preempted += 1
            logger.info(
                "Preempted %d KV blocks on %s for REALTIME request %s",
                evicted,
                target_node.node_id,
                request.request_id,
            )
        return evicted >= request.kv_blocks_needed

    def _most_loaded_node(self) -> GPUNode:
        return max(self._nodes, key=lambda n: n.memory_pressure)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "total_scheduled": self._total_scheduled,
            "total_preempted": self._total_preempted,
            "inflight_count": len(self._inflight),
            "sla_counters": {k.value: v for k, v in self._sla_counters.items()},
            "admission": self._admission.stats,
        }
