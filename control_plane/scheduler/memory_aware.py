"""
Memory-Aware Greedy Scheduler.

Routes each request to the node with the most available KV cache memory,
weighted by compute headroom. No SLA awareness, but significantly better
than round-robin or least-loaded under long-context workloads.

This is the "good baseline" that the KV-Aware SLA Scheduler must beat.
"""

from __future__ import annotations

from control_plane.scheduler.base_scheduler import BaseScheduler, SchedulingResult
from core.models import GPUNode, InferenceRequest


class MemoryAwareScheduler(BaseScheduler):
    """
    Greedy placement: maximise KV cache headroom on the chosen node.

    Score = α * kv_free_ratio + β * (1 - compute_util)

    Strengths : avoids OOM spills, good cache utilisation
    Weaknesses: no SLA differentiation → REALTIME and BATCH treated equally
                no prefix-cache affinity → lower cache hit rate than KVAware
    """

    def __init__(
        self,
        nodes: list[GPUNode],
        alpha: float = 0.7,   # Weight for KV headroom
        beta: float = 0.3,    # Weight for compute headroom
    ) -> None:
        super().__init__(nodes, name="memory_aware")
        self.alpha = alpha
        self.beta = beta

    def _score(self, node: GPUNode) -> float:
        kv_ratio = node.kv_blocks_free / max(1, node.kv_blocks_total)
        compute_headroom = 1.0 - node.compute_utilization
        return self.alpha * kv_ratio + self.beta * compute_headroom

    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        feasible = self._feasible_nodes(request)
        if not feasible:
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None,
                accepted=False,
                rejection_reason="no_kv_headroom",
            )

        node = max(feasible, key=self._score)
        node.allocate(request.kv_blocks_needed, request.estimated_memory_gb)
        return SchedulingResult(
            request_id=request.request_id,
            node_id=node.node_id,
            accepted=True,
        )

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        node.release(request.kv_blocks_needed, request.estimated_memory_gb)
