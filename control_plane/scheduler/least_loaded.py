"""
Least-Loaded Scheduler.

Routes each request to the node with the fewest active requests.
A step up from round-robin: reacts to compute load but still ignores
KV cache pressure and SLA classes.
"""

from __future__ import annotations

from control_plane.scheduler.base_scheduler import BaseScheduler, SchedulingResult
from core.models import GPUNode, InferenceRequest


class LeastLoadedScheduler(BaseScheduler):
    """
    Send each request to the node with the lowest active_requests count.

    Strengths : avoids hot-spots caused by slow requests piling up
    Weaknesses: ignores memory/KV state → still causes OOM under long context
                no prefix-cache awareness → low cache hit rate
    """

    def __init__(self, nodes: list[GPUNode]) -> None:
        super().__init__(nodes, name="least_loaded")

    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        feasible = self._feasible_nodes(request)
        if not feasible:
            return SchedulingResult(
                request_id=request.request_id,
                node_id=None,
                accepted=False,
                rejection_reason="no_feasible_node",
            )

        # Sort by (active_requests ASC, memory_pressure ASC)
        node = min(feasible, key=lambda n: (n.active_requests, n.memory_pressure))
        node.allocate(request.kv_blocks_needed, request.estimated_memory_gb)
        return SchedulingResult(
            request_id=request.request_id,
            node_id=node.node_id,
            accepted=True,
        )

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        node.release(request.kv_blocks_needed, request.estimated_memory_gb)
