"""
Baseline Round-Robin Scheduler.

Distributes requests evenly across all nodes in a circular fashion.
No awareness of memory pressure, KV cache state, or SLA class.
Used as the performance baseline for benchmark comparisons.
"""

from __future__ import annotations

import itertools

from control_plane.scheduler.base_scheduler import BaseScheduler, SchedulingResult
from core.models import GPUNode, InferenceRequest


class RoundRobinScheduler(BaseScheduler):
    """
    Naive round-robin: pick the next node in sequence, regardless of load.

    Strengths : simple, perfectly fair in terms of request count
    Weaknesses: ignores memory pressure → causes OOM and high tail latency
                under bursty workloads or long-context requests
    """

    def __init__(self, nodes: list[GPUNode]) -> None:
        super().__init__(nodes, name="round_robin")
        self._cycle = itertools.cycle(nodes)
        self._counter = 0

    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        # Cycle through nodes; if the chosen node is full, skip to the next
        for _ in range(len(self._nodes)):
            node = next(self._cycle)
            self._counter += 1
            if node.can_fit(request.kv_blocks_needed, request.estimated_memory_gb):
                node.allocate(request.kv_blocks_needed, request.estimated_memory_gb)
                return SchedulingResult(
                    request_id=request.request_id,
                    node_id=node.node_id,
                    accepted=True,
                )

        return SchedulingResult(
            request_id=request.request_id,
            node_id=None,
            accepted=False,
            rejection_reason="all_nodes_full",
        )

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        node.release(request.kv_blocks_needed, request.estimated_memory_gb)

    def update_nodes(self, nodes: list[GPUNode]) -> None:
        super().update_nodes(nodes)
        self._cycle = itertools.cycle(nodes)
