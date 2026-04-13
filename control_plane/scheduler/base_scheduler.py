"""
Abstract base class for all schedulers.

Every scheduler must implement `schedule()` which maps an InferenceRequest
to a GPUNode (or None to reject). Subclasses share the same interface so the
benchmark harness can swap them without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from core.models import GPUNode, InferenceRequest


@dataclass
class SchedulingResult:
    """Outcome of a single scheduling call."""

    request_id: str
    node_id: Optional[str]   # None = rejected / deferred
    accepted: bool
    rejection_reason: str = ""
    estimated_ttft_ms: float = 0.0

    @property
    def rejected(self) -> bool:
        return not self.accepted


class BaseScheduler(ABC):
    """
    Interface contract for all scheduling policies.

    Lifecycle:
        scheduler.schedule(request) -> SchedulingResult
        # ... runtime executes the request on the assigned node ...
        scheduler.on_complete(request, node)  # update internal state
    """

    def __init__(self, nodes: list[GPUNode], name: str = "base") -> None:
        self._nodes = nodes
        self._node_map: dict[str, GPUNode] = {n.node_id: n for n in nodes}
        self.name = name

    @abstractmethod
    def schedule(self, request: InferenceRequest) -> SchedulingResult:
        """Select a node for *request*. Return SchedulingResult."""

    def on_complete(self, request: InferenceRequest, node: GPUNode) -> None:
        """Called when a request finishes. Override to update scheduler state."""

    def on_fail(self, request: InferenceRequest, node: Optional[GPUNode]) -> None:
        """Called when a request fails or is preempted."""

    def update_nodes(self, nodes: list[GPUNode]) -> None:
        """Replace the node list (used when cluster topology changes)."""
        self._nodes = nodes
        self._node_map = {n.node_id: n for n in nodes}

    def _feasible_nodes(self, request: InferenceRequest) -> list[GPUNode]:
        return [
            n for n in self._nodes
            if n.can_fit(request.kv_blocks_needed, request.estimated_memory_gb)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={len(self._nodes)})"
