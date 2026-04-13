"""
Admission Controller.

Decides whether an incoming request should be:
  ACCEPT  – route to a GPU node
  QUEUE   – defer until resources free up (BATCH class)
  REJECT  – return an overload error immediately

Two modes:
  1. Threshold-based (default): static utilization thresholds
  2. ML-guided: uses a trained spill-risk model to predict whether
     accepting the request would cause SLA violations

The admission controller wraps the SLA policy and KV cache manager
so it can make informed decisions about memory headroom.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

from control_plane.sla_policy.sla_policy import SLAPolicy
from core.models import GPUNode, InferenceRequest, SLAClass

if TYPE_CHECKING:
    from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
    from ml.spill_risk_model.spill_risk_model import SpillRiskModel

logger = logging.getLogger(__name__)


class AdmissionDecision(str, Enum):
    ACCEPT = "accept"
    QUEUE = "queue"
    REJECT = "reject"


class AdmissionController:
    """
    Gate between the network edge and the scheduling layer.

    The controller evaluates each request against cluster capacity,
    SLA budgets, and (optionally) an ML model that predicts memory-spill risk.
    """

    def __init__(
        self,
        nodes: list[GPUNode],
        kv_manager: "KVCacheManager",
        sla_policy: SLAPolicy,
        max_queue_depth: int = 512,
        use_ml_model: bool = False,
        spill_model: Optional["SpillRiskModel"] = None,
        # Thresholds (used in threshold mode)
        memory_reject_threshold: float = 0.95,
        memory_queue_threshold: float = 0.85,
        compute_reject_threshold: float = 0.95,
    ) -> None:
        self._nodes = {n.node_id: n for n in nodes}
        self._kv_manager = kv_manager
        self._sla_policy = sla_policy
        self._max_queue = max_queue_depth
        self._use_ml = use_ml_model
        self._spill_model = spill_model

        self._mem_reject_thr = memory_reject_threshold
        self._mem_queue_thr = memory_queue_threshold
        self._compute_reject_thr = compute_reject_threshold

        self._queue: list[InferenceRequest] = []
        self._total_accepted = 0
        self._total_queued = 0
        self._total_rejected = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, request: InferenceRequest) -> tuple[AdmissionDecision, str]:
        """
        Make an admission decision for *request*.

        Returns (decision, reason).
        """
        cluster_load = self._cluster_load()
        mem_pressure = self._cluster_memory_pressure()

        # 1. Check SLA policy veto
        reject, reason = self._sla_policy.should_reject_at_load(
            request, cluster_load, mem_pressure
        )
        if reject:
            self._total_rejected += 1
            logger.debug("REJECT %s: %s", request.request_id, reason)
            return AdmissionDecision.REJECT, reason

        # 2. OOM protection: hard reject if memory is critically full
        if mem_pressure >= self._mem_reject_thr:
            self._total_rejected += 1
            reason = f"cluster_memory_pressure={mem_pressure:.2f}"
            logger.debug("REJECT %s: %s", request.request_id, reason)
            return AdmissionDecision.REJECT, reason

        # 3. ML-guided spill risk prediction
        if self._use_ml and self._spill_model is not None:
            spill_risk = self._spill_model.predict_risk(request, list(self._nodes.values()))
            if spill_risk > 0.85:
                if request.sla_class == SLAClass.BATCH:
                    self._enqueue(request)
                    return AdmissionDecision.QUEUE, f"ml_spill_risk={spill_risk:.2f}"
                else:
                    self._total_rejected += 1
                    return AdmissionDecision.REJECT, f"ml_spill_risk={spill_risk:.2f}"

        # 4. Memory pressure: queue BATCH, reject others
        if mem_pressure >= self._mem_queue_thr:
            if request.sla_class == SLAClass.BATCH:
                self._enqueue(request)
                return AdmissionDecision.QUEUE, f"memory_pressure={mem_pressure:.2f}"
            if cluster_load >= self._compute_reject_thr:
                self._total_rejected += 1
                return AdmissionDecision.REJECT, "compute_saturated"

        # 5. Check individual node KV headroom
        blocks_needed = request.kv_blocks_needed
        has_node = any(
            self._kv_manager.free_blocks(nid) >= blocks_needed
            for nid in self._nodes
        )
        if not has_node:
            if request.sla_class == SLAClass.BATCH:
                self._enqueue(request)
                return AdmissionDecision.QUEUE, "no_kv_headroom"
            self._total_rejected += 1
            return AdmissionDecision.REJECT, "no_kv_headroom"

        self._total_accepted += 1
        return AdmissionDecision.ACCEPT, "ok"

    def pop_queued(self, n: int = 16) -> list[InferenceRequest]:
        """Return up to *n* queued requests that can now be scheduled."""
        released: list[InferenceRequest] = []
        remaining: list[InferenceRequest] = []
        for req in self._queue:
            if (
                len(released) < n
                and self._cluster_memory_pressure() < self._mem_queue_thr
            ):
                released.append(req)
                self._total_accepted += 1
            else:
                remaining.append(req)
        self._queue = remaining
        return released

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, request: InferenceRequest) -> None:
        if len(self._queue) >= self._max_queue:
            self._total_rejected += 1
            logger.warning("Admission queue full, dropping %s", request.request_id)
            return
        self._queue.append(request)
        self._total_queued += 1

    def _cluster_load(self) -> float:
        if not self._nodes:
            return 0.0
        return sum(n.compute_utilization for n in self._nodes.values()) / len(self._nodes)

    def _cluster_memory_pressure(self) -> float:
        if not self._nodes:
            return 0.0
        return sum(n.memory_pressure for n in self._nodes.values()) / len(self._nodes)

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict:
        return {
            "total_accepted": self._total_accepted,
            "total_queued": self._total_queued,
            "total_rejected": self._total_rejected,
            "current_queue_depth": self.queue_depth,
        }
