"""
SLA Policy engine.

Maps SLA classes to concrete latency budgets and drives enforcement:
  - Admission gate: is there enough headroom to serve this request within SLA?
  - Preemption trigger: which running request should yield if resources are scarce?
  - Cost accounting: penalize SLA violations in the reporting layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.models import InferenceRequest, RequestResult, SLAClass, TenantPriority


@dataclass
class SLABudget:
    """Latency budget for a single SLA class (all values in milliseconds)."""

    sla_class: SLAClass
    max_ttft_ms: float          # Max acceptable time-to-first-token
    max_tpot_ms: float          # Max acceptable time-per-output-token
    max_queue_wait_ms: float    # Max time a request can sit in the queue
    max_total_latency_ms: float
    admission_reject_p: float   # Probability of admission rejection when loaded (0–1)

    def is_ttft_met(self, ttft_ms: float) -> bool:
        return ttft_ms <= self.max_ttft_ms

    def is_tpot_met(self, tpot_ms: float) -> bool:
        return tpot_ms <= self.max_tpot_ms

    def is_met(self, result: RequestResult) -> bool:
        return (
            self.is_ttft_met(result.ttft_ms)
            and self.is_tpot_met(result.tpot_ms)
            and result.total_latency_ms <= self.max_total_latency_ms
        )


# Default budgets per SLA class
DEFAULT_BUDGETS: dict[SLAClass, SLABudget] = {
    SLAClass.REALTIME: SLABudget(
        sla_class=SLAClass.REALTIME,
        max_ttft_ms=200.0,
        max_tpot_ms=50.0,
        max_queue_wait_ms=50.0,
        max_total_latency_ms=5_000.0,
        admission_reject_p=0.95,
    ),
    SLAClass.INTERACTIVE: SLABudget(
        sla_class=SLAClass.INTERACTIVE,
        max_ttft_ms=1_000.0,
        max_tpot_ms=100.0,
        max_queue_wait_ms=500.0,
        max_total_latency_ms=30_000.0,
        admission_reject_p=0.70,
    ),
    SLAClass.BATCH: SLABudget(
        sla_class=SLAClass.BATCH,
        max_ttft_ms=30_000.0,
        max_tpot_ms=500.0,
        max_queue_wait_ms=60_000.0,
        max_total_latency_ms=300_000.0,
        admission_reject_p=0.0,
    ),
}

# Cost per SLA violation (arbitrary units, used for cost accounting)
SLA_VIOLATION_PENALTY: dict[SLAClass, float] = {
    SLAClass.REALTIME: 10.0,
    SLAClass.INTERACTIVE: 3.0,
    SLAClass.BATCH: 0.5,
}


class SLAPolicy:
    """
    Central SLA policy registry and enforcement helper.

    The scheduler, admission controller, and benchmark harness all call
    into this class to check budgets and record violations.
    """

    def __init__(self, budgets: Optional[dict[SLAClass, SLABudget]] = None) -> None:
        self._budgets = budgets or DEFAULT_BUDGETS
        self._violation_counts: dict[SLAClass, int] = {c: 0 for c in SLAClass}
        self._violation_penalty_total: float = 0.0

    def get_budget(self, sla_class: SLAClass) -> SLABudget:
        return self._budgets[sla_class]

    def check_result(self, result: RequestResult, sla_class: SLAClass) -> bool:
        """Return True if result satisfies SLA; record violation otherwise."""
        budget = self._budgets[sla_class]
        met = budget.is_met(result)
        if not met:
            self._violation_counts[sla_class] += 1
            self._violation_penalty_total += SLA_VIOLATION_PENALTY[sla_class]
        return met

    def should_reject_at_load(
        self,
        request: InferenceRequest,
        cluster_load: float,   # 0.0–1.0
        memory_pressure: float,
    ) -> tuple[bool, str]:
        """
        Heuristic admission gate: reject a request when the cluster is overloaded
        and accepting it would likely cause SLA violations for existing requests.

        Returns (should_reject, reason).
        """
        budget = self._budgets[request.sla_class]

        # BATCH requests are never hard-rejected (they just queue)
        if request.sla_class == SLAClass.BATCH:
            return False, ""

        # Under extreme memory pressure, protect existing REALTIME requests
        if memory_pressure > 0.95 and request.sla_class == SLAClass.INTERACTIVE:
            return True, "memory_pressure_critical"

        if memory_pressure > 0.99:
            return True, "oom_risk"

        # If cluster is overloaded and the request has tight TTFT budget
        if cluster_load > 0.85 and budget.max_ttft_ms < 500.0:
            # Probabilistic rejection based on load overshoot
            overshoot = (cluster_load - 0.85) / 0.15   # 0 at 85 %, 1 at 100 %
            effective_p = budget.admission_reject_p * overshoot
            import random
            if random.random() < effective_p:
                return True, "load_shedding"

        return False, ""

    def preemption_score(self, request: InferenceRequest) -> float:
        """
        Lower score = higher preemption priority (i.e., evict this request first).

        Used when the system needs to preempt running requests to free memory.
        BATCH requests are most likely to be preempted; REALTIME least likely.
        """
        base = {
            SLAClass.REALTIME: 100.0,
            SLAClass.INTERACTIVE: 50.0,
            SLAClass.BATCH: 10.0,
        }[request.sla_class]
        return base * request.priority.value

    def violation_summary(self) -> dict:
        return {
            "violations_by_class": {k.value: v for k, v in self._violation_counts.items()},
            "total_violation_penalty": self._violation_penalty_total,
        }
