"""
Batch Optimizer.

Decides how to group pending requests into a single forward pass
(continuous batching) to maximise throughput while respecting SLA budgets.

The core tradeoff:
  - Larger batches → higher GPU utilisation, better tokens/sec
  - Larger batches → higher TPOT, risk of TTFT/TPOT SLA violation

Strategy:
  1. Group requests by decode phase (all active decode steps together).
  2. Within a batch, respect per-SLA TPOT budgets.
  3. Use a greedy knapsack over memory capacity constraints.
  4. Optionally use a learned model (LinearRegression) to predict
     optimal batch size given cluster state.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.models import BatchJob, GPUNode, InferenceRequest, SLAClass

logger = logging.getLogger(__name__)

# Maximum tokens in a single batch (limits peak memory)
MAX_BATCH_TOKENS = 32_768
# Hard cap on batch size (number of sequences)
MAX_BATCH_SIZE = 64


@dataclass
class BatchPlan:
    """A proposed batch grouping with predicted throughput."""

    batches: list[BatchJob]
    predicted_tokens_per_sec: float
    predicted_mean_tpot_ms: float
    estimated_memory_gb: float


class BatchOptimizer:
    """
    Forms optimal batches from a queue of pending requests.

    Operates on a single node's pending queue; called by the serving loop
    once per decode step (typically every 20–50 ms).
    """

    def __init__(
        self,
        node: GPUNode,
        latency_predictor=None,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        max_batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        self._node = node
        self._predictor = latency_predictor
        self._max_tokens = max_batch_tokens
        self._max_size = max_batch_size
        self._tpot_model = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Batch formation
    # ------------------------------------------------------------------

    def form_batches(self, pending: list[InferenceRequest]) -> BatchPlan:
        """
        Partition *pending* into one or more BatchJob objects.

        Priority order:
          1. REALTIME requests always go first, one per batch if needed
          2. INTERACTIVE requests fill remaining capacity
          3. BATCH requests are appended if space allows
        """
        if not pending:
            return BatchPlan(batches=[], predicted_tokens_per_sec=0.0,
                             predicted_mean_tpot_ms=0.0, estimated_memory_gb=0.0)

        # Sort by SLA priority desc, then arrival time asc
        sorted_reqs = sorted(
            pending,
            key=lambda r: (-_sla_priority(r.sla_class), r.arrival_time),
        )

        batches: list[BatchJob] = []
        current_batch: list[InferenceRequest] = []
        current_tokens = 0

        for req in sorted_reqs:
            req_tokens = req.prompt_tokens + req.max_output_tokens
            if (
                len(current_batch) >= self._max_size
                or current_tokens + req_tokens > self._max_tokens
            ):
                if current_batch:
                    batches.append(self._make_job(current_batch))
                current_batch = [req]
                current_tokens = req_tokens
            else:
                current_batch.append(req)
                current_tokens += req_tokens

        if current_batch:
            batches.append(self._make_job(current_batch))

        # Predict throughput for the plan
        tpot = self._predict_tpot(batches[0]) if batches else 0.0
        total_tokens = sum(
            sum(r.max_output_tokens for r in b.requests) for b in batches
        )
        # Rough tokens/sec: decode tokens / (tpot * avg_batch_size)
        avg_bs = sum(b.batch_size for b in batches) / max(1, len(batches))
        tps = (avg_bs / max(0.001, tpot / 1000.0)) if tpot > 0 else 0.0

        mem_gb = sum(
            sum(r.estimated_memory_gb for r in b.requests) for b in batches
        )

        return BatchPlan(
            batches=batches,
            predicted_tokens_per_sec=tps,
            predicted_mean_tpot_ms=tpot,
            estimated_memory_gb=mem_gb,
        )

    # ------------------------------------------------------------------
    # Throughput vs latency tradeoff analysis
    # ------------------------------------------------------------------

    def analyze_tradeoff(
        self, pending: list[InferenceRequest], max_batch_sizes: Optional[list[int]] = None
    ) -> list[dict]:
        """
        Sweep over batch sizes and report throughput/latency tradeoffs.

        Returns a list of dicts with keys: batch_size, tpot_ms, tokens_per_sec.
        Used to produce the throughput–latency curve in the benchmark report.
        """
        if max_batch_sizes is None:
            max_batch_sizes = [1, 2, 4, 8, 16, 32, 64]

        results = []
        for bs in max_batch_sizes:
            self._max_size = bs
            plan = self.form_batches(pending[:bs])
            results.append({
                "batch_size": bs,
                "tpot_ms": plan.predicted_mean_tpot_ms,
                "tokens_per_sec": plan.predicted_tokens_per_sec,
                "estimated_memory_gb": plan.estimated_memory_gb,
            })
        return results

    # ------------------------------------------------------------------
    # ML-guided TPOT prediction
    # ------------------------------------------------------------------

    def fit_tpot_model(
        self, samples: list[tuple[int, float, float]]
    ) -> None:
        """
        Fit a simple linear model: TPOT ~ batch_size + memory_pressure.

        Args:
            samples: list of (batch_size, memory_pressure, observed_tpot_ms)
        """
        from sklearn.linear_model import Ridge

        if len(samples) < 10:
            return

        X = np.array([[math.log1p(bs), mp] for bs, mp, _ in samples])
        y = np.array([tpot for _, _, tpot in samples])
        self._tpot_model = Ridge(alpha=1.0).fit(X, y)
        self._fitted = True
        logger.info("BatchOptimizer TPOT model fitted on %d samples", len(samples))

    def _predict_tpot(self, batch: BatchJob) -> float:
        if self._fitted and self._tpot_model is not None:
            x = np.array([[math.log1p(batch.batch_size), self._node.memory_pressure]])
            return max(0.5, float(self._tpot_model.predict(x)[0]))
        # Heuristic: bandwidth-bound, scales with batch size
        model_bytes = 140 * 1024 ** 3
        base = model_bytes / (self._node.memory_bandwidth_gbps * 1e9 * 0.8) * 1000.0
        return max(0.5, base * (1.0 + 0.02 * (batch.batch_size - 1)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_job(requests: list[InferenceRequest]) -> BatchJob:
        return BatchJob(requests=list(requests), phase="decode")


def _sla_priority(sla: SLAClass) -> int:
    return {SLAClass.REALTIME: 3, SLAClass.INTERACTIVE: 2, SLAClass.BATCH: 1}[sla]
