"""
Contention Model.

Captures how multiple co-located requests interfere with each other:
  - Memory bandwidth contention (decode phase)
  - KV cache thrashing (frequent eviction under high utilisation)
  - Compute preemption overhead (context switches between requests)
  - NVLink saturation (for tensor-parallel workloads)

The model produces a latency multiplier that is applied to baseline TTFT/TPOT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.models import GPUNode


@dataclass
class ContentionFactors:
    """Per-request latency multipliers due to resource contention."""

    compute_multiplier: float = 1.0     # > 1 = slower due to compute sharing
    bandwidth_multiplier: float = 1.0   # > 1 = slower due to HBM bandwidth sharing
    kv_eviction_penalty_ms: float = 0.0 # Extra latency from KV eviction
    nvlink_contention: float = 1.0      # > 1 on saturated NVLink


class ContentionModel:
    """
    Computes per-request contention factors based on node state.

    The model is calibrated to match empirical observations from vLLM
    benchmarks on H100 hardware (approximate, not exact).
    """

    # Bandwidth saturation curve parameters
    # f(u) = 1 + alpha * u^beta, where u = utilisation (0–1)
    BW_ALPHA = 0.8
    BW_BETA = 2.5

    # Compute sharing: linear above 50 % utilisation
    COMPUTE_ALPHA = 0.4

    # KV eviction overhead per evicted block (ms)
    EVICTION_MS_PER_BLOCK = 0.5

    # NVLink saturation threshold
    NVLINK_SAT_THRESHOLD = 0.80

    def compute(
        self,
        node: GPUNode,
        active_requests: int,
        kv_evictions_this_step: int = 0,
        nvlink_utilisation: float = 0.0,
    ) -> ContentionFactors:
        """
        Compute contention factors for a request running on *node*.

        Args:
            node:                   The GPU node.
            active_requests:        Number of requests currently in decode phase.
            kv_evictions_this_step: KV blocks evicted to fit this request.
            nvlink_utilisation:     Fraction of NVLink bandwidth in use (0–1).
        """
        # ── Memory bandwidth contention ──
        # More decode-phase requests → more HBM reads → bandwidth bottleneck
        bw_util = min(1.0, active_requests / 20.0)   # saturates at 20 concurrent
        bw_mult = 1.0 + self.BW_ALPHA * (bw_util ** self.BW_BETA)

        # ── Compute contention ──
        compute_util = node.compute_utilization
        if compute_util > 0.5:
            compute_mult = 1.0 + self.COMPUTE_ALPHA * (compute_util - 0.5) / 0.5
        else:
            compute_mult = 1.0

        # ── KV eviction penalty ──
        eviction_penalty_ms = kv_evictions_this_step * self.EVICTION_MS_PER_BLOCK

        # ── NVLink contention ──
        nvlink_mult = 1.0
        if node.nvlink_enabled and nvlink_utilisation > self.NVLINK_SAT_THRESHOLD:
            overshoot = (nvlink_utilisation - self.NVLINK_SAT_THRESHOLD) / (1.0 - self.NVLINK_SAT_THRESHOLD)
            nvlink_mult = 1.0 + 0.3 * overshoot

        return ContentionFactors(
            compute_multiplier=compute_mult,
            bandwidth_multiplier=bw_mult,
            kv_eviction_penalty_ms=eviction_penalty_ms,
            nvlink_contention=nvlink_mult,
        )

    def apply(
        self,
        factors: ContentionFactors,
        ttft_ms: float,
        tpot_ms: float,
    ) -> tuple[float, float]:
        """Apply contention factors to baseline TTFT and TPOT estimates."""
        adjusted_ttft = ttft_ms * factors.compute_multiplier + factors.kv_eviction_penalty_ms
        adjusted_tpot = tpot_ms * factors.bandwidth_multiplier * factors.nvlink_contention
        return adjusted_ttft, adjusted_tpot
