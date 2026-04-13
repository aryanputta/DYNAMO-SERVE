"""
Workload Generator.

Produces synthetic LLM inference request traces with configurable:
  - Arrival process (Poisson / bursty / diurnal)
  - Prompt length distribution (short / long / mixed)
  - Tenant mix and SLA class proportions
  - Prefix sharing probability (for cache-hit rate testing)
  - MoE routing overhead (for MoE-specific workloads)

All generators return lists of InferenceRequest objects.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Optional

from core.models import InferenceRequest, SLAClass, TenantPriority


@dataclass
class WorkloadProfile:
    """
    Parameterises a synthetic workload trace.

    Arrival rates are requests per second (Poisson λ).
    Token lengths follow log-normal distributions specified by (mean, std).
    """

    name: str = "default"
    duration_s: float = 60.0           # Total simulation window
    arrival_rate_rps: float = 10.0     # Mean requests/sec (Poisson)
    burst_factor: float = 1.0          # Multiplier during burst windows
    burst_duration_s: float = 0.0      # Duration of burst windows (0 = none)
    burst_interval_s: float = 30.0     # How often bursts occur

    # Token length distributions (log-normal mu/sigma)
    prompt_len_mean: int = 512
    prompt_len_std: int = 256
    output_len_mean: int = 256
    output_len_std: int = 128
    max_prompt_tokens: int = 32_768
    max_output_tokens: int = 4_096

    # SLA class distribution (must sum to 1.0)
    sla_realtime_fraction: float = 0.15
    sla_interactive_fraction: float = 0.65
    sla_batch_fraction: float = 0.20

    # Multi-tenancy
    num_tenants: int = 5
    tenant_skew: float = 1.5    # Zipf exponent; 1.0 = uniform, >1 = hot tenants

    # Prefix sharing: fraction of requests that share a common prefix
    prefix_sharing_prob: float = 0.30
    num_shared_prefixes: int = 10

    # Model
    model_id: str = "llama-3-70b"


# Pre-defined profiles for benchmark scenarios
PROFILES: dict[str, WorkloadProfile] = {
    "baseline": WorkloadProfile(
        name="baseline",
        arrival_rate_rps=5.0,
        duration_s=60.0,
    ),
    "long_context": WorkloadProfile(
        name="long_context",
        arrival_rate_rps=3.0,
        duration_s=120.0,
        prompt_len_mean=16_384,
        prompt_len_std=8_192,
        output_len_mean=512,
        max_prompt_tokens=32_768,
        sla_realtime_fraction=0.05,
        sla_interactive_fraction=0.70,
        sla_batch_fraction=0.25,
    ),
    "burst_traffic": WorkloadProfile(
        name="burst_traffic",
        arrival_rate_rps=8.0,
        duration_s=120.0,
        burst_factor=5.0,
        burst_duration_s=10.0,
        burst_interval_s=30.0,
    ),
    "multi_tenant": WorkloadProfile(
        name="multi_tenant",
        arrival_rate_rps=15.0,
        duration_s=120.0,
        num_tenants=20,
        tenant_skew=2.0,
        sla_realtime_fraction=0.20,
        sla_interactive_fraction=0.50,
        sla_batch_fraction=0.30,
    ),
    "moe_serving": WorkloadProfile(
        name="moe_serving",
        arrival_rate_rps=6.0,
        duration_s=120.0,
        model_id="mixtral-8x7b",
        prompt_len_mean=1024,
        output_len_mean=512,
        sla_realtime_fraction=0.10,
        sla_interactive_fraction=0.60,
        sla_batch_fraction=0.30,
    ),
    "high_load": WorkloadProfile(
        name="high_load",
        arrival_rate_rps=20.0,
        duration_s=60.0,
        burst_factor=3.0,
        burst_duration_s=15.0,
        burst_interval_s=20.0,
        num_tenants=10,
    ),
}


class WorkloadGenerator:
    """
    Generates InferenceRequest traces from a WorkloadProfile.

    The generator produces a flat list of requests sorted by arrival_time.
    The benchmark harness replays them in order.
    """

    def __init__(self, profile: WorkloadProfile, seed: Optional[int] = 42) -> None:
        self.profile = profile
        self._rng = random.Random(seed)
        self._shared_prefixes = self._generate_shared_prefixes()

    def generate(self) -> list[InferenceRequest]:
        """Generate the full trace for the configured profile."""
        requests: list[InferenceRequest] = []
        t = 0.0
        req_count = 0

        while t < self.profile.duration_s:
            # Compute effective arrival rate (may be boosted during burst)
            rate = self._effective_rate(t)

            # Inter-arrival time from Poisson process
            iat = self._rng.expovariate(rate)
            t += iat
            if t >= self.profile.duration_s:
                break

            req = self._make_request(t, req_count)
            requests.append(req)
            req_count += 1

        # Sort by arrival time (should already be sorted, but be safe)
        requests.sort(key=lambda r: r.arrival_time)
        return requests

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_rate(self, t: float) -> float:
        p = self.profile
        rate = p.arrival_rate_rps
        if p.burst_duration_s > 0:
            cycle = p.burst_interval_s + p.burst_duration_s
            phase = t % cycle
            if phase >= p.burst_interval_s:
                rate *= p.burst_factor
        return max(0.01, rate)

    def _make_request(self, arrival_time: float, seq: int) -> InferenceRequest:
        p = self.profile

        # Token lengths (log-normal)
        prompt_tokens = self._sample_lognormal(p.prompt_len_mean, p.prompt_len_std)
        prompt_tokens = max(16, min(prompt_tokens, p.max_prompt_tokens))

        output_tokens = self._sample_lognormal(p.output_len_mean, p.output_len_std)
        output_tokens = max(1, min(output_tokens, p.max_output_tokens))

        # SLA class
        sla_class = self._sample_sla_class()

        # Tenant (Zipf distribution)
        tenant_id = self._sample_tenant()

        # Priority based on SLA class
        priority = {
            SLAClass.REALTIME: TenantPriority.HIGH,
            SLAClass.INTERACTIVE: TenantPriority.NORMAL,
            SLAClass.BATCH: TenantPriority.LOW,
        }[sla_class]

        # Prefix sharing
        prefix_hash = None
        if self._rng.random() < p.prefix_sharing_prob and self._shared_prefixes:
            prefix_hash = self._rng.choice(self._shared_prefixes)

        req = InferenceRequest(
            prompt_tokens=prompt_tokens,
            max_output_tokens=output_tokens,
            sla_class=sla_class,
            tenant_id=tenant_id,
            priority=priority,
            model_id=p.model_id,
            prefix_hash=prefix_hash,
        )
        req.arrival_time = arrival_time
        return req

    def _sample_lognormal(self, mean: int, std: int) -> int:
        if std <= 0:
            return mean
        mu = math.log(mean ** 2 / math.sqrt(mean ** 2 + std ** 2))
        sigma = math.sqrt(math.log(1 + (std / mean) ** 2))
        return max(1, int(self._rng.lognormvariate(mu, sigma)))

    def _sample_sla_class(self) -> SLAClass:
        p = self.profile
        roll = self._rng.random()
        if roll < p.sla_realtime_fraction:
            return SLAClass.REALTIME
        if roll < p.sla_realtime_fraction + p.sla_interactive_fraction:
            return SLAClass.INTERACTIVE
        return SLAClass.BATCH

    def _sample_tenant(self) -> str:
        n = self.profile.num_tenants
        skew = self.profile.tenant_skew
        # Zipf: P(rank k) ∝ 1/k^skew
        weights = [1.0 / (i ** skew) for i in range(1, n + 1)]
        total = sum(weights)
        weights = [w / total for w in weights]
        roll = self._rng.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if roll <= cumulative:
                return f"tenant_{i:02d}"
        return f"tenant_{n - 1:02d}"

    def _generate_shared_prefixes(self) -> list[str]:
        """Pre-compute a fixed set of prefix hashes for prefix-sharing workloads."""
        return [
            hashlib.sha256(f"system_prompt_{i}".encode()).hexdigest()[:16]
            for i in range(self.profile.num_shared_prefixes)
        ]
