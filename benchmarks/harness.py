"""
Benchmark Harness.

Replays a workload trace against a scheduler, collects results,
and computes SchedulerMetrics for the benchmark report.

The harness simulates the passage of time by replaying requests in
arrival-time order. Each request is scheduled, executed by the mock
runtime, and its result fed to the tracer. The harness also drives
periodic admission-queue draining and node-state snapshots.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.scheduler.base_scheduler import BaseScheduler
from control_plane.sla_policy.sla_policy import SLAPolicy
from core.models import (
    GPUNode,
    InferenceRequest,
    RequestResult,
    SchedulerMetrics,
    SLAClass,
)
from data_plane.mock_gpu_runtime.gpu_runtime import MockGPURuntime
from data_plane.tracing_hooks.tracer import RequestTracer
from simulator.contention_model.contention import ContentionModel

logger = logging.getLogger(__name__)


class BenchmarkHarness:
    """
    Drives a scheduling simulation end-to-end.

    Usage::

        harness = BenchmarkHarness(scheduler, nodes, kv_manager)
        metrics = harness.run(requests, scenario_name="burst_traffic")
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        nodes: list[GPUNode],
        kv_manager: KVCacheManager,
        sla_policy: Optional[SLAPolicy] = None,
        runtime_seed: Optional[int] = 42,
    ) -> None:
        self._scheduler = scheduler
        self._nodes = nodes
        self._node_map = {n.node_id: n for n in nodes}
        self._kv_manager = kv_manager
        self._sla_policy = sla_policy or SLAPolicy()
        self._runtime = MockGPURuntime(seed=runtime_seed)
        self._tracer = RequestTracer()
        self._contention = ContentionModel()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        requests: list[InferenceRequest],
        scenario_name: str = "default",
    ) -> SchedulerMetrics:
        """
        Replay *requests* through the scheduler and return aggregate metrics.
        """
        wall_start = time.monotonic()
        self._tracer.reset()

        results: list[RequestResult] = []
        node_util_snapshots: list[float] = []
        mem_pressure_snapshots: list[float] = []

        for req in requests:
            self._tracer.record("request_arrived", req.request_id)

            # Schedule
            sched_result = self._scheduler.schedule(req)

            if not sched_result.accepted:
                # Rejected or queued
                result = RequestResult(
                    request_id=req.request_id,
                    success=False,
                    rejected=True,
                    rejection_reason=sched_result.rejection_reason,
                    prompt_tokens=req.prompt_tokens,
                )
                results.append(result)
                self._tracer.record_result(result)
                continue

            node = self._node_map[sched_result.node_id]
            req.assigned_node_id = sched_result.node_id

            # Allocate KV blocks
            kv_blocks, cache_hit = self._kv_manager.allocate(sched_result.node_id, req)

            # Compute contention factors
            factors = self._contention.compute(
                node,
                active_requests=node.active_requests,
                kv_evictions_this_step=max(0, req.kv_blocks_needed - len(kv_blocks)),
            )

            # Execute on mock runtime
            result = self._runtime.execute(req, node, kv_blocks, cache_hit)

            # Apply contention
            result.ttft_ms, result.tpot_ms = self._contention.apply(
                factors, result.ttft_ms, result.tpot_ms
            )
            result.total_latency_ms = (
                result.ttft_ms + result.tpot_ms * result.output_tokens + result.queue_wait_ms
            )

            # Check SLA
            sla_met = self._sla_policy.check_result(result, req.sla_class)

            # Release resources
            self._kv_manager.free(sched_result.node_id, kv_blocks, keep_prefix=True)
            self._scheduler.on_complete(req, node)

            results.append(result)
            self._tracer.record_result(result)

            # Snapshot cluster state every 50 requests
            if len(results) % 50 == 0:
                utils = [n.compute_utilization for n in self._nodes]
                mems = [n.memory_pressure for n in self._nodes]
                node_util_snapshots.append(float(np.mean(utils)))
                mem_pressure_snapshots.append(float(np.mean(mems)))

        wall_duration = time.monotonic() - wall_start
        return self._compile_metrics(
            results, scenario_name, wall_duration,
            node_util_snapshots, mem_pressure_snapshots,
        )

    # ------------------------------------------------------------------
    # Metrics compilation
    # ------------------------------------------------------------------

    def _compile_metrics(
        self,
        results: list[RequestResult],
        scenario: str,
        duration_s: float,
        util_snapshots: list[float],
        mem_snapshots: list[float],
    ) -> SchedulerMetrics:

        completed = [r for r in results if r.success and not r.rejected]
        rejected = [r for r in results if r.rejected]

        ttfts = np.array([r.ttft_ms for r in completed]) if completed else np.array([0.0])
        tpots = np.array([r.tpot_ms for r in completed]) if completed else np.array([0.0])
        totals = np.array([r.total_latency_ms for r in completed]) if completed else np.array([0.0])

        sla_violations = sum(
            1 for r in completed
            # Simplified check: TTFT > 1000ms for interactive, > 200ms for realtime
            if r.ttft_ms > 1000.0 or r.tpot_ms > 100.0
        )

        cache_hits = sum(1 for r in completed if r.kv_cache_hit)
        spills = sum(1 for r in completed if r.kv_spilled)
        total_tokens = sum(r.total_tokens for r in completed)
        tps_vals = [r.tokens_per_second for r in completed if r.tokens_per_second > 0]

        # Jain's fairness index across tenants
        fairness = self._jains_fairness(completed)

        # Average cost
        costs = [r.cost_per_1m_tokens for r in completed]
        mean_cost = float(np.mean(costs)) if costs else 0.0

        return SchedulerMetrics(
            scheduler_name=self._scheduler.name,
            scenario=scenario,
            total_requests=len(results),
            completed_requests=len(completed),
            rejected_requests=len(rejected),
            sla_violations=sla_violations,
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p95_ttft_ms=float(np.percentile(ttfts, 95)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            mean_ttft_ms=float(np.mean(ttfts)),
            p50_tpot_ms=float(np.percentile(tpots, 50)),
            p95_tpot_ms=float(np.percentile(tpots, 95)),
            p99_tpot_ms=float(np.percentile(tpots, 99)),
            p50_total_ms=float(np.percentile(totals, 50)),
            p99_total_ms=float(np.percentile(totals, 99)),
            mean_tokens_per_sec=float(np.mean(tps_vals)) if tps_vals else 0.0,
            total_tokens_generated=total_tokens,
            cache_hit_rate=cache_hits / max(1, len(completed)),
            total_kv_spills=spills,
            mean_gpu_utilization=float(np.mean(util_snapshots)) if util_snapshots else 0.0,
            mean_memory_pressure=float(np.mean(mem_snapshots)) if mem_snapshots else 0.0,
            cost_per_1m_tokens=mean_cost,
            fairness_index=fairness,
            simulation_duration_s=duration_s,
        )

    @staticmethod
    def _jains_fairness(results: list[RequestResult]) -> float:
        """
        Jain's fairness index over per-request throughput (tokens/sec).
        Range [1/n, 1.0] where 1.0 = perfectly fair.
        """
        tpss = [r.tokens_per_second for r in results if r.tokens_per_second > 0]
        if not tpss:
            return 1.0
        arr = np.array(tpss)
        return float((arr.sum() ** 2) / (len(arr) * (arr ** 2).sum()))

    @property
    def tracer(self) -> RequestTracer:
        return self._tracer
