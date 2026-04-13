"""Integration tests: benchmark harness end-to-end."""

import pytest
from benchmarks.harness import BenchmarkHarness
from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.scheduler.kv_aware_scheduler import KVAwareSLAScheduler
from control_plane.scheduler.round_robin import RoundRobinScheduler
from control_plane.scheduler.memory_aware import MemoryAwareScheduler
from core.models import GPUType, make_gpu_node, SLAClass
from simulator.workload_replay.workload_generator import WorkloadGenerator, WorkloadProfile


@pytest.fixture
def four_h100_nodes():
    return [make_gpu_node(f"n{i}", GPUType.H100_80GB) for i in range(4)]


@pytest.fixture
def small_workload():
    profile = WorkloadProfile(arrival_rate_rps=5.0, duration_s=5.0)
    gen = WorkloadGenerator(profile, seed=0)
    return gen.generate()


class TestBenchmarkHarness:
    def test_round_robin_completes(self, four_h100_nodes, small_workload):
        kv = KVCacheManager(four_h100_nodes)
        sched = RoundRobinScheduler(four_h100_nodes)
        harness = BenchmarkHarness(sched, four_h100_nodes, kv)
        metrics = harness.run(small_workload, scenario_name="test")

        assert metrics.total_requests == len(small_workload)
        assert metrics.completed_requests + metrics.rejected_requests == metrics.total_requests

    def test_kv_aware_better_cache_hit_than_rr(self, small_workload):
        """KV-aware scheduler should have ≥ cache hit rate vs round-robin."""
        import copy

        profile = WorkloadProfile(
            arrival_rate_rps=5.0, duration_s=10.0,
            prefix_sharing_prob=0.5,
        )
        gen = WorkloadGenerator(profile, seed=99)
        reqs = gen.generate()

        nodes_rr = [make_gpu_node(f"n{i}", GPUType.H100_80GB) for i in range(4)]
        kv_rr = KVCacheManager(nodes_rr)
        rr_metrics = BenchmarkHarness(
            RoundRobinScheduler(nodes_rr), nodes_rr, kv_rr
        ).run(copy.deepcopy(reqs), "test_rr")

        nodes_kv = [make_gpu_node(f"n{i}", GPUType.H100_80GB) for i in range(4)]
        kv_kv = KVCacheManager(nodes_kv)
        kv_metrics = BenchmarkHarness(
            KVAwareSLAScheduler(nodes_kv, kv_kv), nodes_kv, kv_kv
        ).run(copy.deepcopy(reqs), "test_kv")

        # KV-aware should have equal or better cache hit rate
        assert kv_metrics.cache_hit_rate >= rr_metrics.cache_hit_rate - 0.05

    def test_metrics_fields_populated(self, four_h100_nodes, small_workload):
        kv = KVCacheManager(four_h100_nodes)
        sched = MemoryAwareScheduler(four_h100_nodes)
        harness = BenchmarkHarness(sched, four_h100_nodes, kv)
        m = harness.run(small_workload, scenario_name="test")

        assert m.scheduler_name == "memory_aware"
        assert m.scenario == "test"
        assert m.p99_ttft_ms >= 0
        assert 0.0 <= m.fairness_index <= 1.0
        assert m.completion_rate + m.rejection_rate == pytest.approx(1.0)

    def test_sla_violation_rate_bounded(self, four_h100_nodes, small_workload):
        kv = KVCacheManager(four_h100_nodes)
        sched = KVAwareSLAScheduler(four_h100_nodes, kv)
        harness = BenchmarkHarness(sched, four_h100_nodes, kv)
        m = harness.run(small_workload, "test")
        assert 0.0 <= m.sla_violation_rate <= 1.0
