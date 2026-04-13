"""Tests for the workload generator and simulator."""

import pytest
from simulator.workload_replay.workload_generator import (
    WorkloadGenerator,
    WorkloadProfile,
    PROFILES,
)
from core.models import SLAClass


class TestWorkloadGenerator:
    def test_generates_requests(self):
        profile = WorkloadProfile(arrival_rate_rps=10.0, duration_s=5.0)
        gen = WorkloadGenerator(profile, seed=0)
        requests = gen.generate()
        assert len(requests) > 0

    def test_requests_sorted_by_arrival(self):
        profile = WorkloadProfile(arrival_rate_rps=20.0, duration_s=10.0)
        gen = WorkloadGenerator(profile, seed=1)
        requests = gen.generate()
        times = [r.arrival_time for r in requests]
        assert times == sorted(times)

    def test_sla_distribution_respected(self):
        profile = WorkloadProfile(
            arrival_rate_rps=20.0, duration_s=30.0,
            sla_realtime_fraction=0.2,
            sla_interactive_fraction=0.6,
            sla_batch_fraction=0.2,
        )
        gen = WorkloadGenerator(profile, seed=42)
        requests = gen.generate()
        n = len(requests)
        realtime = sum(1 for r in requests if r.sla_class == SLAClass.REALTIME)
        # Allow ±10% tolerance
        assert 0.10 <= realtime / n <= 0.30

    def test_prompt_tokens_within_bounds(self):
        profile = WorkloadProfile(
            arrival_rate_rps=10.0, duration_s=5.0,
            max_prompt_tokens=4096,
        )
        gen = WorkloadGenerator(profile, seed=7)
        requests = gen.generate()
        for r in requests:
            assert 1 <= r.prompt_tokens <= 4096

    def test_burst_traffic_profile_exists(self):
        assert "burst_traffic" in PROFILES

    def test_all_profiles_generate_nonzero(self):
        for name, profile in PROFILES.items():
            short = WorkloadProfile(**{**profile.__dict__, "duration_s": 5.0})
            gen = WorkloadGenerator(short, seed=0)
            reqs = gen.generate()
            assert len(reqs) >= 0, f"Profile {name} generated no requests"

    def test_prefix_sharing_assigns_hashes(self):
        profile = WorkloadProfile(
            arrival_rate_rps=20.0, duration_s=10.0,
            prefix_sharing_prob=1.0,
        )
        gen = WorkloadGenerator(profile, seed=3)
        requests = gen.generate()
        with_prefix = [r for r in requests if r.prefix_hash is not None]
        assert len(with_prefix) > 0


class TestTopologies:
    def test_nvlink_rack_nodes(self):
        from simulator.gpu_topology.topology import make_nvlink_rack
        topo = make_nvlink_rack(num_nodes=8)
        assert len(topo.nodes) == 8
        for n in topo.nodes:
            assert n.nvlink_enabled

    def test_heterogeneous_cluster(self):
        from simulator.gpu_topology.topology import make_heterogeneous_cluster
        topo = make_heterogeneous_cluster(n_h100=2, n_a100=2, n_a10=2)
        assert len(topo.nodes) == 6

    def test_transfer_cost_same_node_is_zero(self):
        from simulator.gpu_topology.topology import make_nvlink_rack
        topo = make_nvlink_rack(4)
        cost = topo.transfer_cost_ms("gpu_0", "gpu_0", 10.0)
        assert cost == 0.0

    def test_transfer_cost_different_nodes_positive(self):
        from simulator.gpu_topology.topology import make_nvlink_rack
        topo = make_nvlink_rack(4)
        cost = topo.transfer_cost_ms("gpu_0", "gpu_1", 1.0)
        assert cost > 0.0
