"""Tests for all four scheduling policies."""

import pytest
from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.scheduler.kv_aware_scheduler import KVAwareSLAScheduler
from control_plane.scheduler.least_loaded import LeastLoadedScheduler
from control_plane.scheduler.memory_aware import MemoryAwareScheduler
from control_plane.scheduler.round_robin import RoundRobinScheduler
from control_plane.sla_policy.sla_policy import SLAPolicy
from core.models import GPUType, InferenceRequest, SLAClass, make_gpu_node


@pytest.fixture
def nodes():
    return [make_gpu_node(f"n{i}", GPUType.H100_80GB) for i in range(4)]


@pytest.fixture
def kv_manager(nodes):
    return KVCacheManager(nodes)


class TestRoundRobinScheduler:
    def test_accepts_request(self, nodes):
        sched = RoundRobinScheduler(nodes)
        req = InferenceRequest(prompt_tokens=256, max_output_tokens=128)
        result = sched.schedule(req)
        assert result.accepted
        assert result.node_id is not None

    def test_distributes_across_nodes(self, nodes):
        sched = RoundRobinScheduler(nodes)
        assigned = set()
        for _ in range(len(nodes) * 2):
            req = InferenceRequest(prompt_tokens=64, max_output_tokens=32)
            r = sched.schedule(req)
            if r.accepted:
                assigned.add(r.node_id)
        assert len(assigned) > 1

    def test_rejects_when_all_full(self, nodes):
        sched = RoundRobinScheduler(nodes)
        # Flood with very large requests
        for node in nodes:
            node.used_memory_gb = node.total_memory_gb - 0.1
            node.kv_blocks_used = node.kv_blocks_total
        req = InferenceRequest(prompt_tokens=4096, max_output_tokens=2048)
        result = sched.schedule(req)
        assert not result.accepted


class TestLeastLoadedScheduler:
    def test_prefers_idle_node(self, nodes):
        sched = LeastLoadedScheduler(nodes)
        # Manually load node 0
        nodes[0].active_requests = 10
        req = InferenceRequest(prompt_tokens=256, max_output_tokens=128)
        result = sched.schedule(req)
        assert result.accepted
        assert result.node_id != "n0"

    def test_accepts_basic_request(self, nodes):
        sched = LeastLoadedScheduler(nodes)
        req = InferenceRequest(prompt_tokens=512, max_output_tokens=256)
        result = sched.schedule(req)
        assert result.accepted


class TestMemoryAwareScheduler:
    def test_prefers_node_with_more_kv_headroom(self, nodes):
        sched = MemoryAwareScheduler(nodes)
        # Partially fill node 0
        nodes[0].kv_blocks_used = int(nodes[0].kv_blocks_total * 0.8)
        nodes[0].used_memory_gb = nodes[0].total_memory_gb * 0.8
        req = InferenceRequest(prompt_tokens=256, max_output_tokens=128)
        result = sched.schedule(req)
        assert result.accepted
        # Should prefer a non-loaded node
        assert result.node_id != "n0"

    def test_score_function(self, nodes):
        sched = MemoryAwareScheduler(nodes)
        n = nodes[0]
        score = sched._score(n)
        assert 0.0 <= score <= 1.0


class TestKVAwareSLAScheduler:
    def test_accepts_interactive_request(self, nodes, kv_manager):
        sched = KVAwareSLAScheduler(nodes, kv_manager)
        req = InferenceRequest(
            prompt_tokens=512, max_output_tokens=256,
            sla_class=SLAClass.INTERACTIVE,
        )
        result = sched.schedule(req)
        assert result.accepted

    def test_accepts_realtime_request(self, nodes, kv_manager):
        sched = KVAwareSLAScheduler(nodes, kv_manager)
        req = InferenceRequest(
            prompt_tokens=128, max_output_tokens=64,
            sla_class=SLAClass.REALTIME,
        )
        result = sched.schedule(req)
        assert result.accepted

    def test_on_complete_releases_resources(self, nodes, kv_manager):
        sched = KVAwareSLAScheduler(nodes, kv_manager)
        req = InferenceRequest(prompt_tokens=256, max_output_tokens=128)
        result = sched.schedule(req)
        assert result.accepted

        node = next(n for n in nodes if n.node_id == result.node_id)
        before_active = node.active_requests
        sched.on_complete(req, node)
        assert node.active_requests <= before_active

    def test_stats_populated(self, nodes, kv_manager):
        sched = KVAwareSLAScheduler(nodes, kv_manager)
        req = InferenceRequest(prompt_tokens=256, max_output_tokens=128)
        sched.schedule(req)
        stats = sched.stats
        assert stats["total_scheduled"] >= 1
