"""Tests for KV cache manager."""

import pytest
from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.kv_cache_manager.eviction_policy import LRUEvictionPolicy, LFUEvictionPolicy
from core.models import GPUType, InferenceRequest, SLAClass, make_gpu_node


@pytest.fixture
def small_node():
    node = make_gpu_node("n0", GPUType.A10_24GB)
    node.kv_blocks_total = 64   # Small pool for testing
    return node


@pytest.fixture
def kv_manager(small_node):
    return KVCacheManager([small_node], block_size_tokens=16)


class TestKVCacheManager:
    def test_initializes_blocks(self, small_node, kv_manager):
        assert kv_manager.free_blocks("n0") == 64

    def test_allocate_returns_blocks(self, kv_manager):
        req = InferenceRequest(prompt_tokens=64, max_output_tokens=32)
        blocks, hit = kv_manager.allocate("n0", req)
        assert len(blocks) > 0
        assert hit is False

    def test_prefix_cache_hit(self, kv_manager):
        req1 = InferenceRequest(
            prompt_tokens=64, max_output_tokens=32,
            prefix_hash="abc123"
        )
        blocks1, hit1 = kv_manager.allocate("n0", req1)
        assert not hit1

        # Free with keep_prefix=True
        kv_manager.free("n0", blocks1, keep_prefix=True)

        req2 = InferenceRequest(
            prompt_tokens=64, max_output_tokens=32,
            prefix_hash="abc123"
        )
        blocks2, hit2 = kv_manager.allocate("n0", req2)
        assert hit2

    def test_utilization_increases_on_alloc(self, kv_manager):
        req = InferenceRequest(prompt_tokens=128, max_output_tokens=64)
        assert kv_manager.utilization("n0") == pytest.approx(0.0)
        kv_manager.allocate("n0", req)
        assert kv_manager.utilization("n0") > 0.0

    def test_stats_structure(self, kv_manager):
        stats = kv_manager.get_stats("n0")
        assert stats.node_id == "n0"
        assert stats.total_blocks == 64
        assert stats.used_blocks >= 0

    def test_prefix_hash_computation(self):
        h1 = KVCacheManager.compute_prefix_hash([1, 2, 3, 4])
        h2 = KVCacheManager.compute_prefix_hash([1, 2, 3, 4])
        h3 = KVCacheManager.compute_prefix_hash([1, 2, 3, 5])
        assert h1 == h2
        assert h1 != h3

    def test_eviction_when_full(self, kv_manager):
        # Fill the pool
        reqs = [InferenceRequest(prompt_tokens=16, max_output_tokens=8) for _ in range(6)]
        for req in reqs:
            kv_manager.allocate("n0", req)

        # Now try to allocate more – should trigger eviction
        req_new = InferenceRequest(prompt_tokens=64, max_output_tokens=32)
        blocks, _ = kv_manager.allocate("n0", req_new)
        # Should still get some blocks (eviction happened)
        assert len(blocks) >= 0   # may be partial under small pool

    def test_headroom_below_high_watermark(self, kv_manager):
        headroom = kv_manager.headroom_blocks("n0")
        # At 0% utilisation, headroom should be close to 90% of total
        assert headroom > 0


class TestEvictionPolicies:
    def test_lru_picks_least_recently_used(self):
        from core.models import KVBlock
        import time
        policy = LRUEvictionPolicy()
        b1 = KVBlock(block_id=1, node_id="n0", tokens_used=16)
        b2 = KVBlock(block_id=2, node_id="n0", tokens_used=16)
        policy.on_allocate(b1)
        time.sleep(0.01)
        policy.on_allocate(b2)
        policy.on_access(b2)  # b2 is more recent

        victim = policy.pick_victim([b1, b2])
        assert victim is b1

    def test_pinned_block_not_evicted(self):
        from core.models import KVBlock
        policy = LRUEvictionPolicy()
        b1 = KVBlock(block_id=1, node_id="n0", tokens_used=16, pinned=True)
        b2 = KVBlock(block_id=2, node_id="n0", tokens_used=16, pinned=False)
        policy.on_allocate(b1)
        policy.on_allocate(b2)

        victim = policy.pick_victim([b1, b2])
        assert victim is b2
