"""
KV Cache Simulator.

Replays a request trace against different eviction policies and measures:
  - Hit rate
  - Eviction count
  - Spill events (blocks needed > total capacity)
  - Per-request latency impact

Used to compare LRU vs LFU vs Belady (optimal) and to calibrate
the KV-Aware SLA Scheduler's eviction parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from control_plane.kv_cache_manager.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
)
from core.models import InferenceRequest, KVBlock


@dataclass
class CacheSimResult:
    """Aggregated result from a cache simulation run."""

    policy_name: str
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    spills: int = 0
    total_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def spill_rate(self) -> float:
        return self.spills / self.total_requests if self.total_requests > 0 else 0.0


class KVCacheSimulator:
    """
    Simulates KV cache behaviour over a request trace.

    Isolates the caching subsystem from scheduling decisions so we can
    quantify the impact of eviction policy alone on cache hit rates.
    """

    def __init__(
        self,
        total_blocks: int = 1024,
        block_size_tokens: int = 16,
    ) -> None:
        self.total_blocks = total_blocks
        self.block_size = block_size_tokens

    def run(
        self,
        requests: list[InferenceRequest],
        policy: EvictionPolicy,
        policy_name: str = "lru",
    ) -> CacheSimResult:
        """Simulate the cache over *requests* using *policy*."""
        result = CacheSimResult(
            policy_name=policy_name,
            total_requests=len(requests),
            total_blocks=self.total_blocks,
        )

        # Simple flat pool of blocks
        block_pool: list[KVBlock] = [
            KVBlock(block_id=i, node_id="sim", token_capacity=self.block_size)
            for i in range(self.total_blocks)
        ]
        used_blocks: dict[str, list[KVBlock]] = {}  # prefix_hash -> blocks
        next_bid = self.total_blocks

        for req in requests:
            n_needed = req.kv_blocks_needed

            # Check prefix cache
            hit = False
            if req.prefix_hash and req.prefix_hash in used_blocks:
                cached = used_blocks[req.prefix_hash]
                n_needed = max(0, n_needed - len(cached))
                hit = True
                for b in cached:
                    policy.on_access(b)

            if hit:
                result.hits += 1
            else:
                result.misses += 1

            # Find free blocks
            free = [b for b in block_pool if b.tokens_used == 0 and not b.pinned]

            # Evict if needed
            while len(free) < n_needed:
                candidates = [b for b in block_pool if b.tokens_used > 0 and not b.pinned]
                victim = policy.pick_victim(candidates)
                if victim is None:
                    result.spills += 1
                    break
                # Evict victim
                if victim.prefix_hash and victim.prefix_hash in used_blocks:
                    cached_list = used_blocks[victim.prefix_hash]
                    if victim in cached_list:
                        cached_list.remove(victim)
                    if not cached_list:
                        del used_blocks[victim.prefix_hash]
                policy.on_free(victim)
                victim.tokens_used = 0
                victim.prefix_hash = None
                free.append(victim)
                result.evictions += 1

            # Allocate fresh blocks
            new_blocks = free[:n_needed]
            for b in new_blocks:
                b.tokens_used = self.block_size
                b.prefix_hash = req.prefix_hash
                policy.on_allocate(b)

            if req.prefix_hash:
                existing = used_blocks.get(req.prefix_hash, [])
                used_blocks[req.prefix_hash] = existing + new_blocks

        return result

    def compare_policies(
        self, requests: list[InferenceRequest]
    ) -> dict[str, CacheSimResult]:
        """Run all built-in policies and return comparative results."""
        import copy
        results = {}
        for name, policy in [("lru", LRUEvictionPolicy()), ("lfu", LFUEvictionPolicy())]:
            results[name] = self.run(copy.deepcopy(requests), policy, name)
        return results
