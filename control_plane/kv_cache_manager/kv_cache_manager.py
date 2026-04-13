"""
KV Cache Manager – the heart of memory-aware scheduling.

Responsibilities:
  1. Maintain a pool of KVBlock objects per GPU node.
  2. Allocate blocks to incoming requests.
  3. Track prefix hashes for reuse (PagedAttention-style prefix caching).
  4. Drive eviction when a node approaches its memory limit.
  5. Report per-node cache health via KVCacheStats.

Design notes:
  - Block size is fixed at 16 tokens (configurable).
  - Prefix hashes are computed over the first N tokens and stored per-block.
  - When a matching prefix is found the blocks are "pinned" during inference
    and the hit is credited to the requesting tenant.
  - Eviction uses a pluggable EvictionPolicy (LRU by default).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from control_plane.kv_cache_manager.eviction_policy import (
    EvictionPolicy,
    LRUEvictionPolicy,
)
from core.models import GPUNode, InferenceRequest, KVBlock, KVCacheStats

logger = logging.getLogger(__name__)

# Eviction watermarks
HIGH_WATERMARK = 0.90   # Start evicting when utilization exceeds this
LOW_WATERMARK = 0.75    # Evict until utilization drops to this


class KVCacheManager:
    """
    Manages KV cache blocks across all GPU nodes in the cluster.

    One KVCacheManager instance is shared by the scheduler and placement engine
    so they can query headroom before committing to a placement decision.
    """

    def __init__(
        self,
        nodes: list[GPUNode],
        block_size_tokens: int = 16,
        eviction_policy: Optional[EvictionPolicy] = None,
    ) -> None:
        self.block_size = block_size_tokens
        self.eviction_policy: EvictionPolicy = eviction_policy or LRUEvictionPolicy()

        # node_id -> list[KVBlock]
        self._blocks: dict[str, list[KVBlock]] = {}
        # prefix_hash -> list[KVBlock]  (shared across nodes for now – single cluster)
        self._prefix_cache: dict[str, list[KVBlock]] = {}

        # Per-node stats
        self._stats: dict[str, KVCacheStats] = {}

        # Global block counter
        self._next_block_id = 0

        for node in nodes:
            self._init_node(node)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_node(self, node: GPUNode) -> None:
        blocks: list[KVBlock] = []
        for _ in range(node.kv_blocks_total):
            block = KVBlock(
                block_id=self._next_block_id,
                node_id=node.node_id,
                token_capacity=self.block_size,
            )
            self.eviction_policy.on_allocate(block)
            blocks.append(block)
            self._next_block_id += 1

        self._blocks[node.node_id] = blocks
        self._stats[node.node_id] = KVCacheStats(
            node_id=node.node_id,
            total_blocks=node.kv_blocks_total,
            used_blocks=0,
        )
        logger.debug(
            "KVCacheManager: initialised %d blocks on node %s",
            node.kv_blocks_total,
            node.node_id,
        )

    def add_node(self, node: GPUNode) -> None:
        """Dynamically add a new node (e.g., scale-out event)."""
        self._init_node(node)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(
        self, node_id: str, request: InferenceRequest
    ) -> tuple[list[KVBlock], bool]:
        """
        Allocate KV blocks for *request* on *node_id*.

        Returns:
            (allocated_blocks, cache_hit)
            cache_hit is True if a prefix match was found and reused.
        """
        stats = self._stats[node_id]
        n_blocks_needed = request.kv_blocks_needed

        # 1. Check prefix cache for reuse
        cache_hit = False
        reused_blocks: list[KVBlock] = []
        if request.prefix_hash and request.prefix_hash in self._prefix_cache:
            cached = self._prefix_cache[request.prefix_hash]
            # Only reuse if cached blocks are on the same node
            node_cached = [b for b in cached if b.node_id == node_id]
            if node_cached:
                reused_blocks = node_cached
                n_blocks_needed = max(0, n_blocks_needed - len(reused_blocks))
                cache_hit = True
                for b in reused_blocks:
                    b.pinned = True
                    self.eviction_policy.on_access(b)
                stats.hit_count += 1
                logger.debug(
                    "KV prefix cache HIT for request %s on node %s (%d blocks reused)",
                    request.request_id,
                    node_id,
                    len(reused_blocks),
                )
            else:
                stats.miss_count += 1
        else:
            stats.miss_count += 1

        # 2. Ensure we have enough free blocks; evict if necessary
        self._maybe_evict(node_id, n_blocks_needed)

        # 3. Allocate fresh blocks
        free_blocks = [
            b for b in self._blocks[node_id]
            if b.tokens_used == 0 and not b.pinned and b.prefix_hash is None
        ]

        if len(free_blocks) < n_blocks_needed:
            # Still not enough after eviction – signal spill
            stats.spill_count += 1
            logger.warning(
                "KV spill on node %s: needed %d, available %d",
                node_id,
                n_blocks_needed,
                len(free_blocks),
            )
            # Give what we have (partial allocation triggers spill path)
            n_blocks_needed = len(free_blocks)

        new_blocks = free_blocks[:n_blocks_needed]
        for b in new_blocks:
            b.tokens_used = self.block_size  # Mark as in-use
            if request.prefix_hash:
                b.prefix_hash = request.prefix_hash
            b.pinned = True
            b.last_accessed = time.monotonic()
            self.eviction_policy.on_access(b)

        stats.used_blocks += len(new_blocks)
        all_blocks = reused_blocks + new_blocks

        # Register prefix
        if request.prefix_hash and new_blocks:
            self._prefix_cache[request.prefix_hash] = all_blocks

        return all_blocks, cache_hit

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def free(self, node_id: str, blocks: list[KVBlock], keep_prefix: bool = True) -> None:
        """
        Release blocks back to the free pool.

        If *keep_prefix* is True, the block's prefix hash is retained so the
        next matching request can reuse it (LRU semantics: the block stays
        in the pool but becomes evictable).
        """
        stats = self._stats[node_id]
        for b in blocks:
            b.pinned = False
            if not keep_prefix:
                b.tokens_used = 0
                b.prefix_hash = None
                stats.used_blocks = max(0, stats.used_blocks - 1)
            # Leave tokens_used > 0 so eviction policy can track it
            self.eviction_policy.on_access(b)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_evict(self, node_id: str, n_needed: int) -> int:
        """
        Evict blocks until there is room for *n_needed* new blocks.

        Returns the number of blocks evicted.
        """
        stats = self._stats[node_id]
        all_blocks = self._blocks[node_id]
        free_count = sum(1 for b in all_blocks if b.tokens_used == 0 and not b.pinned)

        evicted = 0
        while free_count < n_needed:
            candidates = [
                b for b in all_blocks
                if not b.pinned and b.tokens_used > 0
            ]
            victim = self.eviction_policy.pick_victim(candidates)
            if victim is None:
                break

            # Remove from prefix cache
            if victim.prefix_hash and victim.prefix_hash in self._prefix_cache:
                cached = self._prefix_cache[victim.prefix_hash]
                if victim in cached:
                    cached.remove(victim)
                if not cached:
                    del self._prefix_cache[victim.prefix_hash]

            # Reset block
            self.eviction_policy.on_free(victim)
            victim.tokens_used = 0
            victim.prefix_hash = None
            victim.last_accessed = time.monotonic()
            stats.used_blocks = max(0, stats.used_blocks - 1)
            stats.eviction_count += 1
            free_count += 1
            evicted += 1

        return evicted

    def force_evict(self, node_id: str, n: int) -> int:
        """Explicitly evict *n* blocks (used by admission controller)."""
        return self._maybe_evict(node_id, n)

    # ------------------------------------------------------------------
    # Stats & queries
    # ------------------------------------------------------------------

    def get_stats(self, node_id: str) -> KVCacheStats:
        return self._stats[node_id]

    def get_all_stats(self) -> dict[str, KVCacheStats]:
        return dict(self._stats)

    def free_blocks(self, node_id: str) -> int:
        stats = self._stats[node_id]
        return stats.total_blocks - stats.used_blocks

    def utilization(self, node_id: str) -> float:
        stats = self._stats[node_id]
        if stats.total_blocks == 0:
            return 0.0
        return stats.used_blocks / stats.total_blocks

    def headroom_blocks(self, node_id: str) -> int:
        """Blocks available before hitting the HIGH_WATERMARK."""
        stats = self._stats[node_id]
        high_mark = int(stats.total_blocks * HIGH_WATERMARK)
        return max(0, high_mark - stats.used_blocks)

    # ------------------------------------------------------------------
    # Prefix hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prefix_hash(token_ids: list[int], prefix_len: int = 64) -> str:
        """
        Compute a deterministic prefix hash for KV reuse lookups.

        Uses the first *prefix_len* token IDs (or all if shorter).
        """
        tokens = token_ids[:prefix_len]
        raw = ",".join(str(t) for t in tokens).encode()
        return hashlib.sha256(raw).hexdigest()[:16]
