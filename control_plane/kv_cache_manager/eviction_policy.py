"""
KV cache eviction policies.

When a GPU node runs low on KV blocks, the eviction policy selects
which blocks to evict to make room for incoming requests.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from core.models import KVBlock


class EvictionPolicy(ABC):
    """Abstract base for KV block eviction strategies."""

    @abstractmethod
    def on_access(self, block: KVBlock) -> None:
        """Called when a block is accessed (hit or fill)."""

    @abstractmethod
    def on_allocate(self, block: KVBlock) -> None:
        """Called when a new block is allocated."""

    @abstractmethod
    def on_free(self, block: KVBlock) -> None:
        """Called when a block is explicitly freed."""

    @abstractmethod
    def pick_victim(self, candidates: list[KVBlock]) -> Optional[KVBlock]:
        """Choose the best block to evict from the candidate list."""


class LRUEvictionPolicy(EvictionPolicy):
    """
    Least-Recently-Used eviction.

    Standard choice: evict the block that was accessed least recently.
    Works well for workloads where recently-used prefixes are likely
    to be reused again (temporal locality).
    """

    def __init__(self) -> None:
        self._access_times: dict[int, float] = {}

    def on_access(self, block: KVBlock) -> None:
        self._access_times[block.block_id] = time.monotonic()
        block.last_accessed = self._access_times[block.block_id]

    def on_allocate(self, block: KVBlock) -> None:
        self._access_times[block.block_id] = time.monotonic()

    def on_free(self, block: KVBlock) -> None:
        self._access_times.pop(block.block_id, None)

    def pick_victim(self, candidates: list[KVBlock]) -> Optional[KVBlock]:
        evictable = [b for b in candidates if not b.pinned]
        if not evictable:
            return None
        return min(evictable, key=lambda b: self._access_times.get(b.block_id, 0.0))


class LFUEvictionPolicy(EvictionPolicy):
    """
    Least-Frequently-Used eviction.

    Better for workloads with repeated shared system-prompt prefixes
    (e.g., RAG pipelines with a fixed context window) where a hot prefix
    should survive even if not accessed recently.
    """

    def __init__(self) -> None:
        self._freq: dict[int, int] = {}

    def on_access(self, block: KVBlock) -> None:
        self._freq[block.block_id] = self._freq.get(block.block_id, 0) + 1

    def on_allocate(self, block: KVBlock) -> None:
        self._freq[block.block_id] = 0

    def on_free(self, block: KVBlock) -> None:
        self._freq.pop(block.block_id, None)

    def pick_victim(self, candidates: list[KVBlock]) -> Optional[KVBlock]:
        evictable = [b for b in candidates if not b.pinned]
        if not evictable:
            return None
        return min(evictable, key=lambda b: self._freq.get(b.block_id, 0))


class BeladyEvictionPolicy(EvictionPolicy):
    """
    Belady's (offline optimal) policy – evict the block used furthest in the future.

    Only usable in the simulator where future access patterns are known.
    Provides an upper bound on cache hit rates for comparison.
    """

    def __init__(self, future_accesses: Optional[dict[int, list[float]]] = None) -> None:
        self._future: dict[int, list[float]] = future_accesses or {}
        self._now = time.monotonic()

    def on_access(self, block: KVBlock) -> None:
        # Pop the immediate next access time
        q = self._future.get(block.block_id, [])
        if q:
            q.pop(0)

    def on_allocate(self, block: KVBlock) -> None:
        pass

    def on_free(self, block: KVBlock) -> None:
        self._future.pop(block.block_id, None)

    def pick_victim(self, candidates: list[KVBlock]) -> Optional[KVBlock]:
        evictable = [b for b in candidates if not b.pinned]
        if not evictable:
            return None

        def next_use(block: KVBlock) -> float:
            q = self._future.get(block.block_id, [])
            return q[0] if q else float("inf")

        return max(evictable, key=next_use)
