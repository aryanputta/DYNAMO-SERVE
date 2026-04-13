"""
Core data models for DYNAMO-SERVE.

These types are the shared vocabulary across all subsystems:
control plane, data plane, simulator, ML layer, and benchmarks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GPUType(str, Enum):
    """Supported GPU generations with their memory and bandwidth characteristics."""

    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"
    H100_80GB = "H100_80GB"
    H100_NVL = "H100_NVL"       # NVLink connected H100
    B200_192GB = "B200_192GB"   # Blackwell class
    A10_24GB = "A10_24GB"       # Edge / smaller workloads


class SLAClass(str, Enum):
    """
    SLA tiers that govern scheduling priority, TTFT budgets, and eviction policy.

    REALTIME   – interactive chat, strict TTFT < 200 ms, TPOT < 50 ms
    INTERACTIVE – API callers, TTFT < 1 s, TPOT < 100 ms
    BATCH       – offline jobs, best-effort throughput
    """

    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BATCH = "batch"


class TenantPriority(int, Enum):
    """Numeric priority used for preemption decisions (higher = more important)."""

    CRITICAL = 10
    HIGH = 7
    NORMAL = 5
    LOW = 2
    BACKGROUND = 1


# ---------------------------------------------------------------------------
# GPU Node
# ---------------------------------------------------------------------------


@dataclass
class GPUNode:
    """
    Represents a single GPU accelerator within the serving cluster.

    Memory is tracked at the block level (KV blocks + model weights).
    Compute capacity is expressed as abstract throughput units (tokens/sec).
    """

    node_id: str
    gpu_type: GPUType
    total_memory_gb: float          # Physical HBM capacity
    memory_bandwidth_gbps: float    # HBM bandwidth
    compute_tflops: float           # BF16 peak TFLOPs
    nvlink_enabled: bool = False    # NVLink / NVSwitch connectivity
    num_gpus: int = 1               # GPUs in this node (tensor-parallel group)

    # Runtime state (mutable)
    used_memory_gb: float = 0.0
    active_requests: int = 0
    kv_blocks_used: int = 0
    kv_blocks_total: int = 0        # Set by KVCacheManager on startup

    def __post_init__(self) -> None:
        if self.kv_blocks_total == 0:
            # Reserve 40 % for model weights; remaining for KV cache
            kv_memory_gb = self.total_memory_gb * 0.60
            # Each KV block = 2 MB (typical for 7B model, 32 layers, fp16)
            self.kv_blocks_total = int(kv_memory_gb * 1024 / 2)

    @property
    def free_memory_gb(self) -> float:
        return max(0.0, self.total_memory_gb - self.used_memory_gb)

    @property
    def memory_pressure(self) -> float:
        """0.0 = empty, 1.0 = full."""
        return self.used_memory_gb / self.total_memory_gb

    @property
    def kv_blocks_free(self) -> int:
        return max(0, self.kv_blocks_total - self.kv_blocks_used)

    @property
    def compute_utilization(self) -> float:
        """Rough utilization estimate based on active requests."""
        # Assume each request uses ~5 % of compute; saturates at 20 concurrent
        return min(1.0, self.active_requests * 0.05)

    def can_fit(self, blocks_needed: int, memory_needed_gb: float) -> bool:
        return (
            self.kv_blocks_free >= blocks_needed
            and self.free_memory_gb >= memory_needed_gb
        )

    def allocate(self, blocks: int, memory_gb: float) -> None:
        self.kv_blocks_used += blocks
        self.used_memory_gb += memory_gb
        self.active_requests += 1

    def release(self, blocks: int, memory_gb: float) -> None:
        self.kv_blocks_used = max(0, self.kv_blocks_used - blocks)
        self.used_memory_gb = max(0.0, self.used_memory_gb - memory_gb)
        self.active_requests = max(0, self.active_requests - 1)

    def __repr__(self) -> str:
        return (
            f"GPUNode({self.node_id}, {self.gpu_type.value}, "
            f"mem={self.used_memory_gb:.1f}/{self.total_memory_gb}GB, "
            f"kv={self.kv_blocks_used}/{self.kv_blocks_total})"
        )


# GPU specification catalogue
GPU_SPECS: dict[GPUType, dict] = {
    GPUType.A100_40GB: {
        "total_memory_gb": 40.0,
        "memory_bandwidth_gbps": 1555.0,
        "compute_tflops": 312.0,
        "nvlink_enabled": False,
    },
    GPUType.A100_80GB: {
        "total_memory_gb": 80.0,
        "memory_bandwidth_gbps": 2000.0,
        "compute_tflops": 312.0,
        "nvlink_enabled": True,
    },
    GPUType.H100_80GB: {
        "total_memory_gb": 80.0,
        "memory_bandwidth_gbps": 3350.0,
        "compute_tflops": 989.0,
        "nvlink_enabled": False,
    },
    GPUType.H100_NVL: {
        "total_memory_gb": 94.0,
        "memory_bandwidth_gbps": 3900.0,
        "compute_tflops": 989.0,
        "nvlink_enabled": True,
    },
    GPUType.B200_192GB: {
        "total_memory_gb": 192.0,
        "memory_bandwidth_gbps": 8000.0,
        "compute_tflops": 4500.0,
        "nvlink_enabled": True,
    },
    GPUType.A10_24GB: {
        "total_memory_gb": 24.0,
        "memory_bandwidth_gbps": 600.0,
        "compute_tflops": 125.0,
        "nvlink_enabled": False,
    },
}


def make_gpu_node(node_id: str, gpu_type: GPUType, **overrides) -> GPUNode:
    """Factory that creates a GPUNode from the GPU spec catalogue."""
    spec = {**GPU_SPECS[gpu_type], **overrides}
    return GPUNode(node_id=node_id, gpu_type=gpu_type, **spec)


# ---------------------------------------------------------------------------
# KV Cache types
# ---------------------------------------------------------------------------


@dataclass
class KVBlock:
    """
    A fixed-size unit of KV cache memory.

    Following PagedAttention (vLLM), KV cache is managed in discrete blocks
    rather than contiguous per-sequence allocations.
    """

    block_id: int
    node_id: str
    size_bytes: int = 2 * 1024 * 1024   # 2 MiB default
    token_capacity: int = 16             # tokens per block (block_size)
    tokens_used: int = 0
    prefix_hash: Optional[str] = None   # Hash of token prefix for reuse
    last_accessed: float = field(default_factory=time.monotonic)
    pinned: bool = False                 # True = cannot be evicted

    @property
    def is_full(self) -> bool:
        return self.tokens_used >= self.token_capacity

    @property
    def utilization(self) -> float:
        return self.tokens_used / self.token_capacity


@dataclass
class KVCacheStats:
    """Snapshot of KV cache health for a single node."""

    node_id: str
    total_blocks: int
    used_blocks: int
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    spill_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks if self.total_blocks > 0 else 0.0


# ---------------------------------------------------------------------------
# Inference Request
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """
    A single LLM inference request flowing through the serving system.

    Contains everything the scheduler needs to make placement decisions:
    token counts, SLA class, tenant metadata, and KV cache hints.
    """

    prompt_tokens: int
    max_output_tokens: int
    sla_class: SLAClass = SLAClass.INTERACTIVE
    tenant_id: str = "default"
    priority: TenantPriority = TenantPriority.NORMAL
    model_id: str = "llama-3-70b"

    # Auto-generated fields
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    arrival_time: float = field(default_factory=time.monotonic)

    # Optional routing hints
    preferred_node_id: Optional[str] = None
    prefix_hash: Optional[str] = None   # For KV prefix reuse

    # Filled in during scheduling
    assigned_node_id: Optional[str] = None
    queue_time: Optional[float] = None

    @property
    def kv_blocks_needed(self) -> int:
        """Blocks needed for the full prompt (prefill) + expected output (decode)."""
        total_tokens = self.prompt_tokens + self.max_output_tokens
        # 16 tokens per block (must match KVBlock.token_capacity)
        return max(1, (total_tokens + 15) // 16)

    @property
    def estimated_memory_gb(self) -> float:
        """Rough memory footprint: 2 MiB per KV block."""
        return self.kv_blocks_needed * 2 / 1024

    @property
    def is_long_context(self) -> bool:
        return self.prompt_tokens > 8192

    def __repr__(self) -> str:
        return (
            f"Request({self.request_id}, {self.prompt_tokens}+{self.max_output_tokens}tok, "
            f"{self.sla_class.value}, tenant={self.tenant_id})"
        )


# ---------------------------------------------------------------------------
# Results and metrics
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """
    Outcome of a completed (or rejected/failed) inference request.

    Captures the latency breakdown needed for SLA accounting:
    TTFT (time to first token), TPOT (time per output token), total throughput.
    """

    request_id: str
    success: bool = True
    rejected: bool = False
    rejection_reason: Optional[str] = None

    # Latency breakdown (milliseconds)
    ttft_ms: float = 0.0            # Time to first token
    tpot_ms: float = 0.0            # Average time per output token
    total_latency_ms: float = 0.0   # End-to-end wall time
    queue_wait_ms: float = 0.0      # Time spent waiting in scheduler queue

    # Throughput
    prompt_tokens: int = 0
    output_tokens: int = 0

    # Resource accounting
    gpu_node_id: Optional[str] = None
    kv_blocks_allocated: int = 0
    kv_cache_hit: bool = False       # Prefix reuse hit
    kv_spilled: bool = False         # Had to evict/spill KV blocks

    # Cost accounting (per 1M tokens)
    cost_per_1m_tokens: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_latency_ms / 1000.0)

    def sla_met(self, sla_class: SLAClass) -> bool:
        budgets = {
            SLAClass.REALTIME:    {"ttft": 200.0, "tpot": 50.0},
            SLAClass.INTERACTIVE: {"ttft": 1000.0, "tpot": 100.0},
            SLAClass.BATCH:       {"ttft": 30000.0, "tpot": 500.0},
        }
        b = budgets[sla_class]
        return self.ttft_ms <= b["ttft"] and self.tpot_ms <= b["tpot"]


@dataclass
class SchedulerMetrics:
    """
    Aggregate metrics from a scheduling run, used for benchmark comparison.

    All latency values in milliseconds; throughput in tokens/sec.
    """

    scheduler_name: str
    scenario: str
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    sla_violations: int = 0

    # Latency percentiles (ms)
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_ttft_ms: float = 0.0

    p50_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    p50_total_ms: float = 0.0
    p99_total_ms: float = 0.0

    # Throughput
    mean_tokens_per_sec: float = 0.0
    total_tokens_generated: int = 0

    # KV cache
    cache_hit_rate: float = 0.0
    total_kv_spills: int = 0
    mean_kv_blocks_used: float = 0.0

    # GPU utilization
    mean_gpu_utilization: float = 0.0
    mean_memory_pressure: float = 0.0

    # Cost
    cost_per_1m_tokens: float = 0.0

    # Fairness (Jain's index across tenants)
    fairness_index: float = 1.0

    # Wall-clock duration of the simulated run
    simulation_duration_s: float = 0.0

    @property
    def completion_rate(self) -> float:
        return self.completed_requests / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected_requests / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def sla_violation_rate(self) -> float:
        c = self.completed_requests
        return self.sla_violations / c if c > 0 else 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BatchJob:
    """A group of requests batched together for a single forward pass."""

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    requests: list[InferenceRequest] = field(default_factory=list)
    node_id: Optional[str] = None
    phase: str = "prefill"   # "prefill" or "decode"
    created_at: float = field(default_factory=time.monotonic)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.max_output_tokens for r in self.requests)

    @property
    def batch_size(self) -> int:
        return len(self.requests)


@dataclass
class NodeUtilization:
    """Point-in-time snapshot of a GPU node's utilization."""

    node_id: str
    timestamp: float = field(default_factory=time.monotonic)
    compute_utilization: float = 0.0
    memory_pressure: float = 0.0
    kv_cache_utilization: float = 0.0
    active_requests: int = 0
    queued_requests: int = 0
    tokens_per_sec: float = 0.0
