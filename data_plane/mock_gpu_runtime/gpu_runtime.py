"""
Mock GPU Runtime.

Simulates the execution of an LLM inference request on a GPU node, producing
realistic TTFT and TPOT values based on the node's hardware spec and current
load without actually running any model.

Physics model
-------------
Prefill phase (compute-bound):
    prefill_time = prompt_tokens * bytes_per_token / (tflops * efficiency * batch_factor)

Decode phase (memory-bandwidth-bound):
    per_token_time = model_size_bytes / (bandwidth_gbps * 1e9 * efficiency)

Both phases include:
    - A stochastic noise term (Gaussian, ±10 %)
    - A load-based contention penalty (linear in active_requests)
    - An NVLink transfer bonus for long-context requests on NVLink nodes
"""

from __future__ import annotations

import random
import time
from typing import Optional

from core.models import GPUNode, GPUType, InferenceRequest, KVBlock, RequestResult, SLAClass

# ── Model-size assumptions (bytes) ──
# 7B fp16: 14 GB, 70B fp16: 140 GB
MODEL_SIZE_BYTES: dict[str, int] = {
    "llama-3-7b":  14 * 1024 ** 3,
    "llama-3-70b": 140 * 1024 ** 3,
    "mistral-7b":  14 * 1024 ** 3,
    "mixtral-8x7b": 90 * 1024 ** 3,   # MoE routing overhead included
    "default":     70 * 1024 ** 3,
}

# bytes transferred per attention token in the decode phase (KV cache read)
KV_BYTES_PER_TOKEN = 2 * 128 * 32 * 2   # 2 * head_dim * n_heads * sizeof(fp16)

# Compute efficiency factor (accounts for kernel launch overhead, fragmentation)
COMPUTE_EFFICIENCY = 0.65
BANDWIDTH_EFFICIENCY = 0.80

# Per-active-request contention penalty (ms)
CONTENTION_MS_PER_REQUEST = 5.0

# Noise fraction (Gaussian, sigma = NOISE_SIGMA * value)
NOISE_SIGMA = 0.08

# Base cost per 1M tokens (USD, rough cloud GPU parity estimate)
BASE_COST_PER_1M: dict[GPUType, float] = {
    GPUType.A10_24GB:   2.50,
    GPUType.A100_40GB:  4.00,
    GPUType.A100_80GB:  5.00,
    GPUType.H100_80GB:  8.00,
    GPUType.H100_NVL:   9.00,
    GPUType.B200_192GB: 15.00,
}


class MockGPURuntime:
    """
    Simulates inference execution on a GPUNode, returning a RequestResult.

    The simulation does NOT use wall-clock time for the "GPU execution";
    it computes latency analytically and injects controlled noise to produce
    realistic-looking benchmark distributions.

    Usage::

        runtime = MockGPURuntime()
        result = runtime.execute(request, node, kv_blocks)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def execute(
        self,
        request: InferenceRequest,
        node: GPUNode,
        kv_blocks: list[KVBlock],
        cache_hit: bool = False,
    ) -> RequestResult:
        """
        Simulate executing *request* on *node* and return a RequestResult.

        Args:
            request: The inference request.
            node:    The GPU node executing the request.
            kv_blocks: KV blocks allocated for this request.
            cache_hit: Whether a prefix cache hit occurred.
        """
        arrival = request.arrival_time
        schedule_time = time.monotonic()
        queue_wait_ms = max(0.0, (schedule_time - arrival) * 1000.0)

        # ── Prefill (TTFT) ──
        ttft_ms = self._compute_ttft(request, node, cache_hit)

        # ── Decode (TPOT) ──
        tpot_ms = self._compute_tpot(request, node)

        # ── Total latency ──
        output_tokens = self._sample_output_tokens(request)
        total_latency_ms = ttft_ms + tpot_ms * output_tokens + queue_wait_ms

        # ── Spill detection ──
        spilled = len(kv_blocks) < request.kv_blocks_needed

        # ── Cost accounting ──
        cost = self._compute_cost(node, request.prompt_tokens + output_tokens)

        return RequestResult(
            request_id=request.request_id,
            success=True,
            rejected=False,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            queue_wait_ms=queue_wait_ms,
            prompt_tokens=request.prompt_tokens,
            output_tokens=output_tokens,
            gpu_node_id=node.node_id,
            kv_blocks_allocated=len(kv_blocks),
            kv_cache_hit=cache_hit,
            kv_spilled=spilled,
            cost_per_1m_tokens=cost,
        )

    # ------------------------------------------------------------------
    # Physics model
    # ------------------------------------------------------------------

    def _compute_ttft(
        self,
        request: InferenceRequest,
        node: GPUNode,
        cache_hit: bool,
    ) -> float:
        """
        Time to first token (ms).

        Dominated by the prefill forward pass, which is compute-bound for long prompts.
        KV prefix hit reduces effective prompt length.
        """
        effective_tokens = request.prompt_tokens
        if cache_hit:
            # Reuse reduces effective prefill to the non-cached suffix
            effective_tokens = max(1, int(request.prompt_tokens * 0.25))

        # Compute-bound prefill: tokens * FLOP_per_token / (TFLOPS * efficiency)
        flop_per_token = 2 * MODEL_SIZE_BYTES.get(request.model_id, MODEL_SIZE_BYTES["default"])
        compute_time_ms = (
            effective_tokens * flop_per_token
            / (node.compute_tflops * 1e12 * COMPUTE_EFFICIENCY)
            * 1000.0
        )

        # Contention penalty
        contention_ms = node.active_requests * CONTENTION_MS_PER_REQUEST

        # NVLink bonus: long-context on NVLink nodes avoids PCIe transfer overhead
        nvlink_bonus_ms = 0.0
        if node.nvlink_enabled and request.is_long_context:
            nvlink_bonus_ms = -20.0  # negative = improvement

        # Memory bandwidth for KV cache read
        kv_read_ms = (
            effective_tokens * KV_BYTES_PER_TOKEN
            / (node.memory_bandwidth_gbps * 1e9 * BANDWIDTH_EFFICIENCY)
            * 1000.0
        )

        base_ttft = compute_time_ms + kv_read_ms + contention_ms + nvlink_bonus_ms
        base_ttft = max(1.0, base_ttft)

        return self._add_noise(base_ttft)

    def _compute_tpot(self, request: InferenceRequest, node: GPUNode) -> float:
        """
        Time per output token (ms).

        The decode phase is memory-bandwidth-bound: each step loads the full
        model weights once to generate one token.
        """
        model_bytes = MODEL_SIZE_BYTES.get(request.model_id, MODEL_SIZE_BYTES["default"])
        base_tpot = (
            model_bytes
            / (node.memory_bandwidth_gbps * 1e9 * BANDWIDTH_EFFICIENCY)
            * 1000.0
        )
        contention_ms = node.active_requests * 1.0   # smaller per-token effect
        base_tpot = max(0.5, base_tpot + contention_ms)
        return self._add_noise(base_tpot)

    def _sample_output_tokens(self, request: InferenceRequest) -> int:
        """Sample actual output length; typically shorter than max_output_tokens."""
        # Most responses are 50–100 % of max; heavy-tail for long generations
        mean = request.max_output_tokens * 0.65
        std = request.max_output_tokens * 0.20
        sampled = int(self._rng.gauss(mean, std))
        return max(1, min(request.max_output_tokens, sampled))

    def _compute_cost(self, node: GPUNode, total_tokens: int) -> float:
        """USD cost per 1M tokens based on GPU type."""
        rate = BASE_COST_PER_1M.get(node.gpu_type, 6.0)
        return rate  # Already expressed as $/1M tokens

    def _add_noise(self, value: float) -> float:
        noise = self._rng.gauss(0, NOISE_SIGMA * value)
        return max(0.1, value + noise)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        requests: list[InferenceRequest],
        node: GPUNode,
        kv_blocks_map: dict[str, list[KVBlock]],
        cache_hits: dict[str, bool],
    ) -> list[RequestResult]:
        """
        Execute a batch of requests on *node* with continuous batching.

        Decode phases are interleaved: all requests share the same memory
        bandwidth for their decode steps, increasing effective throughput
        but slightly increasing TPOT.
        """
        results = []
        batch_size = len(requests)
        batch_overhead_factor = 1.0 + 0.02 * (batch_size - 1)   # 2 % per extra request

        for req in requests:
            blocks = kv_blocks_map.get(req.request_id, [])
            hit = cache_hits.get(req.request_id, False)
            result = self.execute(req, node, blocks, hit)
            # Apply batching overhead to TPOT (shared bandwidth)
            result.tpot_ms *= batch_overhead_factor
            result.total_latency_ms = (
                result.ttft_ms
                + result.tpot_ms * result.output_tokens
                + result.queue_wait_ms
            )
            results.append(result)
        return results
