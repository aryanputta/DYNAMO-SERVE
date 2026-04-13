"""
vLLM Backend.

Wraps a running vLLM server (OpenAI-compatible API) and makes it a drop-in
replacement for MockGPURuntime. When VLLM_BASE_URL is set in the environment,
the serving system routes real inference requests to the actual model server
and measures wall-clock TTFT / TPOT from the streaming response.

If vLLM is not reachable, the backend automatically falls back to the mock
runtime so benchmarks can always run without GPU hardware.

Usage::

    export VLLM_BASE_URL=http://localhost:8000
    export VLLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

    backend = VLLMBackend()
    result  = backend.execute(request, node, kv_blocks=[])

Requirements (optional)::

    pip install openai>=1.30       # OpenAI-compatible client
    pip install vllm>=0.4          # for running the server locally
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from core.models import GPUNode, InferenceRequest, KVBlock, RequestResult
from data_plane.mock_gpu_runtime.gpu_runtime import MockGPURuntime

logger = logging.getLogger(__name__)

VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL", "")
VLLM_API_KEY   = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL     = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
VLLM_TIMEOUT_S = float(os.getenv("VLLM_TIMEOUT_S", "120"))


class VLLMBackend:
    """
    Real vLLM inference backend with automatic mock fallback.

    The backend streams the response token-by-token and records:
      - TTFT: time from request send → first token received
      - TPOT: (total_time - TTFT) / (output_tokens - 1)
    """

    def __init__(self, fallback_seed: Optional[int] = 42) -> None:
        self._mock = MockGPURuntime(seed=fallback_seed)
        self._vllm_available = self._check_vllm()

    # ------------------------------------------------------------------
    # Public API  (mirrors MockGPURuntime.execute)
    # ------------------------------------------------------------------

    def execute(
        self,
        request: InferenceRequest,
        node: GPUNode,
        kv_blocks: list[KVBlock],
        cache_hit: bool = False,
    ) -> RequestResult:
        if self._vllm_available and VLLM_BASE_URL:
            try:
                return self._execute_real(request, node, kv_blocks, cache_hit)
            except Exception as exc:
                logger.warning("vLLM call failed (%s), falling back to mock", exc)

        return self._mock.execute(request, node, kv_blocks, cache_hit)

    # ------------------------------------------------------------------
    # Real vLLM execution (streaming)
    # ------------------------------------------------------------------

    def _execute_real(
        self,
        request: InferenceRequest,
        node: GPUNode,
        kv_blocks: list[KVBlock],
        cache_hit: bool,
    ) -> RequestResult:
        from openai import OpenAI  # lazy import

        client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key=VLLM_API_KEY)

        # Build a synthetic prompt of the right token length
        # (in production you'd pass the actual prompt text)
        prompt = self._make_synthetic_prompt(request.prompt_tokens)
        arrival = request.arrival_time
        queue_wait_ms = max(0.0, (time.monotonic() - arrival) * 1000.0)

        send_time = time.monotonic()
        first_token_time: Optional[float] = None
        output_tokens = 0
        content_chunks: list[str] = []

        try:
            stream = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=request.max_output_tokens,
                stream=True,
                timeout=VLLM_TIMEOUT_S,
                extra_body={
                    "ignore_eos": False,
                    "skip_special_tokens": True,
                },
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta and first_token_time is None:
                    first_token_time = time.monotonic()
                content_chunks.append(delta)
                output_tokens += 1   # approximate: 1 chunk ≈ 1 token

        except Exception:
            raise

        end_time = time.monotonic()
        if first_token_time is None:
            first_token_time = end_time

        ttft_ms  = (first_token_time - send_time) * 1000.0
        total_ms = (end_time - send_time) * 1000.0
        tpot_ms  = (
            (total_ms - ttft_ms) / max(1, output_tokens - 1)
            if output_tokens > 1 else ttft_ms
        )

        spilled = len(kv_blocks) < request.kv_blocks_needed
        from data_plane.mock_gpu_runtime.gpu_runtime import BASE_COST_PER_1M
        cost = BASE_COST_PER_1M.get(node.gpu_type, 6.0)

        return RequestResult(
            request_id=request.request_id,
            success=True,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_ms + queue_wait_ms,
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
    # Helpers
    # ------------------------------------------------------------------

    def _check_vllm(self) -> bool:
        if not VLLM_BASE_URL:
            return False
        try:
            import urllib.request
            urllib.request.urlopen(f"{VLLM_BASE_URL}/health", timeout=3)
            logger.info("vLLM backend reachable at %s", VLLM_BASE_URL)
            return True
        except Exception:
            logger.info("vLLM not reachable at %s, using mock runtime", VLLM_BASE_URL)
            return False

    @staticmethod
    def _make_synthetic_prompt(token_count: int) -> str:
        """
        Generate a synthetic prompt of approximately *token_count* tokens.
        Uses a simple word-repetition scheme (≈ 1.3 tokens/word on average).
        """
        words_needed = max(1, int(token_count / 1.3))
        word = "inference "
        return (word * words_needed).strip()

    @property
    def is_real(self) -> bool:
        return self._vllm_available
