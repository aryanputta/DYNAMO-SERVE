"""Tests for core data models."""

import pytest
from core.models import (
    GPUType,
    GPUNode,
    InferenceRequest,
    RequestResult,
    SLAClass,
    TenantPriority,
    make_gpu_node,
)


class TestGPUNode:
    def test_kv_blocks_initialized(self):
        node = make_gpu_node("n0", GPUType.H100_80GB)
        assert node.kv_blocks_total > 0

    def test_memory_pressure_empty(self):
        node = make_gpu_node("n0", GPUType.H100_80GB)
        assert node.memory_pressure == pytest.approx(0.0)

    def test_can_fit_when_empty(self):
        node = make_gpu_node("n0", GPUType.H100_80GB)
        assert node.can_fit(100, 1.0)

    def test_allocate_and_release(self):
        node = make_gpu_node("n0", GPUType.H100_80GB)
        node.allocate(50, 2.0)
        assert node.kv_blocks_used == 50
        assert node.used_memory_gb == pytest.approx(2.0)
        assert node.active_requests == 1

        node.release(50, 2.0)
        assert node.kv_blocks_used == 0
        assert node.used_memory_gb == pytest.approx(0.0)
        assert node.active_requests == 0

    def test_cannot_over_allocate(self):
        node = make_gpu_node("n0", GPUType.A10_24GB)
        # A10 has 24 GB total; try to allocate 30 GB
        assert not node.can_fit(0, 30.0)

    def test_kv_blocks_free_decreases_on_alloc(self):
        node = make_gpu_node("n0", GPUType.H100_80GB)
        before = node.kv_blocks_free
        node.allocate(100, 0.5)
        assert node.kv_blocks_free == before - 100


class TestInferenceRequest:
    def test_kv_blocks_needed_short(self):
        req = InferenceRequest(prompt_tokens=128, max_output_tokens=64)
        # (128 + 64 + 15) // 16 = 192//16 = 12
        assert req.kv_blocks_needed == 12

    def test_kv_blocks_needed_minimum_one(self):
        req = InferenceRequest(prompt_tokens=1, max_output_tokens=1)
        assert req.kv_blocks_needed >= 1

    def test_is_long_context_false(self):
        req = InferenceRequest(prompt_tokens=512, max_output_tokens=256)
        assert not req.is_long_context

    def test_is_long_context_true(self):
        req = InferenceRequest(prompt_tokens=16384, max_output_tokens=512)
        assert req.is_long_context

    def test_estimated_memory_gb_positive(self):
        req = InferenceRequest(prompt_tokens=1024, max_output_tokens=512)
        assert req.estimated_memory_gb > 0

    def test_request_id_unique(self):
        r1 = InferenceRequest(prompt_tokens=100, max_output_tokens=50)
        r2 = InferenceRequest(prompt_tokens=100, max_output_tokens=50)
        assert r1.request_id != r2.request_id


class TestRequestResult:
    def test_total_tokens(self):
        r = RequestResult(request_id="x", prompt_tokens=100, output_tokens=50)
        assert r.total_tokens == 150

    def test_tokens_per_second(self):
        r = RequestResult(
            request_id="x",
            output_tokens=100,
            total_latency_ms=1000.0,
        )
        assert r.tokens_per_second == pytest.approx(100.0)

    def test_sla_met_interactive(self):
        r = RequestResult(
            request_id="x",
            ttft_ms=500.0,
            tpot_ms=80.0,
            total_latency_ms=1000.0,
        )
        assert r.sla_met(SLAClass.INTERACTIVE)

    def test_sla_violated_realtime(self):
        r = RequestResult(
            request_id="x",
            ttft_ms=500.0,   # > 200ms realtime budget
            tpot_ms=30.0,
            total_latency_ms=1000.0,
        )
        assert not r.sla_met(SLAClass.REALTIME)
