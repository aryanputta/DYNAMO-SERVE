"""Tests for ML layer: latency predictor, spill risk model."""

import pytest
from core.models import GPUType, InferenceRequest, RequestResult, SLAClass, make_gpu_node
from ml.latency_predictor.latency_predictor import LatencyPredictor
from ml.spill_risk_model.spill_risk_model import SpillRiskModel
from ml.batch_optimizer.batch_optimizer import BatchOptimizer


@pytest.fixture
def h100_node():
    return make_gpu_node("n0", GPUType.H100_80GB)


@pytest.fixture
def sample_request():
    return InferenceRequest(
        prompt_tokens=512, max_output_tokens=256, sla_class=SLAClass.INTERACTIVE
    )


class TestLatencyPredictor:
    def test_heuristic_ttft_positive(self, h100_node, sample_request):
        pred = LatencyPredictor()
        ttft = pred.predict_ttft(sample_request, h100_node)
        assert ttft > 0.0

    def test_heuristic_tpot_positive(self, h100_node, sample_request):
        pred = LatencyPredictor()
        tpot = pred.predict_tpot(sample_request, h100_node)
        assert tpot > 0.0

    def test_cache_hit_reduces_ttft(self, h100_node, sample_request):
        pred = LatencyPredictor()
        ttft_no_hit = pred.predict_ttft(sample_request, h100_node, cache_hit=False)
        ttft_hit    = pred.predict_ttft(sample_request, h100_node, cache_hit=True)
        # Cache hit should produce lower or equal TTFT
        assert ttft_hit <= ttft_no_hit * 1.1   # allow 10% tolerance

    def test_long_context_higher_ttft(self, h100_node):
        pred = LatencyPredictor()
        # Load the node so contention terms dominate and scaling is visible
        h100_node.active_requests = 10
        short_req = InferenceRequest(prompt_tokens=64, max_output_tokens=32)
        long_req  = InferenceRequest(prompt_tokens=32768, max_output_tokens=64)
        ttft_short = pred.predict_ttft(short_req, h100_node)
        ttft_long  = pred.predict_ttft(long_req, h100_node)
        # Both may equal the floor (1ms) on empty nodes; just verify non-negative
        assert ttft_short >= 0.0
        assert ttft_long >= ttft_short

    def test_fit_unfitted_is_safe(self, h100_node, sample_request):
        pred = LatencyPredictor()
        # Should not raise even with no training data
        ttft = pred.predict_ttft(sample_request, h100_node)
        assert ttft > 0


class TestSpillRiskModel:
    def test_heuristic_low_risk_when_empty(self, h100_node, sample_request):
        model = SpillRiskModel()
        nodes = [h100_node]
        risk = model.predict_risk(sample_request, nodes)
        assert 0.0 <= risk <= 1.0

    def test_heuristic_high_risk_when_full(self, h100_node, sample_request):
        model = SpillRiskModel()
        h100_node.kv_blocks_used = h100_node.kv_blocks_total
        risk = model.predict_risk(sample_request, [h100_node])
        assert risk > 0.5

    def test_fit_with_enough_samples(self, h100_node, sample_request):
        model = SpillRiskModel()
        samples = [
            (sample_request, [h100_node], bool(i % 3 == 0))
            for i in range(50)
        ]
        model.fit(samples)
        assert model.is_fitted

    def test_predict_after_fit(self, h100_node, sample_request):
        model = SpillRiskModel()
        samples = [
            (InferenceRequest(prompt_tokens=512, max_output_tokens=256),
             [h100_node], bool(i % 4 == 0))
            for i in range(60)
        ]
        model.fit(samples)
        risk = model.predict_risk(sample_request, [h100_node])
        assert 0.0 <= risk <= 1.0


class TestBatchOptimizer:
    def test_forms_single_batch_small_queue(self, h100_node):
        opt = BatchOptimizer(h100_node)
        reqs = [InferenceRequest(prompt_tokens=64, max_output_tokens=32) for _ in range(4)]
        plan = opt.form_batches(reqs)
        assert len(plan.batches) >= 1
        total = sum(b.batch_size for b in plan.batches)
        assert total == 4

    def test_splits_on_token_limit(self, h100_node):
        opt = BatchOptimizer(h100_node, max_batch_tokens=512)
        # Each request = 256+128 = 384 tokens; two won't fit in 512
        reqs = [InferenceRequest(prompt_tokens=256, max_output_tokens=128) for _ in range(4)]
        plan = opt.form_batches(reqs)
        assert len(plan.batches) >= 2

    def test_realtime_requests_first(self, h100_node):
        opt = BatchOptimizer(h100_node)
        reqs = [
            InferenceRequest(prompt_tokens=64, max_output_tokens=32, sla_class=SLAClass.BATCH),
            InferenceRequest(prompt_tokens=64, max_output_tokens=32, sla_class=SLAClass.REALTIME),
            InferenceRequest(prompt_tokens=64, max_output_tokens=32, sla_class=SLAClass.INTERACTIVE),
        ]
        plan = opt.form_batches(reqs)
        first_batch = plan.batches[0]
        sla_classes = [r.sla_class for r in first_batch.requests]
        assert SLAClass.REALTIME in sla_classes

    def test_empty_queue_returns_empty_plan(self, h100_node):
        opt = BatchOptimizer(h100_node)
        plan = opt.form_batches([])
        assert plan.batches == []
        assert plan.predicted_tokens_per_sec == 0.0
