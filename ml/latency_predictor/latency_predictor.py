"""
Latency Predictor.

A lightweight ML model that predicts TTFT and TPOT for an
(InferenceRequest, GPUNode) pair *before* scheduling the request.

The predictor is trained on data collected from the mock GPU runtime
and used by the KVAwareSLAScheduler's PlacementEngine to pick the node
that will deliver the lowest latency for a given SLA class.

Architecture
------------
  Features (per request+node):
    - prompt_tokens (log-scaled)
    - max_output_tokens (log-scaled)
    - node_memory_pressure
    - node_compute_utilization
    - node_active_requests
    - node_kv_blocks_free_ratio
    - node_bandwidth_gbps (log-scaled)
    - node_compute_tflops (log-scaled)
    - nvlink_enabled (binary)
    - is_long_context (binary)
    - cache_hit (binary, estimated from KV manager hit rate)
    - sla_class_enc (0/1/2)

  Targets:
    - ttft_ms (log-transformed before training)
    - tpot_ms (log-transformed before training)

  Model: GradientBoostingRegressor (scikit-learn), one per target.
  Falls back to the heuristic formula when not fitted.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from core.models import GPUNode, InferenceRequest, RequestResult, SLAClass

logger = logging.getLogger(__name__)

SLA_ENC = {SLAClass.REALTIME: 0, SLAClass.INTERACTIVE: 1, SLAClass.BATCH: 2}


def _featurize(request: InferenceRequest, node: GPUNode, cache_hit: bool = False) -> np.ndarray:
    return np.array([
        math.log1p(request.prompt_tokens),
        math.log1p(request.max_output_tokens),
        node.memory_pressure,
        node.compute_utilization,
        node.active_requests,
        node.kv_blocks_free / max(1, node.kv_blocks_total),
        math.log1p(node.memory_bandwidth_gbps),
        math.log1p(node.compute_tflops),
        float(node.nvlink_enabled),
        float(request.is_long_context),
        float(cache_hit),
        SLA_ENC.get(request.sla_class, 1),
    ], dtype=np.float32)


class LatencyPredictor:
    """
    TTFT and TPOT predictor trained on empirical serving data.

    Usage::

        predictor = LatencyPredictor()
        predictor.fit(training_results)   # list[tuple[request, node, result]]
        ttft = predictor.predict_ttft(request, node)
        tpot = predictor.predict_tpot(request, node)
    """

    def __init__(self) -> None:
        self._ttft_model = None
        self._tpot_model = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        samples: list[tuple[InferenceRequest, GPUNode, RequestResult]],
        n_estimators: int = 100,
    ) -> "LatencyPredictor":
        """
        Train TTFT and TPOT models from observed serving data.

        Args:
            samples: List of (request, node, result) tuples collected from
                     the mock runtime or real serving.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if len(samples) < 20:
            logger.warning(
                "LatencyPredictor: only %d training samples, skipping fit", len(samples)
            )
            return self

        X, y_ttft, y_tpot = [], [], []
        for req, node, res in samples:
            X.append(_featurize(req, node, cache_hit=res.kv_cache_hit))
            y_ttft.append(math.log1p(max(0.1, res.ttft_ms)))
            y_tpot.append(math.log1p(max(0.1, res.tpot_ms)))

        X = np.array(X)
        y_ttft = np.array(y_ttft)
        y_tpot = np.array(y_tpot)

        self._ttft_model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.1, random_state=42
        )
        self._ttft_model.fit(X, y_ttft)

        self._tpot_model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.1, random_state=42
        )
        self._tpot_model.fit(X, y_tpot)

        self._fitted = True
        logger.info(
            "LatencyPredictor fitted on %d samples (TTFT train R²=%.3f, TPOT R²=%.3f)",
            len(samples),
            self._ttft_model.score(X, y_ttft),
            self._tpot_model.score(X, y_tpot),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_ttft(
        self, request: InferenceRequest, node: GPUNode, cache_hit: bool = False
    ) -> float:
        """Predict TTFT in milliseconds."""
        if not self._fitted:
            return self._heuristic_ttft(request, node, cache_hit)
        x = _featurize(request, node, cache_hit).reshape(1, -1)
        return float(math.expm1(self._ttft_model.predict(x)[0]))

    def predict_tpot(
        self, request: InferenceRequest, node: GPUNode
    ) -> float:
        """Predict TPOT in milliseconds."""
        if not self._fitted:
            return self._heuristic_tpot(request, node)
        x = _featurize(request, node).reshape(1, -1)
        return float(math.expm1(self._tpot_model.predict(x)[0]))

    def predict_total_latency(
        self, request: InferenceRequest, node: GPUNode, cache_hit: bool = False
    ) -> float:
        ttft = self.predict_ttft(request, node, cache_hit)
        tpot = self.predict_tpot(request, node)
        return ttft + tpot * request.max_output_tokens

    # ------------------------------------------------------------------
    # Fallback heuristics (no model)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_ttft(
        request: InferenceRequest, node: GPUNode, cache_hit: bool
    ) -> float:
        effective = max(1, int(request.prompt_tokens * (0.25 if cache_hit else 1.0)))
        bytes_per_token = 2 * 4096 * 2
        transfer_ms = (
            effective * bytes_per_token
            / (node.memory_bandwidth_gbps * 1e9 * 0.8)
            * 1000.0
        )
        contention_ms = node.active_requests * 5.0
        return max(1.0, transfer_ms + contention_ms)

    @staticmethod
    def _heuristic_tpot(request: InferenceRequest, node: GPUNode) -> float:
        model_bytes = 140 * 1024 ** 3
        base = model_bytes / (node.memory_bandwidth_gbps * 1e9 * 0.8) * 1000.0
        return max(0.5, base + node.active_requests * 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ttft": self._ttft_model, "tpot": self._tpot_model}, f)
        logger.info("LatencyPredictor saved to %s", path)

    def load(self, path: str | Path) -> "LatencyPredictor":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._ttft_model = obj["ttft"]
        self._tpot_model = obj["tpot"]
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted
