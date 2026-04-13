"""
Spill Risk Model.

Predicts the probability that accepting an incoming request will cause
a KV cache spill event (blocks needed > blocks available after eviction).

Used by the AdmissionController in ML-guided mode to pre-emptively queue
or reject requests before they cause memory pressure across all nodes.

Features:
  - cluster_memory_pressure      (mean across nodes)
  - cluster_kv_utilization       (mean KV block usage)
  - request_kv_blocks_needed
  - request_prompt_tokens (log)
  - max_node_kv_free             (headroom on best node)
  - active_requests_total
  - sla_class_enc
  - is_long_context

Target: binary (0 = no spill, 1 = spill occurred)
Model: LogisticRegression with polynomial features (degree 2)
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


def _featurize(request: InferenceRequest, nodes: list[GPUNode]) -> np.ndarray:
    if not nodes:
        return np.zeros(8, dtype=np.float32)

    pressures = [n.memory_pressure for n in nodes]
    kv_utils = [n.kv_blocks_used / max(1, n.kv_blocks_total) for n in nodes]
    kv_frees = [n.kv_blocks_free for n in nodes]
    active_total = sum(n.active_requests for n in nodes)

    return np.array([
        float(np.mean(pressures)),
        float(np.mean(kv_utils)),
        math.log1p(request.kv_blocks_needed),
        math.log1p(request.prompt_tokens),
        math.log1p(max(kv_frees)),
        math.log1p(active_total),
        float(request.is_long_context),
        float(SLA_ENC.get(request.sla_class, 1)),
    ], dtype=np.float32)


class SpillRiskModel:
    """
    Binary classifier: will accepting this request cause a KV spill?

    The model is trained on (request, cluster_state) → spill_occurred pairs
    collected from simulation runs. In production it runs in the hot path
    of the AdmissionController so it must be fast (< 1 ms inference).
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        samples: list[tuple[InferenceRequest, list[GPUNode], bool]],
    ) -> "SpillRiskModel":
        """
        Fit on labelled samples.

        Args:
            samples: List of (request, node_snapshot, spilled) tuples.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        if len(samples) < 30:
            logger.warning("SpillRiskModel: too few samples (%d), skipping fit", len(samples))
            return self

        X, y = [], []
        for req, nodes, spilled in samples:
            X.append(_featurize(req, nodes))
            y.append(int(spilled))

        X = np.array(X)
        y = np.array(y)

        self._pipeline = Pipeline([
            ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        self._pipeline.fit(X, y)
        self._fitted = True

        spill_rate = y.mean()
        train_acc = self._pipeline.score(X, y)
        logger.info(
            "SpillRiskModel fitted: %d samples, base_spill_rate=%.2f, train_acc=%.3f",
            len(samples), spill_rate, train_acc,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_risk(self, request: InferenceRequest, nodes: list[GPUNode]) -> float:
        """Return probability (0–1) that this request will cause a spill."""
        if not self._fitted:
            return self._heuristic_risk(request, nodes)
        x = _featurize(request, nodes).reshape(1, -1)
        return float(self._pipeline.predict_proba(x)[0][1])

    def predict_spill(self, request: InferenceRequest, nodes: list[GPUNode]) -> bool:
        return self.predict_risk(request, nodes) > 0.5

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_risk(request: InferenceRequest, nodes: list[GPUNode]) -> float:
        """Simple rule-based spill risk when model is not fitted."""
        if not nodes:
            return 1.0
        max_free = max(n.kv_blocks_free for n in nodes)
        if max_free >= request.kv_blocks_needed * 2:
            return 0.05
        if max_free >= request.kv_blocks_needed:
            return 0.30
        return 0.90

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f)

    def load(self, path: str | Path) -> "SpillRiskModel":
        import pickle
        with open(path, "rb") as f:
            self._pipeline = pickle.load(f)
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted
