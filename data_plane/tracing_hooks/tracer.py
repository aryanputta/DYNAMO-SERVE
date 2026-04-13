"""
Request Tracer.

Collects per-request and per-node telemetry throughout the serving pipeline.
Produces Prometheus-compatible metrics and can emit structured JSON traces
for offline analysis in the benchmark harness.

Events recorded:
  - request_arrived
  - admission_decision (accept/queue/reject)
  - scheduled (node assigned)
  - prefill_start / prefill_done   (TTFT boundary)
  - decode_start / decode_done
  - kv_cache_hit / kv_cache_miss
  - kv_eviction
  - request_complete / request_failed
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """A single timestamped event in the lifecycle of a request."""

    event: str
    request_id: str
    timestamp: float = field(default_factory=time.monotonic)
    node_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RequestTracer:
    """
    Thread-safe tracer that collects events from all subsystems.

    The tracer is intentionally lightweight – it just appends events
    to an in-memory list and lets the benchmark harness drain them.

    For production use, events would be shipped to OpenTelemetry or Jaeger.
    """

    def __init__(self, max_events: int = 100_000) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._max_events = max_events

        # Counters for Prometheus-style metrics
        self._counters: dict[str, int] = {
            "requests_arrived": 0,
            "requests_accepted": 0,
            "requests_queued": 0,
            "requests_rejected": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "kv_cache_hits": 0,
            "kv_cache_misses": 0,
            "kv_evictions": 0,
            "kv_spills": 0,
        }
        self._latencies: dict[str, list[float]] = {
            "ttft_ms": [],
            "tpot_ms": [],
            "total_ms": [],
            "queue_wait_ms": [],
        }

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record(
        self,
        event: str,
        request_id: str,
        node_id: Optional[str] = None,
        **metadata,
    ) -> None:
        ev = TraceEvent(
            event=event,
            request_id=request_id,
            node_id=node_id,
            metadata=metadata,
        )
        with self._lock:
            if len(self._events) < self._max_events:
                self._events.append(ev)
            self._update_counters(ev)

    def record_result(self, result) -> None:
        """Convenience method to ingest a RequestResult."""
        from core.models import RequestResult
        assert isinstance(result, RequestResult)

        if result.rejected:
            self.record("request_rejected", result.request_id, reason=result.rejection_reason)
            return

        if not result.success:
            self.record("request_failed", result.request_id, node_id=result.gpu_node_id)
            return

        self.record(
            "request_completed",
            result.request_id,
            node_id=result.gpu_node_id,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            total_ms=result.total_latency_ms,
        )
        with self._lock:
            self._latencies["ttft_ms"].append(result.ttft_ms)
            self._latencies["tpot_ms"].append(result.tpot_ms)
            self._latencies["total_ms"].append(result.total_latency_ms)
            self._latencies["queue_wait_ms"].append(result.queue_wait_ms)
            self._counters["requests_completed"] += 1
            if result.kv_cache_hit:
                self._counters["kv_cache_hits"] += 1
            else:
                self._counters["kv_cache_misses"] += 1
            if result.kv_spilled:
                self._counters["kv_spills"] += 1

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def latency_percentiles(self, key: str = "ttft_ms") -> dict[str, float]:
        import numpy as np
        with self._lock:
            data = list(self._latencies.get(key, []))
        if not data:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0}
        arr = np.array(data)
        return {
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
            "p99":  float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "max":  float(np.max(arr)),
        }

    def cache_hit_rate(self) -> float:
        with self._lock:
            hits = self._counters["kv_cache_hits"]
            misses = self._counters["kv_cache_misses"]
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def all_latencies(self) -> dict[str, list[float]]:
        with self._lock:
            return {k: list(v) for k, v in self._latencies.items()}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def dump_json(self, path: str | Path) -> None:
        with self._lock:
            data = [e.to_dict() for e in self._events]
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Trace written to %s (%d events)", path, len(data))

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            for k in self._counters:
                self._counters[k] = 0
            for k in self._latencies:
                self._latencies[k] = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_counters(self, ev: TraceEvent) -> None:
        mapping = {
            "request_arrived":   "requests_arrived",
            "request_accepted":  "requests_accepted",
            "request_queued":    "requests_queued",
            "request_rejected":  "requests_rejected",
            "request_failed":    "requests_failed",
            "kv_cache_hit":      "kv_cache_hits",
            "kv_cache_miss":     "kv_cache_misses",
            "kv_eviction":       "kv_evictions",
            "kv_spill":          "kv_spills",
        }
        key = mapping.get(ev.event)
        if key:
            self._counters[key] = self._counters.get(key, 0) + 1
