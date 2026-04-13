"""
DYNAMO-SERVE Dashboard.

A Flask web application that serves:
  /            – Live cluster overview (GPU utilisation, memory, KV cache)
  /api/metrics – JSON snapshot of all metrics
  /api/nodes   – Per-node GPU state
  /api/history – Rolling 5-minute latency history
  /metrics     – Prometheus text exposition

Run::

    python dashboard/app.py [--host 0.0.0.0] [--port 8080] [--demo]

With --demo the dashboard generates synthetic live data from a running
benchmark so you can see the UI without a real serving cluster.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, Response, jsonify, render_template_string

# ── Prometheus client ──
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (module-level singletons)
# ---------------------------------------------------------------------------

if PROM_AVAILABLE:
    prom_requests_total     = Counter("dynamo_requests_total", "Total inference requests", ["status"])
    prom_ttft_histogram     = Histogram("dynamo_ttft_ms", "TTFT latency (ms)", buckets=[50,100,200,500,1000,2000,5000])
    prom_tpot_histogram     = Histogram("dynamo_tpot_ms", "TPOT latency (ms)", buckets=[10,20,50,100,200,500])
    prom_gpu_memory         = Gauge("dynamo_gpu_memory_pressure", "GPU memory pressure", ["node_id"])
    prom_kv_utilization     = Gauge("dynamo_kv_utilization", "KV cache utilization", ["node_id"])
    prom_kv_hit_rate        = Gauge("dynamo_kv_hit_rate", "KV cache hit rate")
    prom_tokens_per_sec     = Gauge("dynamo_tokens_per_sec", "Token throughput")
    prom_active_requests    = Gauge("dynamo_active_requests", "Active requests", ["node_id"])
    prom_sla_violations     = Counter("dynamo_sla_violations_total", "SLA violations", ["sla_class"])


# ---------------------------------------------------------------------------
# In-memory state (updated by the demo loop or external calls)
# ---------------------------------------------------------------------------

class DashboardState:
    """Thread-safe store for live metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.nodes: list[dict] = []
        self.cluster: dict = {
            "total_requests": 0,
            "completed_requests": 0,
            "rejected_requests": 0,
            "tokens_per_sec": 0.0,
            "cache_hit_rate": 0.0,
            "p99_ttft_ms": 0.0,
            "p99_tpot_ms": 0.0,
            "sla_violation_rate": 0.0,
            "total_kv_spills": 0,
            "scheduler": "kv_aware_sla",
            "uptime_s": 0.0,
        }
        # Rolling 5-min latency history (10-s buckets)
        self.ttft_history: deque = deque(maxlen=30)
        self.tpot_history: deque = deque(maxlen=30)
        self.tps_history:  deque = deque(maxlen=30)
        self.start_time = time.monotonic()

    def update(self, nodes: list[dict], cluster: dict) -> None:
        with self._lock:
            self.nodes = nodes
            self.cluster.update(cluster)
            self.cluster["uptime_s"] = time.monotonic() - self.start_time

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "nodes": list(self.nodes),
                "cluster": dict(self.cluster),
                "ttft_history": list(self.ttft_history),
                "tpot_history": list(self.tpot_history),
                "tps_history":  list(self.tps_history),
            }

    def push_history(self, ttft_p99: float, tpot_p99: float, tps: float) -> None:
        with self._lock:
            ts = time.monotonic() - self.start_time
            self.ttft_history.append({"t": ts, "v": ttft_p99})
            self.tpot_history.append({"t": ts, "v": tpot_p99})
            self.tps_history.append({"t": ts, "v": tps})


STATE = DashboardState()


# ---------------------------------------------------------------------------
# Demo background thread
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Continuously runs a mini benchmark and updates STATE."""
    from benchmarks.harness import BenchmarkHarness
    from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
    from control_plane.scheduler.kv_aware_scheduler import KVAwareSLAScheduler
    from control_plane.sla_policy.sla_policy import SLAPolicy
    from simulator.gpu_topology.topology import make_heterogeneous_cluster
    from simulator.workload_replay.workload_generator import WorkloadGenerator, PROFILES
    import random

    topo = make_heterogeneous_cluster(n_h100=4, n_a100=4, n_a10=4)
    nodes = topo.nodes
    kv_manager = KVCacheManager(nodes)
    sla_policy = SLAPolicy()
    scheduler = KVAwareSLAScheduler(nodes, kv_manager, sla_policy=sla_policy)

    scenario_cycle = list(PROFILES.keys())
    rng = random.Random(0)
    total_req = total_done = total_rej = total_spills = 0
    all_ttfts: list[float] = []
    all_tpots: list[float] = []
    all_tps: list[float] = []

    while True:
        # Pick a random scenario for variety
        scenario = rng.choice(scenario_cycle)
        profile = copy.deepcopy(PROFILES[scenario])
        profile.duration_s = 10.0   # short bursts for live demo
        gen = WorkloadGenerator(profile, seed=rng.randint(0, 9999))
        requests = gen.generate()

        # Fresh cluster snapshot for each iteration
        fresh_topo = make_heterogeneous_cluster(n_h100=4, n_a100=4, n_a10=4)
        fresh_nodes = fresh_topo.nodes
        fresh_kv = KVCacheManager(fresh_nodes)
        fresh_sched = KVAwareSLAScheduler(fresh_nodes, fresh_kv, sla_policy=SLAPolicy())

        harness = BenchmarkHarness(fresh_sched, fresh_nodes, fresh_kv)
        metrics = harness.run(requests, scenario_name=scenario)

        # Accumulate
        total_req   += metrics.total_requests
        total_done  += metrics.completed_requests
        total_rej   += metrics.rejected_requests
        total_spills += metrics.total_kv_spills
        all_ttfts.append(metrics.p99_ttft_ms)
        all_tpots.append(metrics.p99_tpot_ms)
        all_tps.append(metrics.mean_tokens_per_sec)

        import numpy as np
        p99_ttft = float(np.percentile(all_ttfts, 99)) if all_ttfts else 0.0
        p99_tpot = float(np.percentile(all_tpots, 99)) if all_tpots else 0.0
        mean_tps  = float(np.mean(all_tps)) if all_tps else 0.0

        # Build per-node snapshots
        node_data = []
        for n in fresh_nodes:
            node_data.append({
                "node_id": n.node_id,
                "gpu_type": n.gpu_type.value,
                "memory_pressure": round(n.memory_pressure * 100, 1),
                "kv_utilization": round(n.kv_blocks_used / max(1, n.kv_blocks_total) * 100, 1),
                "active_requests": n.active_requests,
                "nvlink": n.nvlink_enabled,
                "total_memory_gb": n.total_memory_gb,
                "used_memory_gb": round(n.used_memory_gb, 1),
            })

        cluster_data = {
            "total_requests": total_req,
            "completed_requests": total_done,
            "rejected_requests": total_rej,
            "tokens_per_sec": round(mean_tps, 1),
            "cache_hit_rate": round(metrics.cache_hit_rate * 100, 1),
            "p99_ttft_ms": round(p99_ttft, 1),
            "p99_tpot_ms": round(p99_tpot, 1),
            "sla_violation_rate": round(metrics.sla_violation_rate * 100, 1),
            "total_kv_spills": total_spills,
            "scheduler": "kv_aware_sla",
            "scenario": scenario,
        }
        STATE.update(node_data, cluster_data)
        STATE.push_history(p99_ttft, p99_tpot, mean_tps)

        # Update Prometheus metrics
        if PROM_AVAILABLE:
            prom_kv_hit_rate.set(metrics.cache_hit_rate)
            prom_tokens_per_sec.set(mean_tps)
            for nd in node_data:
                prom_gpu_memory.labels(nd["node_id"]).set(nd["memory_pressure"] / 100)
                prom_kv_utilization.labels(nd["node_id"]).set(nd["kv_utilization"] / 100)
                prom_active_requests.labels(nd["node_id"]).set(nd["active_requests"])

        time.sleep(8)


# ---------------------------------------------------------------------------
# HTML template (single-page dashboard)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DYNAMO-SERVE — LLM Inference Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --dim: #8b949e; --green: #3fb950;
    --yellow: #d29922; --red: #f85149; --blue: #58a6ff; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', monospace; padding: 20px; }
  h1 { font-size: 1.4rem; color: var(--blue); margin-bottom: 4px; }
  .subtitle { color: var(--dim); font-size: 0.8rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card .label { font-size: 0.7rem; color: var(--dim); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
  .card .value { font-size: 1.6rem; font-weight: 700; }
  .card .unit  { font-size: 0.75rem; color: var(--dim); margin-left: 4px; }
  .green { color: var(--green); } .red { color: var(--red); } .yellow { color: var(--yellow); } .blue { color: var(--blue); }
  .section { margin-bottom: 24px; }
  .section h2 { font-size: 1rem; color: var(--dim); margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { text-align: left; color: var(--dim); padding: 6px 10px; font-weight: normal; border-bottom: 1px solid var(--border); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  tr:hover td { background: #1c2128; }
  .bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 4px; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
  .badge-nvlink { background: #1f3a5f; color: var(--blue); }
  .badge-pcie   { background: #2a1f1f; color: var(--dim); }
  #chart-area canvas { width: 100% !important; }
  footer { color: var(--dim); font-size: 0.72rem; margin-top: 30px; text-align: center; }
</style>
</head>
<body>
<h1>⚡ DYNAMO-SERVE</h1>
<p class="subtitle">KV-Cache-Aware LLM Inference Infrastructure · Live Metrics</p>

<div class="grid" id="kpi-grid">
  <div class="card"><div class="label">Tokens / sec</div><div class="value blue" id="kpi-tps">—</div></div>
  <div class="card"><div class="label">p99 TTFT</div><div class="value" id="kpi-ttft">—<span class="unit">ms</span></div></div>
  <div class="card"><div class="label">p99 TPOT</div><div class="value" id="kpi-tpot">—<span class="unit">ms</span></div></div>
  <div class="card"><div class="label">KV Cache Hit%</div><div class="value green" id="kpi-hit">—</div></div>
  <div class="card"><div class="label">KV Spills</div><div class="value" id="kpi-spills">—</div></div>
  <div class="card"><div class="label">SLA Violations</div><div class="value" id="kpi-sla">—</div></div>
  <div class="card"><div class="label">Requests</div><div class="value" id="kpi-req">—</div></div>
  <div class="card"><div class="label">Rejected%</div><div class="value" id="kpi-rej">—</div></div>
</div>

<div class="section">
  <h2>GPU Node Status</h2>
  <table id="node-table">
    <thead><tr>
      <th>Node</th><th>GPU Type</th><th>Interconnect</th>
      <th>Memory Pressure</th><th>KV Cache Fill</th><th>Active Requests</th>
    </tr></thead>
    <tbody id="node-tbody"></tbody>
  </table>
</div>

<div class="section" id="chart-area">
  <h2>Latency History (p99)</h2>
  <canvas id="latency-chart" height="120"></canvas>
</div>

<footer>DYNAMO-SERVE · Inspired by NVIDIA Dynamo · Blackwell-era inference infrastructure</footer>

<script>
// ── Minimal canvas chart ──
const CHART_W = 900, CHART_H = 120, PAD = 30;
let ttftHistory = [], tpotHistory = [], tpsHistory = [];

function drawChart() {
  const canvas = document.getElementById('latency-chart');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = CHART_H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, CHART_H);

  const draw = (data, colour) => {
    if (data.length < 2) return;
    const maxV = Math.max(...data.map(d => d.v), 1);
    ctx.strokeStyle = colour; ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = PAD + (i / (data.length - 1)) * (canvas.width - 2 * PAD);
      const y = CHART_H - PAD - (d.v / maxV) * (CHART_H - 2 * PAD);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Label at last point
    const last = data[data.length - 1];
    const lx = canvas.width - PAD - 2;
    const ly = CHART_H - PAD - (last.v / maxV) * (CHART_H - 2 * PAD);
    ctx.fillStyle = colour; ctx.font = '11px monospace';
    ctx.fillText(last.v.toFixed(0) + 'ms', lx - 40, Math.max(12, ly - 4));
  };

  // Axes
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, CHART_H - PAD);
  ctx.lineTo(canvas.width - PAD, CHART_H - PAD);
  ctx.stroke();

  draw(ttftHistory, '#58a6ff');
  draw(tpotHistory, '#3fb950');

  // Legend
  ctx.font = '11px monospace';
  ctx.fillStyle = '#58a6ff'; ctx.fillText('▬ TTFT p99', PAD + 4, 14);
  ctx.fillStyle = '#3fb950'; ctx.fillText('▬ TPOT p99', PAD + 90, 14);
}

// ── Polling ──
function colourByValue(v, warn, crit) {
  if (v >= crit) return 'red';
  if (v >= warn) return 'yellow';
  return 'green';
}

function barColour(pct) {
  if (pct > 85) return '#f85149';
  if (pct > 65) return '#d29922';
  return '#3fb950';
}

async function poll() {
  try {
    const resp = await fetch('/api/metrics');
    const data = await resp.json();
    const c = data.cluster;
    const rej_pct = c.total_requests > 0 ? (c.rejected_requests / c.total_requests * 100) : 0;

    document.getElementById('kpi-tps').textContent   = (c.tokens_per_sec || 0).toLocaleString();
    document.getElementById('kpi-ttft').innerHTML    = `<span class="${colourByValue(c.p99_ttft_ms,500,1000)}">${c.p99_ttft_ms}</span><span class="unit">ms</span>`;
    document.getElementById('kpi-tpot').innerHTML    = `<span class="${colourByValue(c.p99_tpot_ms,80,150)}">${c.p99_tpot_ms}</span><span class="unit">ms</span>`;
    document.getElementById('kpi-hit').textContent   = (c.cache_hit_rate || 0) + '%';
    document.getElementById('kpi-spills').textContent = c.total_kv_spills || 0;
    document.getElementById('kpi-sla').innerHTML     = `<span class="${colourByValue(c.sla_violation_rate,5,15)}">${c.sla_violation_rate}%</span>`;
    document.getElementById('kpi-req').textContent   = (c.total_requests || 0).toLocaleString();
    document.getElementById('kpi-rej').innerHTML     = `<span class="${colourByValue(rej_pct,10,25)}">${rej_pct.toFixed(1)}%</span>`;

    // Node table
    const tbody = document.getElementById('node-tbody');
    tbody.innerHTML = '';
    (data.nodes || []).forEach(n => {
      const row = document.createElement('tr');
      const memCol = barColour(n.memory_pressure);
      const kvCol  = barColour(n.kv_utilization);
      row.innerHTML = `
        <td><code>${n.node_id}</code></td>
        <td>${n.gpu_type}</td>
        <td><span class="badge ${n.nvlink ? 'badge-nvlink' : 'badge-pcie'}">${n.nvlink ? 'NVLink' : 'PCIe'}</span></td>
        <td>
          ${n.memory_pressure}% (${n.used_memory_gb}/${n.total_memory_gb} GB)
          <div class="bar"><div class="bar-fill" style="width:${n.memory_pressure}%;background:${memCol}"></div></div>
        </td>
        <td>
          ${n.kv_utilization}%
          <div class="bar"><div class="bar-fill" style="width:${n.kv_utilization}%;background:${kvCol}"></div></div>
        </td>
        <td>${n.active_requests}</td>`;
      tbody.appendChild(row);
    });

    // History
    ttftHistory = data.ttft_history || [];
    tpotHistory = data.tpot_history || [];
    tpsHistory  = data.tps_history  || [];
    drawChart();
  } catch(e) { console.warn('Poll error:', e); }
}

poll();
setInterval(poll, 5000);
window.addEventListener('resize', drawChart);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(STATE.snapshot())


@app.route("/api/nodes")
def api_nodes():
    snap = STATE.snapshot()
    return jsonify(snap["nodes"])


@app.route("/api/history")
def api_history():
    snap = STATE.snapshot()
    return jsonify({
        "ttft": snap["ttft_history"],
        "tpot": snap["tpot_history"],
        "tps":  snap["tps_history"],
    })


@app.route("/metrics")
def prom_metrics():
    if not PROM_AVAILABLE:
        return Response("# prometheus_client not installed\n", content_type="text/plain")
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "uptime_s": time.monotonic() - STATE.start_time})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DYNAMO-SERVE dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run live demo benchmark in background (default: on)")
    args = parser.parse_args()

    if args.demo:
        t = threading.Thread(target=_run_demo, daemon=True)
        t.start()
        print(f"Demo benchmark thread started.")

    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Prometheus: http://{args.host}:{args.port}/metrics")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
