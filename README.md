# DYNAMO-SERVE

**Production-grade distributed LLM inference infrastructure with KV cache-aware scheduling, multi-node GPU placement, and SLA-based admission control.**

Inspired by NVIDIA Dynamo and Blackwell-era inference systems. Built to benchmark the real bottlenecks in LLM serving: GPU memory fragmentation, KV cache pressure, multi-tenant interference, and tail latency under bursty workloads.

---

## Why this exists

Most LLM serving projects stop at "hosted a model API" or "added Kubernetes autoscaling." The actual bottleneck at scale is:

- **KV cache growth** under long-context requests (128K+ tokens)
- **GPU memory fragmentation** across concurrent tenants
- **Multi-tenant interference** when BATCH jobs compete with REALTIME requests
- **Tail latency** (p99 TTFT) blowing SLAs under burst traffic
- **Cost per token** as a first-class metric alongside throughput

DYNAMO-SERVE simulates and benchmarks these exact problems, then shows how a KV-cache-aware scheduler beats naive baselines on every metric that matters.

---

## Architecture

```
Request → AdmissionController → KVAwareSLAScheduler → PlacementEngine
                                        │
                               KVCacheManager (prefix reuse, eviction)
                                        │
                              MockGPURuntime (prefill + decode simulation)
                                        │
                                   RequestTracer → Prometheus metrics
```

### Repository layout

```
control_plane/
  scheduler/              # 4 scheduling policies (round-robin → KV-aware SLA)
  admission_controller/   # Threshold + ML-guided admission gate
  kv_cache_manager/       # PagedAttention-style block manager + LRU/LFU eviction
  placement_engine/       # Multi-signal GPU placement scoring
  sla_policy/             # SLA budgets and violation tracking

data_plane/
  mock_gpu_runtime/       # Physics-based prefill/decode latency simulator
  tracing_hooks/          # Per-request tracer + Prometheus counters

simulator/
  workload_replay/        # Poisson + bursty workload generator (5 scenarios)
  gpu_topology/           # H100/B200/A100 cluster topology models
  nvlink_model/           # NVLink vs PCIe transfer cost
  cache_eviction/         # LRU/LFU/Belady cache simulation
  contention_model/       # Bandwidth + compute contention multipliers

ml/
  latency_predictor/      # GradientBoosting TTFT/TPOT predictor
  spill_risk_model/       # Logistic regression KV spill classifier
  batch_optimizer/        # Throughput-vs-latency batch formation

benchmarks/               # Harness + 5 scenario runners
dashboard/                # Flask live dashboard + Prometheus endpoint
tests/                    # 65 unit + integration tests
```

---

## Scheduling policies

| Policy | Description | Best for |
|---|---|---|
| `round_robin` | Naive circular assignment | Baseline only |
| `least_loaded` | Route to node with fewest active requests | Compute-bound workloads |
| `memory_aware` | Greedy: maximise KV cache headroom | Memory-pressure scenarios |
| `kv_aware_sla` | **KV prefix affinity + SLA tier + NVLink topology + ML prediction** | Everything |

### KV-Aware SLA Scheduler — what makes it different

1. **Prefix cache affinity** — routes requests to the node already holding their KV prefix, turning a cold-cache prefill into a warm-cache suffix-only decode
2. **SLA-class differentiation** — REALTIME requests jump the queue; BATCH requests yield under memory pressure
3. **ML-predicted TTFT** — trained GradientBoosting model predicts latency before committing a placement decision
4. **NVLink topology awareness** — avoids PCIe cross-node transfers for long-context requests on non-NVSwitch nodes
5. **Preemption** — evicts BATCH KV blocks to free room for REALTIME requests under memory pressure

---

## GPU topology support

| Topology | Description |
|---|---|
| `nvlink_rack` | 8× H100 NVL, 600 GB/s NVLink mesh |
| `blackwell_rack` | 8× B200, 1800 GB/s NVSwitch |
| `heterogeneous` | 4× H100 + 4× A100 + 4× A10 (mixed interconnect) |
| `multi_rack` | 2 NVLink racks connected by InfiniBand |

---

## Benchmark scenarios

| Scenario | Description | Key stress |
|---|---|---|
| `baseline` | 5 req/s, mixed SLA | Correctness check |
| `long_context` | 16K mean prompt tokens | KV fragmentation |
| `burst_traffic` | 5× spike every 30s | Admission control |
| `multi_tenant` | 20 tenants, Zipf skew | Fairness + interference |
| `moe_serving` | Mixtral-8×7B workload | MoE routing overhead |
| `high_load` | 20 req/s + 3× bursts | Combined stress |

---

## Metrics

Every benchmark run reports:

- **p50 / p95 / p99 TTFT** (time to first token, ms)
- **p50 / p95 / p99 TPOT** (time per output token, ms)
- **Tokens/sec** (decode throughput)
- **KV cache hit rate** (prefix reuse effectiveness)
- **KV spill count** (evictions that caused partial allocations)
- **Rejection rate** (requests turned away by admission control)
- **SLA violation rate** (requests that exceeded latency budget)
- **Jain's fairness index** (per-tenant throughput equity)
- **Cost per 1M tokens** (GPU-type-weighted serving cost)

---

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run all benchmarks (outputs results/ directory)
make bench

# Quick single-scenario run
python benchmarks/run_benchmarks.py --scenario burst_traffic --verbose

# Compare only KV-aware vs round-robin
python benchmarks/run_benchmarks.py --scenario long_context \
    --scheduler kv_aware --scheduler round_robin

# Live dashboard (demo benchmark runs in background)
make dashboard
# open http://localhost:8080

# Full test suite
make test
```

---

## Docker

```bash
docker-compose up
# Dashboard:   http://localhost:8080
# Prometheus:  http://localhost:9091
# Grafana:     http://localhost:3000  (admin/admin)
```

---

## Sample benchmark output

```
Scenario: burst_traffic  (312 requests, 120s window)

  round_robin         p99_ttft=  1842.3ms  p99_tpot=  94.2ms  tps=   218  cache= 0.0%  spills= 47  reject= 8.2%  sla_ok=62.3%  fairness=0.891
  least_loaded        p99_ttft=  1204.8ms  p99_tpot=  87.6ms  tps=   251  cache= 0.0%  spills= 31  reject= 5.1%  sla_ok=74.1%  fairness=0.903
  memory_aware        p99_ttft=   891.2ms  p99_tpot=  72.1ms  tps=   278  cache= 0.0%  spills= 12  reject= 3.8%  sla_ok=83.7%  fairness=0.921
  kv_aware_sla        p99_ttft=   412.7ms  p99_tpot=  48.3ms  tps=   334  cache=31.4%  spills=  3  reject= 2.1%  sla_ok=96.2%  fairness=0.967
```

---

## ML components

### Latency predictor
- **Model**: `GradientBoostingRegressor` (sklearn), one per target (TTFT, TPOT)
- **Features**: prompt length, output length, node bandwidth, compute utilisation, KV headroom, NVLink flag, cache hit flag, SLA class
- **Training data**: collected from mock runtime across all topology × scenario combinations
- **Usage**: PlacementEngine uses predictions to pick the lowest-latency node for REALTIME requests

### Spill risk classifier
- **Model**: `LogisticRegression` with degree-2 polynomial features
- **Features**: cluster memory pressure, KV utilisation, blocks needed, active requests
- **Usage**: AdmissionController uses predictions to queue/reject requests before they cause OOM

### Batch optimizer
- **Goal**: maximise tokens/sec while keeping TPOT within SLA budget
- **Strategy**: SLA-priority-ordered greedy knapsack over token budget
- **ML component**: `Ridge` regression for TPOT prediction as a function of batch size

---

## Design decisions

**Why PagedAttention-style block management?**
Fixed-size KV blocks prevent memory fragmentation from variable-length sequences. Prefix hashing lets blocks be shared across requests with the same system prompt or conversation history.

**Why simulate rather than run real inference?**
A physics-based simulator lets us stress-test scheduling policies at scale (10K+ requests, heterogeneous topologies) without GPU hardware. The simulator is calibrated to match vLLM benchmarks on H100.

**Why four schedulers instead of one?**
Comparing baselines quantifies exactly how much each mechanism (memory awareness, SLA differentiation, prefix caching, NVLink routing) contributes to the improvement. Resume-friendly: you can point to a concrete 4× improvement in p99 TTFT.

---

## Roadmap

- [ ] Real vLLM backend integration (replace mock runtime)
- [ ] RL-based scheduler (PPO, reward = SLA compliance - cost)
- [ ] Failure injection: node loss, memory spike, NVLink degradation
- [ ] Trace replay from real ChatML datasets (ShareGPT format)
- [ ] Expert-parallel MoE routing (per-expert KV cache partitioning)
- [ ] Grafana dashboard provisioning with pre-built panels
