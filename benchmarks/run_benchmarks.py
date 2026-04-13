"""
DYNAMO-SERVE Benchmark Runner.

Runs all scheduling policies against all benchmark scenarios and produces:
  - A rich console summary table
  - Per-scenario CSV files in --output directory
  - An aggregate comparison JSON

Usage::

    python benchmarks/run_benchmarks.py --all --output results/
    python benchmarks/run_benchmarks.py --scenario burst_traffic
    python benchmarks/run_benchmarks.py --scenario long_context --scheduler kv_aware
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
from pathlib import Path

# Make sure the repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from benchmarks.harness import BenchmarkHarness
from control_plane.kv_cache_manager.kv_cache_manager import KVCacheManager
from control_plane.scheduler.kv_aware_scheduler import KVAwareSLAScheduler
from control_plane.scheduler.least_loaded import LeastLoadedScheduler
from control_plane.scheduler.memory_aware import MemoryAwareScheduler
from control_plane.scheduler.round_robin import RoundRobinScheduler
from control_plane.sla_policy.sla_policy import SLAPolicy
from core.models import SchedulerMetrics, make_gpu_node, GPUType
from simulator.gpu_topology.topology import make_heterogeneous_cluster, make_nvlink_rack
from simulator.workload_replay.workload_generator import PROFILES, WorkloadGenerator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Colour helpers (works on most terminals) ──
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"


def _col(text, colour):
    return f"{colour}{text}{RESET}"


# ---------------------------------------------------------------------------
# Cluster factory
# ---------------------------------------------------------------------------

def build_cluster(topology: str = "heterogeneous"):
    """Return (nodes, kv_manager) for the chosen topology."""
    if topology == "nvlink":
        topo = make_nvlink_rack(num_nodes=8)
    else:
        topo = make_heterogeneous_cluster(n_h100=4, n_a100=4, n_a10=4)

    nodes = topo.nodes
    kv_manager = KVCacheManager(nodes)
    return nodes, kv_manager


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_schedulers(nodes, kv_manager, sla_policy):
    """Build fresh copies of all four schedulers from the same cluster state."""
    return {
        "round_robin":   RoundRobinScheduler(copy.deepcopy(nodes)),
        "least_loaded":  LeastLoadedScheduler(copy.deepcopy(nodes)),
        "memory_aware":  MemoryAwareScheduler(copy.deepcopy(nodes)),
        "kv_aware":      KVAwareSLAScheduler(
            copy.deepcopy(nodes),
            KVCacheManager(copy.deepcopy(nodes)),
            sla_policy=sla_policy,
        ),
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_name: str,
    scheduler_names: list[str],
    topology: str = "heterogeneous",
    verbose: bool = False,
) -> list[SchedulerMetrics]:
    profile = PROFILES.get(scenario_name)
    if profile is None:
        print(f"Unknown scenario '{scenario_name}'. Available: {list(PROFILES)}")
        sys.exit(1)

    gen = WorkloadGenerator(profile, seed=42)
    requests = gen.generate()

    if verbose:
        print(f"\n{BOLD}{CYAN}Scenario: {scenario_name}{RESET}  "
              f"({len(requests)} requests, {profile.duration_s:.0f}s window)")

    sla_policy = SLAPolicy()
    nodes_base, kv_base = build_cluster(topology)
    schedulers = build_schedulers(nodes_base, kv_base, sla_policy)

    all_metrics: list[SchedulerMetrics] = []
    for sched_name, scheduler in schedulers.items():
        if scheduler_names and sched_name not in scheduler_names:
            continue

        # Each scheduler gets fresh nodes + KV manager
        fresh_nodes, fresh_kv = build_cluster(topology)
        if sched_name == "kv_aware":
            sched = KVAwareSLAScheduler(fresh_nodes, fresh_kv, sla_policy=SLAPolicy())
        elif sched_name == "round_robin":
            sched = RoundRobinScheduler(fresh_nodes)
        elif sched_name == "least_loaded":
            sched = LeastLoadedScheduler(fresh_nodes)
        else:
            sched = MemoryAwareScheduler(fresh_nodes)

        harness = BenchmarkHarness(sched, fresh_nodes, fresh_kv)
        metrics = harness.run(copy.deepcopy(requests), scenario_name=scenario_name)
        all_metrics.append(metrics)

        if verbose:
            _print_metrics_row(metrics)

    return all_metrics


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_metrics_row(m: SchedulerMetrics) -> None:
    sla_pct = (1 - m.sla_violation_rate) * 100
    sla_col = GREEN if sla_pct >= 95 else (YELLOW if sla_pct >= 80 else RED)
    print(
        f"  {BOLD}{m.scheduler_name:<18}{RESET}"
        f"  p99_ttft={m.p99_ttft_ms:>8.1f}ms"
        f"  p99_tpot={m.p99_tpot_ms:>7.1f}ms"
        f"  tps={m.mean_tokens_per_sec:>6.0f}"
        f"  cache={m.cache_hit_rate:>5.1%}"
        f"  spills={m.total_kv_spills:>4d}"
        f"  reject={m.rejection_rate:>5.1%}"
        f"  sla_ok={_col(f'{sla_pct:.1f}%', sla_col)}"
        f"  fairness={m.fairness_index:.3f}"
    )


def print_comparison_table(all_metrics: list[SchedulerMetrics]) -> None:
    """Print a rich side-by-side comparison table."""
    cols = [
        ("Scheduler",       lambda m: m.scheduler_name),
        ("Scenario",        lambda m: m.scenario),
        ("Requests",        lambda m: str(m.total_requests)),
        ("Completed",       lambda m: str(m.completed_requests)),
        ("Rejected%",       lambda m: f"{m.rejection_rate:.1%}"),
        ("p50 TTFT(ms)",    lambda m: f"{m.p50_ttft_ms:.1f}"),
        ("p99 TTFT(ms)",    lambda m: f"{m.p99_ttft_ms:.1f}"),
        ("p50 TPOT(ms)",    lambda m: f"{m.p50_tpot_ms:.1f}"),
        ("p99 TPOT(ms)",    lambda m: f"{m.p99_tpot_ms:.1f}"),
        ("tok/s",           lambda m: f"{m.mean_tokens_per_sec:.0f}"),
        ("Cache Hit%",      lambda m: f"{m.cache_hit_rate:.1%}"),
        ("KV Spills",       lambda m: str(m.total_kv_spills)),
        ("SLA OK%",         lambda m: f"{(1-m.sla_violation_rate)*100:.1f}"),
        ("Fairness",        lambda m: f"{m.fairness_index:.3f}"),
        ("$/1M tok",        lambda m: f"{m.cost_per_1m_tokens:.2f}"),
    ]

    widths = [max(len(h), max((len(fn(m)) for m in all_metrics), default=0)) + 2
              for h, fn in cols]

    sep = "+" + "+".join("-" * w for w in widths) + "+"
    header = "|" + "|".join(f" {h:<{w-1}}" for (h, _), w in zip(cols, widths)) + "|"

    print(f"\n{BOLD}{'='*len(sep)}{RESET}")
    print(f"{BOLD}  DYNAMO-SERVE Benchmark Results{RESET}")
    print(f"{BOLD}{'='*len(sep)}{RESET}")
    print(sep)
    print(header)
    print(sep)
    for m in all_metrics:
        row = "|" + "|".join(f" {fn(m):<{w-1}}" for (_, fn), w in zip(cols, widths)) + "|"
        print(row)
    print(sep)


def save_csv(metrics_list: list[SchedulerMetrics], output_dir: Path, scenario: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scenario}.csv"
    if not metrics_list:
        return
    fieldnames = list(metrics_list[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m.to_dict())
    print(f"  CSV saved: {path}")


def save_summary_json(all_metrics: list[SchedulerMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    data = [m.to_dict() for m in all_metrics]
    path.write_text(json.dumps(data, indent=2))
    print(f"  Summary JSON: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DYNAMO-SERVE benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(PROFILES) + ["all"],
        default="all",
        help="Workload scenario to run (default: all)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["round_robin", "least_loaded", "memory_aware", "kv_aware", "all"],
        default="all",
        help="Scheduler to benchmark (default: all)",
    )
    parser.add_argument(
        "--topology",
        choices=["heterogeneous", "nvlink"],
        default="heterogeneous",
        help="Cluster topology (default: heterogeneous)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for CSV/JSON results",
    )
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    scenarios = list(PROFILES) if (args.all or args.scenario == "all") else [args.scenario]
    sched_names = [] if args.scheduler == "all" else [args.scheduler]
    output_dir = Path(args.output)

    all_metrics: list[SchedulerMetrics] = []

    for scenario in scenarios:
        print(f"\n{BOLD}Running scenario: {CYAN}{scenario}{RESET}")
        metrics = run_scenario(
            scenario, sched_names, topology=args.topology, verbose=args.verbose
        )
        save_csv(metrics, output_dir, scenario)
        all_metrics.extend(metrics)

    print_comparison_table(all_metrics)
    save_summary_json(all_metrics, output_dir)
    print(f"\n{GREEN}Benchmark complete.{RESET} Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
