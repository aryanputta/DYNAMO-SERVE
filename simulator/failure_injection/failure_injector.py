"""
Failure Injector.

Injects controlled faults into a running simulation to stress-test
scheduler resilience. Supported failure modes:

  NODE_LOSS        – Takes a GPU node offline; in-flight requests fail.
  MEMORY_SPIKE     – Injects a sudden increase in used_memory_gb (e.g., CUDA
                     fragmentation, rogue process) that reduces KV headroom.
  NVLINK_DEGRADE   – Reduces effective NVLink bandwidth by a configurable factor,
                     increasing cross-node transfer costs.
  SLOW_NODE        – Multiplies TPOT/TTFT by a factor (thermal throttle, PCIe
                     bandwidth contention).
  NETWORK_SLOWDOWN – Increases inter-node latency (IB congestion).

Failures can be:
  - Scheduled  – triggered at a specific simulation timestamp
  - Probabilistic – triggered with a per-step probability (chaos mode)
  - Permanent / transient – with optional auto-recovery after a duration
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from core.models import GPUNode

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    NODE_LOSS        = "node_loss"
    MEMORY_SPIKE     = "memory_spike"
    NVLINK_DEGRADE   = "nvlink_degrade"
    SLOW_NODE        = "slow_node"
    NETWORK_SLOWDOWN = "network_slowdown"


@dataclass
class FailureEvent:
    """Describes a single fault injection event."""

    failure_type:  FailureType
    target_node_id: Optional[str] = None  # None = random node
    trigger_time_s: Optional[float] = None  # None = probabilistic
    trigger_prob:   float = 0.0             # per-step probability
    duration_s:     Optional[float] = None  # None = permanent
    magnitude:      float = 1.0             # e.g. 0.5 = 50% memory spike, 0.3 = 30% NVLink degradation
    auto_recover:   bool = True
    label:          str = ""

    # Runtime state
    active:       bool = False
    triggered_at: Optional[float] = None
    _orig_state:  dict = field(default_factory=dict, repr=False)


class FailureInjector:
    """
    Injects and manages failure events during a simulation run.

    Usage::

        injector = FailureInjector(nodes, seed=42)
        injector.add(FailureEvent(FailureType.NODE_LOSS, trigger_time_s=30.0))
        injector.add(FailureEvent(FailureType.MEMORY_SPIKE, trigger_prob=0.01, magnitude=0.3))

        # In simulation loop:
        injector.step(sim_time=t)
        alive_nodes = injector.alive_nodes()
    """

    def __init__(
        self,
        nodes: list[GPUNode],
        seed: Optional[int] = 42,
        on_failure: Optional[Callable[[FailureEvent], None]] = None,
    ) -> None:
        self._nodes    = {n.node_id: n for n in nodes}
        self._events:  list[FailureEvent] = []
        self._dead:    set[str] = set()
        self._rng      = random.Random(seed)
        self._on_fail  = on_failure
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, event: FailureEvent) -> "FailureInjector":
        """Register a failure event. Returns self for chaining."""
        if not event.label:
            event.label = f"{event.failure_type.value}_{len(self._events)}"
        self._events.append(event)
        return self

    def step(self, sim_time: float) -> list[FailureEvent]:
        """
        Advance the injector by one simulation step.

        Returns list of events that fired this step.
        """
        fired: list[FailureEvent] = []

        for ev in self._events:
            # Recovery check
            if ev.active and ev.auto_recover and ev.duration_s is not None:
                if sim_time - (ev.triggered_at or 0) >= ev.duration_s:
                    self._recover(ev)
                    continue

            if ev.active:
                continue

            # Trigger check
            should_trigger = False
            if ev.trigger_time_s is not None:
                should_trigger = sim_time >= ev.trigger_time_s
            elif ev.trigger_prob > 0:
                should_trigger = self._rng.random() < ev.trigger_prob

            if should_trigger:
                self._apply(ev, sim_time)
                fired.append(ev)
                if self._on_fail:
                    self._on_fail(ev)

        return fired

    def alive_nodes(self) -> list[GPUNode]:
        """Return nodes that are currently online."""
        return [n for nid, n in self._nodes.items() if nid not in self._dead]

    def all_nodes(self) -> list[GPUNode]:
        return list(self._nodes.values())

    def is_alive(self, node_id: str) -> bool:
        return node_id not in self._dead

    def recover_all(self) -> None:
        """Immediately recover all active failures."""
        for ev in self._events:
            if ev.active:
                self._recover(ev)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Failure application
    # ------------------------------------------------------------------

    def _apply(self, ev: FailureEvent, sim_time: float) -> None:
        target = self._pick_target(ev)
        if target is None:
            return

        ev.active = True
        ev.triggered_at = sim_time
        node = self._nodes[target]

        if ev.failure_type == FailureType.NODE_LOSS:
            ev._orig_state = {
                "used_memory_gb": node.used_memory_gb,
                "kv_blocks_used": node.kv_blocks_used,
                "active_requests": node.active_requests,
            }
            # Simulate node going offline: max out memory, zero compute
            node.used_memory_gb = node.total_memory_gb
            node.kv_blocks_used = node.kv_blocks_total
            self._dead.add(target)
            logger.warning("[FAILURE] NODE_LOSS: %s at t=%.1fs", target, sim_time)

        elif ev.failure_type == FailureType.MEMORY_SPIKE:
            spike_gb = node.total_memory_gb * ev.magnitude
            ev._orig_state = {"used_memory_gb": node.used_memory_gb,
                              "kv_blocks_used": node.kv_blocks_used}
            node.used_memory_gb = min(node.total_memory_gb,
                                      node.used_memory_gb + spike_gb)
            # Proportionally reduce free KV blocks
            used_ratio = node.used_memory_gb / node.total_memory_gb
            node.kv_blocks_used = int(node.kv_blocks_total * used_ratio)
            logger.warning("[FAILURE] MEMORY_SPIKE +%.0f%% on %s at t=%.1fs",
                           ev.magnitude * 100, target, sim_time)

        elif ev.failure_type == FailureType.NVLINK_DEGRADE:
            ev._orig_state = {"nvlink_enabled": node.nvlink_enabled,
                              "memory_bandwidth_gbps": node.memory_bandwidth_gbps}
            node.memory_bandwidth_gbps *= (1.0 - ev.magnitude)
            logger.warning("[FAILURE] NVLINK_DEGRADE -%.0f%% bw on %s at t=%.1fs",
                           ev.magnitude * 100, target, sim_time)

        elif ev.failure_type == FailureType.SLOW_NODE:
            ev._orig_state = {"compute_tflops": node.compute_tflops,
                              "memory_bandwidth_gbps": node.memory_bandwidth_gbps}
            node.compute_tflops *= (1.0 - ev.magnitude)
            node.memory_bandwidth_gbps *= (1.0 - ev.magnitude * 0.5)
            logger.warning("[FAILURE] SLOW_NODE -%.0f%% perf on %s at t=%.1fs",
                           ev.magnitude * 100, target, sim_time)

        ev.target_node_id = target
        self._history.append({
            "event": ev.label, "type": ev.failure_type.value,
            "node": target, "sim_time": sim_time, "action": "triggered",
        })

    def _recover(self, ev: FailureEvent) -> None:
        if not ev.active or ev.target_node_id is None:
            ev.active = False
            return

        node = self._nodes.get(ev.target_node_id)
        if node and ev._orig_state:
            for attr, val in ev._orig_state.items():
                setattr(node, attr, val)

        if ev.failure_type == FailureType.NODE_LOSS:
            self._dead.discard(ev.target_node_id)

        ev.active = False
        logger.info("[RECOVERY] %s on %s", ev.failure_type.value, ev.target_node_id)
        self._history.append({
            "event": ev.label, "type": ev.failure_type.value,
            "node": ev.target_node_id, "action": "recovered",
        })

    def _pick_target(self, ev: FailureEvent) -> Optional[str]:
        if ev.target_node_id and ev.target_node_id in self._nodes:
            return ev.target_node_id
        alive = self.alive_nodes()
        if not alive:
            return None
        return self._rng.choice(alive).node_id


# ---------------------------------------------------------------------------
# Pre-built failure scenarios
# ---------------------------------------------------------------------------

def make_node_loss_scenario(nodes: list[GPUNode], at_s: float = 30.0) -> FailureInjector:
    """One node dies at t=30s, recovers after 20s."""
    inj = FailureInjector(nodes)
    inj.add(FailureEvent(
        FailureType.NODE_LOSS,
        trigger_time_s=at_s,
        duration_s=20.0,
        auto_recover=True,
        label="node_loss_t30",
    ))
    return inj


def make_memory_spike_scenario(nodes: list[GPUNode]) -> FailureInjector:
    """Random 30% memory spikes with 1% probability per step."""
    inj = FailureInjector(nodes, seed=7)
    inj.add(FailureEvent(
        FailureType.MEMORY_SPIKE,
        trigger_prob=0.01,
        magnitude=0.30,
        duration_s=5.0,
        auto_recover=True,
        label="memory_spike_chaos",
    ))
    return inj


def make_nvlink_degrade_scenario(nodes: list[GPUNode], at_s: float = 45.0) -> FailureInjector:
    """NVLink bandwidth degrades 50% at t=45s (permanent, simulates congestion)."""
    inj = FailureInjector(nodes)
    inj.add(FailureEvent(
        FailureType.NVLINK_DEGRADE,
        trigger_time_s=at_s,
        magnitude=0.50,
        auto_recover=False,
        label="nvlink_degrade_t45",
    ))
    return inj


def make_chaos_scenario(nodes: list[GPUNode]) -> FailureInjector:
    """Combined chaos: node loss + memory spikes + NVLink degradation."""
    inj = FailureInjector(nodes, seed=13)
    inj.add(FailureEvent(FailureType.NODE_LOSS,
                         trigger_time_s=20.0, duration_s=15.0, auto_recover=True))
    inj.add(FailureEvent(FailureType.MEMORY_SPIKE,
                         trigger_prob=0.008, magnitude=0.25, duration_s=8.0))
    inj.add(FailureEvent(FailureType.NVLINK_DEGRADE,
                         trigger_time_s=60.0, magnitude=0.40, duration_s=30.0))
    return inj
