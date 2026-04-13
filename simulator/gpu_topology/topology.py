"""
GPU Cluster Topology.

Models the physical interconnect topology of a GPU cluster:
  - Homogeneous H100/B200 NVLink racks
  - Heterogeneous mixed-GPU deployments
  - PCIe-only nodes (no NVSwitch)

The topology is represented as a graph where nodes are GPUs and edges
carry bandwidth and latency information. The PlacementEngine uses the
topology to estimate communication costs for tensor-parallel workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.models import GPUNode, GPUType, make_gpu_node


class TopologyType(str, Enum):
    """Pre-defined cluster topology presets."""

    SINGLE_NODE = "single_node"             # One multi-GPU server
    NVLINK_RACK = "nvlink_rack"             # 8x H100 NVLink
    HETEROGENEOUS = "heterogeneous"         # Mixed H100 + A100 + A10
    BLACKWELL_RACK = "blackwell_rack"       # 8x B200 NVSwitch
    MULTI_RACK = "multi_rack"               # 2 racks, InfiniBand between racks


@dataclass
class InterconnectLink:
    """Directed link between two GPU nodes."""

    src: str
    dst: str
    bandwidth_gbps: float   # One-way bandwidth
    latency_us: float       # One-way latency in microseconds
    nvlink: bool = False    # True = NVLink/NVSwitch; False = PCIe or IB


@dataclass
class ClusterTopology:
    """
    Represents the network topology of a GPU cluster.

    Nodes are GPUNode objects; edges are InterconnectLinks.
    The topology graph is used to compute communication costs
    for multi-node or tensor-parallel placements.
    """

    nodes: list[GPUNode] = field(default_factory=list)
    links: list[InterconnectLink] = field(default_factory=list)
    name: str = "cluster"

    def add_link(self, src: str, dst: str, bw_gbps: float, lat_us: float, nvlink: bool = False) -> None:
        self.links.append(InterconnectLink(src, dst, bw_gbps, lat_us, nvlink))
        # Add reverse link
        self.links.append(InterconnectLink(dst, src, bw_gbps, lat_us, nvlink))

    def get_link(self, src: str, dst: str) -> Optional[InterconnectLink]:
        for l in self.links:
            if l.src == src and l.dst == dst:
                return l
        return None

    def transfer_cost_ms(self, src: str, dst: str, size_gb: float) -> float:
        """Estimate data transfer time between two nodes in milliseconds."""
        if src == dst:
            return 0.0
        link = self.get_link(src, dst)
        if link is None:
            # Assume 25 Gbps InfiniBand if no direct link
            bw = 25.0
            lat = 5.0
        else:
            bw = link.bandwidth_gbps
            lat = link.latency_us / 1000.0   # µs → ms
        transfer_ms = size_gb * 8.0 / bw * 1000.0   # GB → Gbit / (Gbps) → ms
        return lat + transfer_ms

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def __repr__(self) -> str:
        return f"ClusterTopology({self.name}, {len(self.nodes)} nodes, {len(self.links)//2} links)"


# ---------------------------------------------------------------------------
# Topology factory functions
# ---------------------------------------------------------------------------

def make_single_node(gpu_type: GPUType = GPUType.H100_80GB, num_gpus: int = 8) -> ClusterTopology:
    """8x GPU single server with NVLink."""
    topo = ClusterTopology(name=f"single_node_{gpu_type.value}")
    nodes = [make_gpu_node(f"gpu_{i}", gpu_type, nvlink_enabled=True) for i in range(num_gpus)]
    topo.nodes = nodes
    # Full NVLink mesh: 600 GB/s, 1 µs
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                topo.links.append(InterconnectLink(
                    src=f"gpu_{i}", dst=f"gpu_{j}",
                    bandwidth_gbps=600.0, latency_us=1.0, nvlink=True,
                ))
    return topo


def make_nvlink_rack(num_nodes: int = 8) -> ClusterTopology:
    """Rack of NVLink H100s (DGX H100 style)."""
    return make_single_node(GPUType.H100_NVL, num_gpus=num_nodes)


def make_blackwell_rack(num_nodes: int = 8) -> ClusterTopology:
    """Rack of B200s with NVSwitch (GB200 NVL72 style)."""
    topo = ClusterTopology(name="blackwell_nvl_rack")
    nodes = [make_gpu_node(f"b200_{i}", GPUType.B200_192GB, nvlink_enabled=True) for i in range(num_nodes)]
    topo.nodes = nodes
    # NVSwitch: 1800 GB/s bisection, 0.5 µs latency
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                topo.links.append(InterconnectLink(
                    src=f"b200_{i}", dst=f"b200_{j}",
                    bandwidth_gbps=1800.0, latency_us=0.5, nvlink=True,
                ))
    return topo


def make_heterogeneous_cluster(
    n_h100: int = 4,
    n_a100: int = 4,
    n_a10: int = 4,
) -> ClusterTopology:
    """Mixed-GPU cluster: H100s (NVLink), A100s (NVLink), A10s (PCIe)."""
    topo = ClusterTopology(name="heterogeneous")
    nodes: list[GPUNode] = []

    for i in range(n_h100):
        nodes.append(make_gpu_node(f"h100_{i}", GPUType.H100_80GB, nvlink_enabled=True))
    for i in range(n_a100):
        nodes.append(make_gpu_node(f"a100_{i}", GPUType.A100_80GB, nvlink_enabled=True))
    for i in range(n_a10):
        nodes.append(make_gpu_node(f"a10_{i}", GPUType.A10_24GB, nvlink_enabled=False))

    topo.nodes = nodes

    # H100–H100 NVLink
    for i in range(n_h100):
        for j in range(n_h100):
            if i != j:
                topo.links.append(InterconnectLink(
                    src=f"h100_{i}", dst=f"h100_{j}",
                    bandwidth_gbps=600.0, latency_us=1.0, nvlink=True,
                ))
    # A100–A100 NVLink
    for i in range(n_a100):
        for j in range(n_a100):
            if i != j:
                topo.links.append(InterconnectLink(
                    src=f"a100_{i}", dst=f"a100_{j}",
                    bandwidth_gbps=400.0, latency_us=1.5, nvlink=True,
                ))
    # Cross-type: InfiniBand (25 Gbps, 5 µs)
    all_ids = [f"h100_{i}" for i in range(n_h100)] + [f"a100_{i}" for i in range(n_a100)]
    for src in all_ids:
        for dst in all_ids:
            if src != dst and not topo.get_link(src, dst):
                topo.links.append(InterconnectLink(
                    src=src, dst=dst,
                    bandwidth_gbps=25.0, latency_us=5.0, nvlink=False,
                ))

    return topo


def make_multi_rack(nodes_per_rack: int = 8) -> ClusterTopology:
    """Two NVLink racks connected by InfiniBand."""
    topo = ClusterTopology(name="multi_rack")
    nodes: list[GPUNode] = []

    for r in range(2):
        for i in range(nodes_per_rack):
            nid = f"rack{r}_h100_{i}"
            nodes.append(make_gpu_node(nid, GPUType.H100_NVL, nvlink_enabled=True))

    topo.nodes = nodes

    # Intra-rack NVLink
    for r in range(2):
        rack_ids = [f"rack{r}_h100_{i}" for i in range(nodes_per_rack)]
        for src in rack_ids:
            for dst in rack_ids:
                if src != dst:
                    topo.links.append(InterconnectLink(
                        src=src, dst=dst,
                        bandwidth_gbps=900.0, latency_us=1.0, nvlink=True,
                    ))

    # Inter-rack InfiniBand (NDR 400G)
    for i in range(nodes_per_rack):
        for j in range(nodes_per_rack):
            topo.links.append(InterconnectLink(
                src=f"rack0_h100_{i}", dst=f"rack1_h100_{j}",
                bandwidth_gbps=50.0, latency_us=10.0, nvlink=False,
            ))
            topo.links.append(InterconnectLink(
                src=f"rack1_h100_{j}", dst=f"rack0_h100_{i}",
                bandwidth_gbps=50.0, latency_us=10.0, nvlink=False,
            ))

    return topo


TOPOLOGY_FACTORY = {
    TopologyType.SINGLE_NODE:    lambda: make_single_node(),
    TopologyType.NVLINK_RACK:    lambda: make_nvlink_rack(),
    TopologyType.HETEROGENEOUS:  lambda: make_heterogeneous_cluster(),
    TopologyType.BLACKWELL_RACK: lambda: make_blackwell_rack(),
    TopologyType.MULTI_RACK:     lambda: make_multi_rack(),
}


def make_topology(ttype: TopologyType) -> ClusterTopology:
    return TOPOLOGY_FACTORY[ttype]()
