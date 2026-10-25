"""Script 21: Discrete-event SD-WSN simulator smoke test.

Executes a 120-second simulation of the 20-node ward scenario using the
ShortestPathOracle test routing protocol (hop-count Dijkstra) and prints
packet delivery ratio (PDR), latency, and energy statistics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from rllcr.routing.base import RoutingProtocol
from rllcr.sim.packet import Packet
from rllcr.sim.simulator import Simulator
from rllcr.sim.topology import Topology, generate_ward_topology
from rllcr.utils.logging import setup_logging

if TYPE_CHECKING:
    from rllcr.sim.simulator import Simulator as SimInstance


class ShortestPathOracle(RoutingProtocol):
    """Trivial oracle routing protocol forwarding along the shortest hop path to sink."""

    name: str = "shortest_path_oracle"
    uses_controller: bool = False

    def __init__(self) -> None:
        """Initialize oracle router."""
        self.sim: SimInstance | None = None
        self.next_hops: dict[int, int] = {}

    def attach(self, sim: SimInstance) -> None:
        """Calculate static shortest paths to sink on graph topology."""
        self.sim = sim
        g = sim.topology.to_networkx(power_dbm=0, prr_threshold=0.5)
        sink = sim.sink_id

        self.next_hops.clear()
        for u in g.nodes():
            if u != sink:
                try:
                    path = nx.shortest_path(g, source=u, target=sink)
                    if len(path) >= 2:
                        self.next_hops[u] = path[1]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

    def on_packet_generated(self, node_id: int, pkt: Packet) -> None:
        """No-op on generation for oracle."""
        pass

    def next_hop(self, node_id: int, pkt: Packet) -> int | None:
        """Look up static shortest hop towards sink."""
        return self.next_hops.get(node_id)

    def on_link_failure(self, node_id: int, nbr: int, pkt: Packet) -> None:
        """No-op on failure for static oracle."""
        pass

    def on_control_timer(self, now: float) -> None:
        """No periodic control messages for oracle."""
        pass

    def control_packets(self) -> list[Packet]:
        """Return zero control packets."""
        return []


def run_smoke_test(
    duration_s: float = 120.0,
    n_nodes: int = 20,
    seed: int = 2026,
    output_dir: str = "results/runs/smoke/shortest_path_oracle",
) -> dict[str, Any]:
    """Execute smoke test simulation and report metrics."""
    t_start = time.perf_counter()

    # 1. Load or generate topology
    topo_files = list(Path("results/topologies").glob(f"ward_{n_nodes}_*.json"))
    if topo_files:
        topo = Topology.load_json(topo_files[0])
    else:
        topo = generate_ward_topology(n_nodes=n_nodes, seed=seed)

    # 2. Collect sample clinical trace files
    trace_files = sorted(Path("data/processed/traces/icu_full").glob("*.parquet"))
    if not trace_files:
        raise FileNotFoundError("No trace files found in data/processed/traces/icu_full")

    # 3. Instantiate protocol and simulator
    protocol = ShortestPathOracle()
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}},
        topology=topo,
        protocol=protocol,
        trace_paths=trace_files,
        seed=seed,
        duration_s=duration_s,
        warmup_s=10.0,
    )

    # 4. Run simulation
    metrics = sim.run()
    elapsed_wall = time.perf_counter() - t_start

    # 5. Persist simulation tables
    saved = sim.save_results(
        exp_name="smoke",
        method_name="shortest_path_oracle",
        results_root=Path(output_dir).parent.parent,
    )

    # 6. Calculate total energy
    total_energy_j = sum(node.energy.total_consumed() for node in sim.nodes.values())

    print("\n" + "=" * 60)
    print("SIMULATOR SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"Topology:           Ward {n_nodes}-node (Scenario: {topo.scenario})")
    print(f"Simulation Clock:   {duration_s:.1f} s")
    print(f"Wall Execution:     {elapsed_wall:.2f} s")
    print(f"Packets Generated:  {metrics.get('total_packets', 0)}")
    print(f"Packets Delivered:  {metrics.get('delivered', 0)}")
    print(f"Packet Delivery:    {metrics.get('pdr', 0.0) * 100:.2f}% (PDR)")
    print(f"Mean Latency:       {metrics.get('mean_delay_ms', 0.0):.2f} ms")
    print(f"Deadline Violation: {metrics.get('dvr', 0.0) * 100:.2f}% (DVR)")
    print(f"Total Net Energy:   {total_energy_j:.2f} Joules")
    print(f"Results Saved:      {list(saved.values())[0].parent}")
    print("=" * 60 + "\n")

    return {
        "metrics": metrics,
        "elapsed_wall_s": elapsed_wall,
        "total_energy_j": total_energy_j,
    }


def main() -> None:
    """CLI entry point for simulator smoke test."""
    parser = argparse.ArgumentParser(description="SimPy SD-WSN simulator smoke test.")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration (s).")
    parser.add_argument("--nodes", type=int, default=20, help="Node count.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    setup_logging("21_sim_smoke_test")
    run_smoke_test(duration_s=args.duration, n_nodes=args.nodes, seed=args.seed)


if __name__ == "__main__":
    main()
