"""Script 32: Demonstration and evaluation of proposed RL-LCR routing protocol.

Executes a trace-driven simulation of the 20-node Ward scenario using the proposed
RL-LCR routing algorithm inside the centralized SDN controller.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from rllcr.routing.rl_lcr import RLLCR
from rllcr.sim.sdn_controller import SDNController
from rllcr.sim.simulator import Simulator
from rllcr.sim.topology import Topology, generate_ward_topology
from rllcr.utils.logging import setup_logging


def run_demo(
    duration_s: float = 120.0,
    n_nodes: int = 20,
    seed: int = 2026,
    iters: int = 50,
    device_pref: str = "auto",
    output_dir: str = "results/runs/demo/rl_lcr/ward_20_2026",
) -> dict[str, Any]:
    """Execute RL-LCR simulation run and print telemetry metrics."""
    t_wall_start = time.perf_counter()

    # 1. Load or generate physical topology
    topo_files = list(Path("results/topologies").glob(f"ward_{n_nodes}_*.json"))
    if topo_files:
        topo = Topology.load_json(topo_files[0])
    else:
        topo = generate_ward_topology(n_nodes=n_nodes, seed=seed)

    # 2. Gather clinical trace files
    trace_files = sorted(Path("data/processed/traces/icu_full").glob("*.parquet"))
    if not trace_files:
        raise FileNotFoundError("No clinical trace files found in data/processed/traces/icu_full")

    # 3. Instantiate RL-LCR protocol and SDN controller
    protocol = RLLCR(
        iters=iters,
        device_pref=device_pref,
    )
    controller = SDNController(
        sink_id=topo.sink_id,
        state_report_interval_s=10.0,
        route_recompute_interval_s=30.0,
        measure_compute_time=True,
    )

    # 4. Instantiate Simulator
    sim = Simulator(
        cfg={
            "radio": {"bitrate_bps": 250000},
            "protocol": "rl_lcr",
            "seed": seed,
        },
        topology=topo,
        protocol=protocol,
        trace_paths=trace_files,
        controller=controller,
        seed=seed,
        duration_s=duration_s,
        warmup_s=0.0,
    )

    controller.attach(sim)

    # 5. Run simulation
    sim.run()

    t_wall_elapsed = time.perf_counter() - t_wall_start

    # 6. Save Parquet telemetry artifacts
    out_path = Path(output_dir)
    saved_files = sim.collector.save(out_path)

    # 7. Compute summary statistics
    metrics = sim.collector.compute_summary_metrics()
    total_energy_j = sum(node.energy.total_consumed() for node in sim.nodes.values())

    mean_comp_time = (
        float(sum(controller.computation_times_s) / len(controller.computation_times_s))
        if controller.computation_times_s
        else 0.0
    )

    # Print summary output
    print("\n" + "=" * 60)
    print("RL-LCR ROUTING PROTOCOL DEMO COMPLETE")
    print("=" * 60)
    print(f"Topology:           Ward {n_nodes}-node (Scenario: {topo.scenario})")
    print(f"Simulation Clock:   {duration_s:.1f} s")
    print(f"Wall Execution:     {t_wall_elapsed:.2f} s")
    print(f"Device Preferred:   {device_pref}")
    print(f"Packets Generated:  {metrics['total_packets']}")
    print(f"Packets Delivered:  {metrics['delivered']}")
    print(f"Packet Delivery:    {metrics['pdr'] * 100:.2f}% (PDR)")
    print(f"Mean Latency:       {metrics['mean_delay_ms']:.2f} ms")
    print(f"Deadline Violation: {metrics['dvr'] * 100:.2f}% (DVR)")
    print(f"Total Net Energy:   {total_energy_j:.2f} Joules")
    n_cycles = len(controller.computation_times_s)
    print(f"Controller Recomp:  {n_cycles} cycles (Mean: {mean_comp_time * 1000.0:.1f} ms)")
    print(f"Results Saved:      {out_path}")
    print("=" * 60 + "\n")

    return {
        "metrics": metrics,
        "saved_files": saved_files,
        "wall_time_s": t_wall_elapsed,
        "controller_cycles": len(controller.computation_times_s),
    }


def main() -> None:
    """CLI entrypoint."""
    setup_logging("32_demo_rllcr")
    parser = argparse.ArgumentParser(description="Demo proposed RL-LCR routing protocol.")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration (s).")
    parser.add_argument("--nodes", type=int, default=20, help="Number of network nodes.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--iters", type=int, default=50, help="Lagrangian PGD iterations.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cpu').")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/runs/demo/rl_lcr/ward_20_2026",
        help="Destination directory for Parquet artifacts.",
    )
    args = parser.parse_args()

    run_demo(
        duration_s=args.duration,
        n_nodes=args.nodes,
        seed=args.seed,
        iters=args.iters,
        device_pref=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
