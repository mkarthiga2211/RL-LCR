"""Script 20: Generate reproducible network topologies for experimental scenarios.

Generates and saves standard topologies across all scenarios (ward, cohort, ambulatory),
node scales, and replication seeds specified in configs/scenarios.yaml and
configs/experiments.yaml:
    results/topologies/<scenario>_<n>_<seed>.json

Ensures Common Random Numbers (CRN) across comparative routing protocol runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from rllcr.sim.topology import Topology, generate_topology
from rllcr.utils.logging import setup_logging
from rllcr.utils.seed import derive_seed


def compute_topology_metrics(
    topo: Topology,
    power_dbm: int = 0,
    prr_threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate mean nodal degree and average hop distance to coordinator sink.

    Args:
        topo: Evaluated topology instance.
        power_dbm: RF transmission power level (default 0 dBm).
        prr_threshold: Minimum PRR considered a viable link (default 0.5).

    Returns:
        dict[str, float]: Dictionary containing 'mean_degree' and 'mean_hops_to_sink'.
    """
    g = topo.to_networkx(power_dbm=power_dbm, prr_threshold=prr_threshold)
    sink = topo.sink_id

    # 1. Mean degree
    degrees = [g.out_degree(n) for n in g.nodes()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0

    # 2. Mean shortest path hop count to sink
    hop_counts: list[int] = []
    for u in g.nodes():
        if u != sink:
            try:
                hops = nx.shortest_path_length(g, source=u, target=sink)
                hop_counts.append(hops)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

    mean_hops = float(np.mean(hop_counts)) if hop_counts else 0.0
    return {
        "mean_degree": mean_deg,
        "mean_hops_to_sink": mean_hops,
    }


def generate_all_topologies(
    scenarios_yaml: Path | str = "configs/scenarios.yaml",
    experiments_yaml: Path | str = "configs/experiments.yaml",
    output_dir: Path | str = "results/topologies",
) -> list[dict[str, Any]]:
    """Generate and persist all experimental topologies.

    Args:
        scenarios_yaml: Path to scenarios.yaml.
        experiments_yaml: Path to experiments.yaml.
        output_dir: Directory where topology JSON files will be written.

    Returns:
        list[dict[str, Any]]: List of topology metadata records.
    """
    sc_cfg = yaml.safe_load(Path(scenarios_yaml).read_text(encoding="utf-8"))["scenarios"]
    exp_cfg = yaml.safe_load(Path(experiments_yaml).read_text(encoding="utf-8"))["experiments"]

    base_seed = int(exp_cfg.get("base_seed", 2026))
    replications = int(exp_cfg.get("replications", 10))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for scenario_name, params in sc_cfg.items():
        node_counts = params.get("nodes", [20, 35, 50])

        for n in node_counts:
            for rep in range(replications):
                seed = derive_seed(base_seed + rep, scenario_name, n)
                topo_filename = f"{scenario_name}_{n}_{seed}.json"
                target_file = out_path / topo_filename

                topo = generate_topology(
                    scenario=scenario_name,
                    n_nodes=n,
                    seed=seed,
                )
                metrics = compute_topology_metrics(topo, power_dbm=0, prr_threshold=0.5)
                topo.metadata.update(metrics)
                topo.metadata["replication"] = rep

                topo.save_json(target_file)

                records.append(
                    {
                        "scenario": scenario_name,
                        "n_nodes": n,
                        "seed": seed,
                        "file": topo_filename,
                        "mean_degree": metrics["mean_degree"],
                        "mean_hops": metrics["mean_hops_to_sink"],
                    }
                )

    return records


def print_summary_table(records: list[dict[str, Any]]) -> None:
    """Print aggregate mean degree and mean hop count to sink per scenario.

    Args:
        records: List of generated topology records.
    """
    scenarios = sorted(set(r["scenario"] for r in records))

    print("\n" + "=" * 76)
    print("RL-LCR REPRODUCIBLE TOPOLOGIES SUMMARY TABLE (PRR >= 0.5 at 0 dBm)")
    header = (
        f"{'Scenario':<12} | {'Topologies':<10} | {'Node Counts':<14} | "
        f"{'Mean Degree':<12} | {'Mean Hops to Sink':<16}"
    )
    print(header)
    print("-" * 76)

    for sc in scenarios:
        sub = [r for r in records if r["scenario"] == sc]
        n_topos = len(sub)
        n_list = sorted(set(r["n_nodes"] for r in sub))
        n_str = ",".join(str(x) for x in n_list)
        avg_deg = float(np.mean([r["mean_degree"] for r in sub]))
        avg_hops = float(np.mean([r["mean_hops"] for r in sub]))

        print(
            f"{sc.capitalize():<14} | {n_topos:<10} | {n_str:<15} | {avg_deg:<12.2f} | {avg_hops:<16.2f}"
        )

    print("=" * 76 + "\n")


def main() -> None:
    """Entry point for topology generation script."""
    parser = argparse.ArgumentParser(
        description="Generate reproducible topologies for RL-LCR simulation scenarios."
    )
    parser.add_argument(
        "--scenarios-yaml",
        default="configs/scenarios.yaml",
        help="Path to scenarios config.",
    )
    parser.add_argument(
        "--experiments-yaml",
        default="configs/experiments.yaml",
        help="Path to experiments config.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/topologies",
        help="Output destination directory.",
    )
    args = parser.parse_args()

    setup_logging("20_generate_topologies")
    records = generate_all_topologies(
        scenarios_yaml=args.scenarios_yaml,
        experiments_yaml=args.experiments_yaml,
        output_dir=args.output_dir,
    )
    print_summary_table(records)


if __name__ == "__main__":
    main()
