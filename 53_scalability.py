"""Script 53: Phase 5.5 controller scalability benchmark.

Measures the controller route-computation time per call across network scales:
n = 20, 35, 50, 100, 150, 200 nodes (single call per method and size, no full simulation).
Saves results to results/metrics/scalability/summary.parquet and reports/tables/scalability.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from rllcr.eval.frozen import load_frozen_params
from rllcr.routing.registry import build
from rllcr.sim.sdn_controller import SDNController
from rllcr.sim.simulator import Simulator
from rllcr.sim.topology import Topology, generate_cohort_topology, generate_ward_topology
from rllcr.utils.logging import setup_logging

SIZES = (20, 35, 50, 100, 150, 200)
METHODS = ("rl_lcr", "generic_lcr", "lcr_class_delay")
LCR_METHODS = ("rl_lcr", "generic_lcr", "lcr_class_delay")
SEED = 2026


def get_topology_for_size(n: int, seed: int = SEED) -> Topology:
    """Return an appropriate topology for the given node count."""
    if n <= 50:
        files = sorted(Path("results/topologies").glob(f"ward_{n}_*.json"))
        if files:
            return Topology.load_json(files[0])
        return generate_ward_topology(n_nodes=n, seed=seed)
    else:
        return generate_cohort_topology(n_nodes=n, seed=seed)


def benchmark_controller(
    method: str,
    n: int,
    device: str = "mps",
    repeats: int = 3,
) -> dict[str, Any]:
    """Benchmark single-call compute_routes time for a method and node count."""
    params = load_frozen_params().get(method, {})
    topo = get_topology_for_size(n)
    scenario = "ward" if n <= 50 else "cohort"

    extras: dict[str, Any] = {}
    if method in LCR_METHODS:
        extras["device_pref"] = device

    proto = build(method, **params, **extras)
    ctrl = SDNController(
        sink_id=topo.sink_id,
        state_report_interval_s=10.0,
        route_recompute_interval_s=30.0,
        measure_compute_time=True,
    )
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}, "protocol": method, "seed": SEED},
        topology=topo,
        protocol=proto,
        trace_paths=[],
        controller=ctrl,
        seed=SEED,
        duration_s=1.0,
        warmup_s=0.0,
    )
    if ctrl is not None:
        ctrl.attach(sim)
    proto.attach(sim)

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proto.compute_routes(view=ctrl.global_view)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(t_elapsed_ms)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    median_ms = float(np.median(times_ms))

    logger.info(f"[{method}] n={n} ({scenario}): mean={mean_ms:.2f} ms (+/- {std_ms:.2f})")
    return {
        "method": method,
        "n": n,
        "scenario": scenario,
        "device": device,
        "mean_compute_ms": mean_ms,
        "std_compute_ms": std_ms,
        "median_compute_ms": median_ms,
        "repeats": repeats,
    }


def main() -> int:
    """CLI entrypoint."""
    setup_logging("53_scalability")
    parser = argparse.ArgumentParser(description="Phase 5.5 controller scalability benchmark.")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="results/metrics/scalability")
    parser.add_argument("--table-dir", default="reports/tables")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        for method in args.methods:
            try:
                row = benchmark_controller(method, n, device=args.device, repeats=args.repeats)
                rows.append(row)
            except Exception as e:
                logger.error(f"Failed benchmark {method} n={n}: {e}")

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "summary.parquet", index=False)
    df.to_csv(out_dir / "summary.csv", index=False)

    tbl_dir = Path(args.table_dir)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tbl_dir / "scalability.csv", index=False)
    logger.info(f"Wrote scalability results to {out_dir} and {tbl_dir / 'scalability.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
