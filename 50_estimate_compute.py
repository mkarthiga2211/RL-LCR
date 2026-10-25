"""Script 50: Estimate Phase 5 compute cost from short timing runs.

Times every method (frozen parameters from FROZEN.json) on ward topologies of several sizes,
using VALIDATION traces so the test split is never touched. One-off ``attach`` cost is timed
separately from the per-simulated-second ``run`` cost; the run cost is extrapolated linearly to
(duration + warm-up) and to the scenario sizes with a per-method power law fitted on node count.
Cohort sizes (100/150/200) lie outside the measured range, so they are extrapolations.

Outputs a console table and ``reports/compute_estimate.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from loguru import logger

from rllcr.utils.hw import perf_cores
from rllcr.utils.logging import setup_logging

METHODS: tuple[str, ...] = ("aodv", "dsr", "olsr", "q_routing", "generic_lcr", "rl_lcr")
LCR_METHODS = ("generic_lcr", "rl_lcr")
SCENARIO_NODES: dict[str, tuple[int, ...]] = {
    "ward": (20, 35, 50),
    "cohort": (100, 150, 200),
    "ambulatory": (20, 35, 50),
}


def time_one(method: str, n_nodes: int, sim_s: float, seed: int, device: str) -> dict[str, Any]:
    """Time one (method, size) run; executed inside a joblib worker."""
    import rllcr.routing  # noqa: F401  (registers protocols)
    from rllcr.eval.frozen import load_frozen_params
    from rllcr.eval.tuning_objective import get_validation_traces
    from rllcr.routing.registry import build
    from rllcr.sim.sdn_controller import SDNController
    from rllcr.sim.simulator import Simulator
    from rllcr.sim.topology import Topology, generate_ward_topology

    params = load_frozen_params()[method]
    extras: dict[str, Any] = {}
    if method in LCR_METHODS:
        extras["device_pref"] = device
    if method in ("olsr", "q_routing"):
        extras["seed"] = seed

    files = sorted(Path("results/topologies").glob(f"ward_{n_nodes}_*.json"))
    topo = Topology.load_json(files[0]) if files else generate_ward_topology(n_nodes=n_nodes, seed=seed)
    traces = get_validation_traces()  # timing only: validation split, never the test split

    proto = build(method, **params, **extras)
    n_route_calls = 0
    if hasattr(proto, "compute_routes"):
        original = proto.compute_routes

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal n_route_calls
            n_route_calls += 1
            return original(*args, **kwargs)

        proto.compute_routes = counted  # type: ignore[method-assign]

    ctrl = (
        SDNController(
            sink_id=topo.sink_id,
            state_report_interval_s=10.0,
            route_recompute_interval_s=30.0,
            measure_compute_time=False,
        )
        if proto.uses_controller
        else None
    )
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}, "protocol": method, "seed": seed},
        topology=topo,
        protocol=proto,
        trace_paths=traces,
        controller=ctrl,
        seed=seed,
        duration_s=sim_s,
        warmup_s=0.0,
    )
    t0 = time.perf_counter()
    (ctrl or proto).attach(sim)
    t_attach = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.run()
    t_run = time.perf_counter() - t0
    m = sim.collector.compute_summary_metrics()
    return {
        "method": method,
        "n": n_nodes,
        "attach_s": t_attach,
        "run_s": t_run,
        "packets": int(m["total_packets"]),
        "route_calls": n_route_calls,
    }


def fit_exponent(sizes: list[int], times: list[float]) -> tuple[float, float]:
    """Fit ``t = a * n^alpha`` by least squares in log space; returns (a, alpha)."""
    alpha, log_a = np.polyfit(np.log(sizes), np.log(np.maximum(times, 1e-9)), 1)
    return float(np.exp(log_a)), float(alpha)


def main() -> None:
    """CLI entrypoint."""
    setup_logging("50_estimate_compute")
    p = argparse.ArgumentParser(description="Estimate Phase 5 compute cost.")
    p.add_argument("--sim-seconds", type=float, default=10.0, help="Simulated seconds per timing run.")
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 35, 50])
    p.add_argument("--seeds", type=int, default=20, help="Seeds per configuration in the plan.")
    p.add_argument("--duration", type=float, default=3600.0)
    p.add_argument("--warmup", type=float, default=120.0)
    p.add_argument("--device", default="mps", help="Device for the Lagrangian methods.")
    p.add_argument("--n-jobs", type=int, default=None, help="Default: perf_cores().")
    p.add_argument("--output", default="reports/compute_estimate.csv")
    args = p.parse_args()

    n_jobs = args.n_jobs or perf_cores()
    tasks = [(m, n) for m in METHODS for n in args.sizes]
    logger.info(f"Timing {len(tasks)} runs of {args.sim_seconds}s each with n_jobs={n_jobs}")
    results = Parallel(n_jobs=n_jobs)(
        delayed(time_one)(m, n, args.sim_seconds, 2026, args.device) for m, n in tasks
    )

    horizon = args.duration + args.warmup
    scale = horizon / args.sim_seconds
    rows: list[dict[str, Any]] = []
    total_cpu_s = 0.0
    per_method_cpu: dict[str, float] = {}

    print(f"\nMeasured ({args.sim_seconds:g} simulated s per run, validation traces):")
    print(f"{'method':12s} {'n':>4s} {'attach_s':>9s} {'run_s':>8s} {'packets':>8s} {'route_calls':>11s}")
    for r in sorted(results, key=lambda r: (METHODS.index(r["method"]), r["n"])):
        print(
            f"{r['method']:12s} {r['n']:4d} {r['attach_s']:9.2f} {r['run_s']:8.2f} "
            f"{r['packets']:8d} {r['route_calls']:11d}"
        )

    print(f"\nExtrapolated to {args.duration:g}s + {args.warmup:g}s warm-up, {args.seeds} seeds:")
    print(f"{'method':12s} {'alpha':>6s} {'s/run@ward35':>13s} {'CPU-h (9 scen x seeds)':>24s}")
    for method in METHODS:
        mine = sorted((r for r in results if r["method"] == method), key=lambda r: r["n"])
        a, alpha = fit_exponent([r["n"] for r in mine], [r["run_s"] for r in mine])
        attach = float(np.mean([r["attach_s"] for r in mine]))
        method_cpu = 0.0
        for scenario, sizes in SCENARIO_NODES.items():
            for n in sizes:
                run_s = a * n**alpha * scale + attach
                cpu_s = run_s * args.seeds
                method_cpu += cpu_s
                rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "n": n,
                        "est_seconds_per_run": round(run_s, 1),
                        "runs": args.seeds,
                        "est_cpu_hours": round(cpu_s / 3600.0, 3),
                        "extrapolated_size": n > max(args.sizes),
                    }
                )
        per_method_cpu[method] = method_cpu
        total_cpu_s += method_cpu
        at35 = a * 35**alpha * scale + attach
        print(f"{method:12s} {alpha:6.2f} {at35:13.0f} {method_cpu / 3600.0:24.1f}")

    total_h = total_cpu_s / 3600.0
    print(f"\nTOTAL CPU-hours : {total_h:,.1f}")
    wall_h = total_h / n_jobs
    print(
        f"Wall-clock @ n_jobs={n_jobs}: {wall_h:,.1f} h ({wall_h / 24:,.1f} days), "
        "assuming perfect scaling"
    )
    print("Note: cohort sizes (100/150/200) are extrapolated beyond the measured range.\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
