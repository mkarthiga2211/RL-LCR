"""Script 51: Size a Phase 5 experiment design from measured per-run costs.

Reuses ``reports/compute_estimate.csv`` (script 50: per-run seconds for a 3600 s + 120 s run, per
method / scenario / n) and rescales linearly to the design's duration. Runs that a later experiment
can reuse from an earlier one are not counted twice. ``lcr_class_delay`` was not timed; it is costed
as ``generic_lcr`` (same pipeline, only the constraint mapper differs) - an assumption, flagged below.

Outputs a console table and ``reports/compute_estimate_reduced.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from rllcr.eval.design import DURATION_S, EXPERIMENTS, SEEDS, WARMUP_S
from rllcr.utils.hw import perf_cores

BASE_HORIZON_S = 3720.0  # horizon script 50 extrapolated to (3600 s + 120 s warm-up)
COST_PROXY = {"lcr_class_delay": "generic_lcr"}


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Size a Phase 5 design from measured costs.")
    p.add_argument("--csv", default="reports/compute_estimate.csv")
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--duration", type=float, default=DURATION_S)
    p.add_argument("--warmup", type=float, default=WARMUP_S)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--output", default="reports/compute_estimate_reduced.csv")
    args = p.parse_args()

    n_jobs = args.n_jobs or perf_cores()
    scale = (args.duration + args.warmup) / BASE_HORIZON_S
    cost = pd.read_csv(args.csv).set_index(["method", "scenario", "n"])["est_seconds_per_run"]

    rows: list[dict[str, Any]] = []
    for experiment, exp in EXPERIMENTS.items():
        methods, cells, datasets = exp.methods, exp.cells, exp.datasets
        for method in methods:
            for scenario, n in cells:
                per_run = float(cost[(COST_PROXY.get(method, method), scenario, n)]) * scale
                runs = args.seeds * len(datasets)
                rows.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "scenario": scenario,
                        "n": n,
                        "runs": runs,
                        "hours_per_run": round(per_run / 3600.0, 3),
                        "cpu_hours": round(per_run * runs / 3600.0, 2),
                    }
                )
    df = pd.DataFrame(rows)

    print(
        f"\nDesign: {args.seeds} seeds, {args.duration:g}s + {args.warmup:g}s warm-up "
        f"(x{scale:.3f} of the 3720 s estimate)\n"
    )
    by_exp = df.groupby("experiment")[["runs", "cpu_hours"]].sum()
    print(by_exp.to_string())
    print("\nBy method (CPU-hours):")
    print(df.groupby("method")["cpu_hours"].sum().round(1).to_string())
    total = float(df["cpu_hours"].sum())
    print(f"\nTOTAL: {int(df['runs'].sum())} runs, {total:,.0f} CPU-hours")
    print(f"Wall-clock @ n_jobs={n_jobs}: {total / n_jobs:,.1f} h ({total / n_jobs / 24:,.1f} days)")
    print("Assumes lcr_class_delay costs the same as generic_lcr (not timed).\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
