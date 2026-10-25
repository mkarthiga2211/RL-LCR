"""Script 54: Paired significance tests and BCa confidence intervals, RL-LCR vs each baseline.

Reads the per-run summaries written by the Phase 5 experiment scripts
(``results/metrics/{primary,generalization,ablation,lifetime}/summary.parquet``; one row per
method x scenario x n x dataset x seed) and writes ``results/stats/paired_tests.{parquet,csv}``.

Pairing unit is the seed within (experiment, scenario, n, dataset). See ``rllcr.eval.stats``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from rllcr.eval.stats import (
    ALPHA,
    N_RESAMPLES,
    SEED,
    compare_summary,
    format_console_summary,
)
from rllcr.utils.logging import setup_logging

EXPERIMENTS: tuple[str, ...] = ("primary", "generalization", "ablation", "lifetime")
DEFAULT_BASELINES: tuple[str, ...] = ("aodv", "dsr", "olsr", "q_routing", "generic_lcr")


def main() -> int:
    """CLI entrypoint. Returns a process exit code."""
    setup_logging("54_statistics")
    parser = argparse.ArgumentParser(description="Paired tests: RL-LCR vs each baseline.")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--output-dir", default="results/stats")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    parser.add_argument("--reference", default="rl_lcr")
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_parquet, out_csv = out_dir / "paired_tests.parquet", out_dir / "paired_tests.csv"
    if (out_parquet.exists() or out_csv.exists()) and not args.overwrite:
        logger.error(f"{out_csv} exists; pass --overwrite to regenerate.")
        return 1

    frames: list[pd.DataFrame] = []
    for experiment in args.experiments:
        path = Path(args.metrics_dir) / experiment / "summary.parquet"
        if not path.exists():
            logger.warning(f"[{experiment}] {path} not found; skipping.")
            continue
        summary = pd.read_parquet(path)
        # Ablation compares against every ladder variant present; others use the fixed baselines.
        baselines = None if experiment == "ablation" else args.baselines
        result = compare_summary(
            summary,
            reference=args.reference,
            baselines=baselines,
            experiment=experiment,
            n_resamples=args.n_resamples,
            seed=args.seed,
            alpha=args.alpha,
        )
        logger.info(f"[{experiment}] {len(result)} comparisons from {len(summary)} runs")
        frames.append(result)

    if not frames:
        logger.error(f"No summary.parquet found under {args.metrics_dir}; run the experiments first.")
        return 1

    results = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_parquet, index=False)
    results.to_csv(out_csv, index=False)
    print(format_console_summary(results))
    logger.info(f"Wrote {out_csv} and {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
