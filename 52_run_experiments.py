"""Script 52: Phase 5 experiment runner (primary, zero-shot, ablation) with common random numbers.

For every (method, scenario, n, dataset, seed) job the runner simulates the run, saves the
per-packet / energy / control / route logs, computes all Step 5.1-5.3 metrics FROM THOSE SAVED LOGS
and writes ``metrics.json``. It then aggregates ``results/metrics/<experiment>/summary.parquet``
for ``scripts/54_statistics.py``.

Integrity rules enforced here
    * refuses to start unless ``reports/verification.md`` says PASS (test/zero-shot runs);
    * method parameters come ONLY from ``results/tuning/FROZEN.json`` (sha256-verified);
    * seed ``s`` fixes topology (the s-th topology file), traffic windows and all simulator randomness
      through ``derive_seed(2026, scenario, n, dataset, s)`` - identical for every method (CRN);
    * ``--split val`` runs a pilot on the VALIDATION split into ``results/runs/pilot`` and never writes
      summaries, so the runner can be exercised without touching the test split.

Idempotent: a run whose ``metrics.json`` exists is skipped unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from rllcr.eval.design import (
    ABLATION_REUSED_METHODS,
    BASE_SEED,
    DURATION_S,
    EXPERIMENTS,
    MIMIC_TEST,
    SEEDS,
    WARMUP_S,
)
from rllcr.eval.frozen import FROZEN_PATH, load_frozen_params, sha256_file
from rllcr.eval.metrics_generalization import summarize_drop, zero_shot_drop
from rllcr.eval.summary import build_summary, compute_run_metrics
from rllcr.utils.hw import perf_cores
from rllcr.utils.logging import setup_logging
from rllcr.utils.seed import derive_seed

RUNS_ROOT = Path("results/runs")
METRICS_ROOT = Path("results/metrics")
VERIFICATION = Path("reports/verification.md")
LCR_METHODS = ("generic_lcr", "lcr_class_delay", "rl_lcr")
TRACE_DIRS = {
    "ward": Path("data/processed/traces/icu_full"),
    "ambulatory": Path("data/processed/traces/ambulatory_subsampled"),
}
SECONDARY_DIR = Path("data/processed/secondary")


# --------------------------------------------------------------------------- gates and inputs
def verification_passed(path: Path = VERIFICATION) -> bool:
    """True iff the verification report exists, says PASS overall and has no FAIL check row."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    overall = re.search(r"Overall Verification Status\*\*:\s*\*\*(\w+)\*\*", text)
    has_fail_row = re.search(r"^\|.*\*\*FAIL\*\*", text, flags=re.MULTILINE) is not None
    return bool(overall and overall.group(1) == "PASS" and not has_fail_row)


def mimic_traces(scenario: str, split: str) -> list[Path]:
    """MIMIC trace files for a split, de-duplicated.

    ``load_traces`` enforces split hygiene (evaluation may only read test, tuning only train/val).
    It appends each file once per forbidden patient, so duplicates are removed here.
    """
    from rllcr.data.splits import load_traces

    purpose = "evaluation" if split == "test" else "tuning"
    paths = load_traces(split=split, purpose=purpose, scenario=scenario, data_dir=TRACE_DIRS[scenario])
    return sorted(set(paths))


def zero_shot_traces(dataset: str, window_s: float, out_root: Path, overwrite: bool) -> list[Path]:
    """Zero-shot trace files for a dataset; short recordings are chained into composites.

    BIDMC recordings are ~480 s, shorter than a run. For any dataset whose median recording is
    shorter than ``window_s``, deterministic composite traces are built by concatenating different
    subjects' recordings back to back (real data only, no looping).
    """
    import numpy as np

    files = sorted((SECONDARY_DIR / dataset / "traces").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No zero-shot traces for '{dataset}' under {SECONDARY_DIR}")
    lengths = {f: float(pd.read_parquet(f, columns=["t_gen_s"])["t_gen_s"].max()) for f in files}
    if float(np.median(list(lengths.values()))) >= window_s:
        return files

    out_dir = out_root / "_composite" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    composites: list[Path] = []
    for j in range(len(files)):
        target = out_dir / f"composite_{j:02d}.parquet"
        if overwrite or not target.exists():
            rng = np.random.default_rng(derive_seed(BASE_SEED, "composite", dataset, j))
            order = rng.permutation(len(files))
            parts, elapsed = [], 0.0
            for idx in order:
                df = pd.read_parquet(files[idx]).sort_values("t_gen_s")
                df["t_gen_s"] = df["t_gen_s"] + elapsed
                parts.append(df)
                elapsed += lengths[files[idx]] + 1.0
                if elapsed >= window_s + 1.0:
                    break
            pd.concat(parts, ignore_index=True).to_parquet(target, index=False)
        composites.append(target)
    return composites


def topology_files(scenario: str, n: int, seeds: int) -> list[Path]:
    """The first ``seeds`` topology files of a configuration (seed s uses the s-th file)."""
    files = sorted(Path("results/topologies").glob(f"{scenario}_{n}_*.json"))
    if len(files) < seeds:
        raise FileNotFoundError(
            f"Need {seeds} topologies for {scenario}_{n}, found {len(files)}; "
            "run scripts/20_generate_topologies.py."
        )
    return files[:seeds]


# --------------------------------------------------------------------------- one run
def run_dir_for(root: Path, method: str, scenario: str, n: int, dataset: str, seed: int) -> Path:
    """Directory holding one run's logs and metrics."""
    return root / method / f"{scenario}_{n}_{dataset}_s{seed:02d}"


def run_one(job: dict[str, Any]) -> dict[str, Any] | None:
    """Simulate one job, save its logs, compute metrics from the saved logs. Never raises."""
    method, scenario, n = job["method"], job["scenario"], job["n"]
    dataset, seed = job["dataset"], job["seed"]
    run_dir = Path(job["run_dir"])
    try:
        import rllcr.routing  # noqa: F401  (registers protocols)
        from rllcr.routing.registry import build
        from rllcr.sim.sdn_controller import SDNController
        from rllcr.sim.simulator import Simulator
        from rllcr.sim.topology import Topology

        t_wall = time.perf_counter()
        topo = Topology.load_json(job["topology"])
        sim_seed = derive_seed(BASE_SEED, scenario, n, dataset, seed)  # same for every method (CRN)

        extras: dict[str, Any] = {}
        if method in LCR_METHODS:
            extras["device_pref"] = job["device"]
        if method in ("olsr", "q_routing"):
            extras["seed"] = derive_seed(BASE_SEED, "protocol", method, scenario, n, dataset, seed)
        proto = build(method, **job["params"], **extras)

        controller = (
            SDNController(
                sink_id=topo.sink_id,
                state_report_interval_s=10.0,
                route_recompute_interval_s=30.0,
                measure_compute_time=False,
            )
            if proto.uses_controller
            else None
        )
        warmup, duration = job["warmup_s"], job["duration_s"]
        sim = Simulator(
            cfg={
                "radio": {"bitrate_bps": 250000},
                "protocol": method,
                "seed": sim_seed,
                "trace_window_s": warmup + duration,
            },
            topology=topo,
            protocol=proto,
            trace_paths=[Path(p) for p in job["traces"]],
            controller=controller,
            seed=sim_seed,
            duration_s=warmup + duration,
            warmup_s=warmup,
        )
        sim.run()
        sim_wall = time.perf_counter() - t_wall

        # Persist logs, then compute every metric from the saved files only.
        sim.collector.save(run_dir)
        nodes = pd.DataFrame(
            [
                {
                    "node_id": nid,
                    "node_type": node.node_type,
                    "initial_j": node.energy.initial_j,
                    "mains_fraction": node.energy.mains_fraction,
                }
                for nid, node in sim.nodes.items()
            ]
        )
        nodes.to_parquet(run_dir / "nodes.parquet", index=False)
        (run_dir / "run_info.json").write_text(
            json.dumps({"warmup_s": warmup, "end_s": warmup + duration, "sim_seed": sim_seed}),
            encoding="utf-8",
        )
        row = {
            "method": method,
            "scenario": scenario,
            "n": n,
            "dataset": dataset,
            "seed": seed,
            "experiment": job["experiment"],
            "split": job["split"],
            "sim_wall_s": sim_wall,
            "frozen_sha256": job["frozen_sha256"],
            **compute_run_metrics(run_dir),
        }
        (run_dir / "metrics.json").write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
        return row
    except Exception:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return None


# --------------------------------------------------------------------------- summaries
def write_summaries(experiment: str) -> None:
    """Aggregate run metrics into ``results/metrics/<dir>/summary.{parquet,csv}`` (+ derived tables)."""
    exp = EXPERIMENTS[experiment]
    out_dir = METRICS_ROOT / exp.summary_dir

    if experiment == "ablation":
        primary = build_summary([RUNS_ROOT / "primary"])
        reused = primary[
            primary["method"].isin(ABLATION_REUSED_METHODS)
            & primary.set_index(["scenario", "n"]).index.isin(list(exp.cells))
        ]
        new = build_summary([RUNS_ROOT / "ablation"])
        summary = pd.concat([reused, new], ignore_index=True)
    else:
        summary = build_summary([RUNS_ROOT / experiment])
    if summary.empty:
        logger.warning(f"[{experiment}] no completed runs; nothing to summarise.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"[{experiment}] wrote {len(summary)} rows -> {out_dir / 'summary.parquet'}")

    if experiment == "primary":
        lifetime_cols = ["method", "scenario", "n", "dataset", "seed", "fnd_h", "hnd_h", "battery_life_h"]
        life_dir = METRICS_ROOT / "lifetime"
        life_dir.mkdir(parents=True, exist_ok=True)
        summary[lifetime_cols].to_parquet(life_dir / "summary.parquet", index=False)
        summary[lifetime_cols].to_csv(life_dir / "summary.csv", index=False)
    if experiment == "zeroshot":
        primary = build_summary([RUNS_ROOT / "primary"])
        if not primary.empty:
            drops = zero_shot_drop(primary, summary)
            drops.to_csv(out_dir / "zero_shot_drop_per_seed.csv", index=False)
            summarize_drop(drops).to_csv(out_dir / "zero_shot_drop.csv", index=False)


# --------------------------------------------------------------------------- main
def parse_cells(values: list[str] | None) -> set[tuple[str, int]] | None:
    """Parse ``ward:20`` style filters."""
    if not values:
        return None
    return {(v.split(":")[0], int(v.split(":")[1])) for v in values}


def main() -> int:
    """CLI entrypoint. Returns a process exit code."""
    setup_logging("52_run_experiments")
    p = argparse.ArgumentParser(description="Run Phase 5 experiments.")
    p.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    p.add_argument("--methods", nargs="+", help="Subset of the experiment's methods.")
    p.add_argument("--cells", nargs="+", help="Subset of cells, e.g. ward:20 ambulatory:35.")
    p.add_argument("--datasets", nargs="+", help="Subset of the experiment's datasets.")
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--duration", type=float, default=DURATION_S)
    p.add_argument("--warmup", type=float, default=WARMUP_S)
    p.add_argument("--split", choices=("test", "val"), default="test", help="val = pilot, no summaries.")
    p.add_argument("--device", default="mps", help="Device for the Lagrangian methods.")
    p.add_argument("--n-jobs", type=int, default=None, help="Default: perf_cores().")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N pending jobs.")
    p.add_argument("--dry-run", action="store_true", help="List pending jobs and exit.")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    exp = EXPERIMENTS[args.experiment]
    pilot = args.split == "val"
    if pilot and args.experiment == "zeroshot":
        logger.error("--split val is only meaningful for MIMIC experiments (primary/ablation).")
        return 2
    if not pilot and not verification_passed():
        logger.error(f"{VERIFICATION} is missing or not all-PASS; refusing to run test-split experiments.")
        return 1
    if not FROZEN_PATH.exists():
        logger.error(f"{FROZEN_PATH} not found; tune and freeze parameters first.")
        return 1

    exp_name = "pilot" if pilot else args.experiment
    root = RUNS_ROOT / exp_name
    if args.summarize_only:
        if pilot:
            logger.error("Pilot runs are never summarised.")
            return 2
        write_summaries(args.experiment)
        return 0

    frozen = load_frozen_params()  # verifies sha256 of every tuning file
    frozen_sha = sha256_file(FROZEN_PATH)
    methods = [m for m in exp.methods if not args.methods or m in args.methods]
    cell_filter = parse_cells(args.cells)
    cells = [c for c in exp.cells if cell_filter is None or c in cell_filter]
    datasets = [d for d in exp.datasets if not args.datasets or d in args.datasets]
    dataset_label = {MIMIC_TEST: "mimic_val_pilot"} if pilot else {}

    window_s = args.warmup + args.duration
    jobs: list[dict[str, Any]] = []
    for scenario, n in cells:
        topos = topology_files(scenario, n, args.seeds)
        for dataset in datasets:
            traces = (
                mimic_traces(scenario, args.split)
                if dataset == MIMIC_TEST
                else zero_shot_traces(dataset, window_s, root, args.overwrite)
            )
            label = dataset_label.get(dataset, dataset)
            for method in methods:
                for seed in range(args.seeds):
                    run_dir = run_dir_for(root, method, scenario, n, label, seed)
                    if (run_dir / "metrics.json").exists() and not args.overwrite:
                        continue
                    jobs.append(
                        {
                            "experiment": exp_name,
                            "split": args.split,
                            "method": method,
                            "scenario": scenario,
                            "n": n,
                            "dataset": label,
                            "seed": seed,
                            "topology": str(topos[seed]),
                            "traces": [str(t) for t in traces],
                            "params": frozen[method],
                            "device": args.device,
                            "warmup_s": args.warmup,
                            "duration_s": args.duration,
                            "run_dir": str(run_dir),
                            "frozen_sha256": frozen_sha,
                        }
                    )

    # Heaviest jobs first (rl_lcr, large n) for better load balance across workers.
    order = {"rl_lcr": 0, "generic_lcr": 1, "lcr_class_delay": 1}
    jobs.sort(key=lambda j: (order.get(j["method"], 2), -j["n"], j["seed"]))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    logger.info(f"[{exp_name}] {len(jobs)} pending jobs ({args.seeds} seeds, split={args.split})")
    if args.dry_run or not jobs:
        for j in jobs:
            print(f"{j['method']:16s} {j['scenario']:10s} n={j['n']:<3d} {j['dataset']:16s} seed={j['seed']}")
        if not jobs and not pilot:
            write_summaries(args.experiment)
        return 0

    n_jobs = args.n_jobs or perf_cores()
    results = Parallel(n_jobs=n_jobs)(delayed(run_one)(j) for j in jobs)
    failed = [j for j, r in zip(jobs, results, strict=True) if r is None]
    logger.info(f"[{exp_name}] finished {len(jobs) - len(failed)}/{len(jobs)} runs")
    for j in failed:
        logger.error(
            f"FAILED {j['method']} {j['scenario']}_{j['n']} seed {j['seed']}: see {j['run_dir']}/error.txt"
        )

    if not pilot:
        write_summaries(args.experiment)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
