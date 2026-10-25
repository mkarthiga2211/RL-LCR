"""Script 40: Tune every method under the same protocol as RL-LCR, then freeze parameters.

Every method is evaluated on the SAME validation traces, topology, simulator settings, seeds,
objective J (``TuningObjective``) and trial budget (``baselines.tuning.budget_trials`` = 40) that
RL-LCR received in script 31. Trial 0 is always the default configuration, so tuning can only help
a baseline. The test split is never touched (``get_validation_traces`` enforces val-only).

Methods
    aodv, dsr, olsr, q_routing   Optuna TPE search over the spaces documented in ``SEARCH_SPACES``.
    generic_lcr                  Optuna over the uniform delay / power budgets (solver params fixed
                                 to RL-LCR's tuned values).
    lcr_class_delay              Optuna over the uniform power budget (ablation variant).
    rl_lcr                       Not re-tuned: converted from script 31's results.

Outputs
    results/tuning/<method>_best.json
    reports/tables/tuning_summary.csv
    results/tuning/FROZEN.json        (method -> params, sha256 of each file, timestamp)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from rllcr.eval.frozen import FROZEN_PATH, write_frozen
from rllcr.eval.tuning_objective import (
    TuningObjective,
    evaluate_protocol_trial,
    get_validation_traces,
)
from rllcr.sim.topology import Topology, generate_ward_topology
from rllcr.utils.logging import setup_logging
from rllcr.utils.seed import set_seed

# Methods frozen into FROZEN.json (six methods + ablation variant).
BASELINE_METHODS: tuple[str, ...] = ("aodv", "dsr", "olsr", "q_routing")
LCR_METHODS: tuple[str, ...] = ("generic_lcr", "lcr_class_delay")
ALL_METHODS: tuple[str, ...] = ("rl_lcr", *BASELINE_METHODS, *LCR_METHODS)

RL_LCR_SOURCE = Path("results/models/best_lagrangian_params.json")
METRIC_KEYS = ("pdr", "dvr", "mean_delay_ms", "total_energy_j")
LAGRANGIAN_KEYS = ("lr", "iters", "rho", "omega_tx", "k_candidates", "lam_delay_init", "lam_energy_init")

# Trial-0 defaults. AODV: node_traversal_ms=50 with net_diameter=30 gives the 3.0 s discovery
# timeout from configs/baselines.yaml. OLSR hold times follow RFC 3626 (3x the interval).
DEFAULTS: dict[str, dict[str, Any]] = {
    "aodv": {"active_route_timeout_s": 3.0, "rreq_retries": 3, "node_traversal_ms": 50.0},
    "dsr": {"cache_timeout_s": 10.0, "reply_from_cache": True},
    "olsr": {"hello_interval_s": 2.0, "tc_interval_s": 5.0},
    "q_routing": {"alpha": 0.10, "epsilon": 0.10, "gamma": 0.95},
    "generic_lcr": {"uniform_delay_budget_ms": 500.0, "uniform_power_budget_mw": 50.0},
    "lcr_class_delay": {"uniform_power_budget_mw": 50.0},
}


def suggest_aodv(trial: optuna.Trial) -> dict[str, Any]:
    """AODV: active_route_timeout_s [1,10], rreq_retries {1,2,3}, node_traversal_ms [10,80]."""
    return {
        "active_route_timeout_s": trial.suggest_float("active_route_timeout_s", 1.0, 10.0),
        "rreq_retries": trial.suggest_categorical("rreq_retries", [1, 2, 3]),
        "node_traversal_ms": trial.suggest_float("node_traversal_ms", 10.0, 80.0),
    }


def suggest_dsr(trial: optuna.Trial) -> dict[str, Any]:
    """DSR: cache_timeout_s [5,120], intermediate_replies {true,false} (= reply_from_cache)."""
    return {
        "cache_timeout_s": trial.suggest_float("cache_timeout_s", 5.0, 120.0),
        "reply_from_cache": trial.suggest_categorical("reply_from_cache", [True, False]),
    }


def suggest_olsr(trial: optuna.Trial) -> dict[str, Any]:
    """OLSR: hello_s [0.5,5], tc_s [2,15] with tc_s >= hello_s; hold times scale 3x (RFC 3626)."""
    hello = trial.suggest_float("hello_interval_s", 0.5, 5.0)
    tc = trial.suggest_float("tc_interval_s", max(2.0, hello), 15.0)
    return {
        "hello_interval_s": hello,
        "tc_interval_s": tc,
        "neighbor_hold_time_s": 3.0 * hello,
        "topology_hold_time_s": 3.0 * tc,
    }


def suggest_q_routing(trial: optuna.Trial) -> dict[str, Any]:
    """Q-routing: alpha [0.01,0.5] (log), epsilon [0.01,0.3], gamma [0.8,0.99]."""
    epsilon = trial.suggest_float("epsilon", 0.01, 0.3)
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 0.5, log=True),
        "epsilon": epsilon,
        "epsilon_min": min(0.05, epsilon),
        "gamma": trial.suggest_float("gamma", 0.8, 0.99),
    }


def suggest_generic_lcr(trial: optuna.Trial) -> dict[str, Any]:
    """Generic-LCR: uniform delay budget [100,2000] ms and uniform power budget [0.5,200] mW."""
    return {
        "uniform_delay_budget_ms": trial.suggest_float("uniform_delay_budget_ms", 100.0, 2000.0, log=True),
        "uniform_power_budget_mw": trial.suggest_float("uniform_power_budget_mw", 0.5, 200.0, log=True),
    }


def suggest_class_delay(trial: optuna.Trial) -> dict[str, Any]:
    """LCR-class-delay: uniform power budget [0.5,200] mW (delay budgets stay clinical)."""
    return {
        "uniform_power_budget_mw": trial.suggest_float("uniform_power_budget_mw", 0.5, 200.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "aodv": suggest_aodv,
    "dsr": suggest_dsr,
    "olsr": suggest_olsr,
    "q_routing": suggest_q_routing,
    "generic_lcr": suggest_generic_lcr,
    "lcr_class_delay": suggest_class_delay,
}


def load_rl_lcr_lagrangian() -> dict[str, Any]:
    """Return RL-LCR's tuned solver parameters (fixed inside the generic LCR variants)."""
    payload = json.loads(RL_LCR_SOURCE.read_text(encoding="utf-8"))
    return {k: payload["best_params"][k] for k in LAGRANGIAN_KEYS}


def convert_rl_lcr(out_dir: Path, overwrite: bool) -> Path:
    """Write ``rl_lcr_best.json`` from script 31's results (RL-LCR is not re-tuned)."""
    out = out_dir / "rl_lcr_best.json"
    if out.exists() and not overwrite:
        logger.info(f"[rl_lcr] {out} exists; skipping (use --overwrite).")
        return out
    src = json.loads(RL_LCR_SOURCE.read_text(encoding="utf-8"))
    payload = {
        "method": "rl_lcr",
        "source": str(RL_LCR_SOURCE),
        "sampler": "numpy random search (script 31), trial 0 = defaults",
        "best_trial": src["best_trial"],
        "best_score": src["best_score"],
        "best_params": src["best_params"],
        "metrics": src["metrics"],
        "budget_trials": src["budget_trials"],
        "seed": src["seed"],
        "history": src["history"],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"[rl_lcr] converted from {RL_LCR_SOURCE} -> {out}")
    return out


def tune_method(
    method: str,
    *,
    topology: Topology,
    traces: list[Path],
    objective: TuningObjective,
    budget_trials: int,
    duration_s: float,
    seed: int,
    fixed: dict[str, Any],
    out_dir: Path,
    overwrite: bool,
) -> Path:
    """Tune one method with Optuna (TPE) and write ``<method>_best.json``."""
    out = out_dir / f"{method}_best.json"
    if out.exists() and not overwrite:
        logger.info(f"[{method}] {out} exists; skipping (use --overwrite).")
        return out

    history: list[dict[str, Any]] = []
    space = SEARCH_SPACES[method]

    def objective_fn(trial: optuna.Trial) -> float:
        tuned = space(trial)
        t0 = time.perf_counter()
        # Same per-trial seed schedule as script 31: seed + trial_idx * 17.
        result = evaluate_protocol_trial(
            method,
            params={**fixed, **tuned},
            topology=topology,
            trace_paths=traces,
            duration_s=duration_s,
            seed=int(seed + trial.number * 17),
            objective=objective,
            protocol_kwargs=_ctor_extras(method, seed + trial.number * 17),
        )
        result["trial"] = trial.number
        result["wall_time_s"] = time.perf_counter() - t0
        history.append(result)
        logger.info(
            f"[{method}] trial {trial.number + 1:02d}/{budget_trials} | score {result['score']:.4f} | "
            f"PDR {result['pdr'] * 100:.2f}% | DVR {result['dvr'] * 100:.2f}% | "
            f"delay {result['mean_delay_ms']:.2f} ms | E {result['total_energy_j']:.2f} J | "
            f"{result['wall_time_s']:.1f}s"
        )
        return float(result["score"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULTS[method])  # trial 0 = default configuration
    t_start = time.perf_counter()
    study.optimize(objective_fn, n_trials=budget_trials)

    best = min(history, key=lambda r: (r["score"], r["trial"]))
    payload = {
        "method": method,
        "sampler": "optuna TPESampler",
        "best_trial": best["trial"],
        "best_score": best["score"],
        "best_params": best["params"],
        "metrics": {k: best[k] for k in METRIC_KEYS + ("total_packets", "delivered")},
        "budget_trials": budget_trials,
        "seed": seed,
        "duration_s": duration_s,
        "objective": asdict(objective),
        "total_wall_time_s": time.perf_counter() - t_start,
        "history": [
            {k: r[k] for k in ("trial", "score", *METRIC_KEYS, "params", "wall_time_s")}
            for r in sorted(history, key=lambda r: r["trial"])
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"[{method}] best trial {best['trial']} score {best['score']:.4f} -> {out}")
    return out


_DEVICE: str = "auto"


def _ctor_extras(method: str, trial_seed: int) -> dict[str, Any]:
    """Non-tuned constructor kwargs that depend on the method (device, protocol seed)."""
    if method in LCR_METHODS:
        return {"device_pref": _DEVICE}
    if method in ("olsr", "q_routing"):
        return {"seed": trial_seed}
    return {}


def write_summary(out_dir: Path, summary_path: Path) -> list[dict[str, Any]]:
    """Write and print the tuning summary table from the per-method JSON files."""
    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        payload = json.loads((out_dir / f"{method}_best.json").read_text(encoding="utf-8"))
        default_score = payload["history"][0]["score"]
        best_score = payload["best_score"]
        rows.append(
            {
                "method": method,
                "n_trials": len(payload["history"]),
                "default_score": round(default_score, 4),
                "best_score": round(best_score, 4),
                "improvement_pct": round(100.0 * (default_score - best_score) / default_score, 2),
                "best_trial": payload["best_trial"],
                "pdr": round(payload["metrics"]["pdr"], 4),
                "dvr": round(payload["metrics"]["dvr"], 4),
                "mean_delay_ms": round(payload["metrics"]["mean_delay_ms"], 2),
                "total_energy_j": round(payload["metrics"]["total_energy_j"], 3),
                "best_params": json.dumps(payload["best_params"], sort_keys=True),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'method':16s} {'trials':>6s} {'default':>8s} {'best':>8s} {'gain%':>7s} "
        f"{'PDR':>7s} {'DVR':>7s} {'delay_ms':>9s} {'E_J':>7s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:16s} {r['n_trials']:6d} {r['default_score']:8.4f} {r['best_score']:8.4f} "
            f"{r['improvement_pct']:7.2f} {r['pdr']:7.4f} {r['dvr']:7.4f} "
            f"{r['mean_delay_ms']:9.2f} {r['total_energy_j']:7.3f}"
        )
    print(f"\nSummary written to {summary_path}\n")
    return rows


def main() -> None:
    """CLI entrypoint."""
    global _DEVICE
    setup_logging("40_tune_baselines")
    parser = argparse.ArgumentParser(
        description="Tune all baselines on validation data and freeze parameters."
    )
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    parser.add_argument("--trials", type=int, default=40, help="Trials per method (matches RL-LCR budget).")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation seconds per trial.")
    parser.add_argument("--nodes", type=int, default=20, help="Ward topology size (same as script 31).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="mps", help="Device for the Lagrangian variants: auto|mps|cpu.")
    parser.add_argument("--output-dir", default="results/tuning")
    parser.add_argument("--summary", default="reports/tables/tuning_summary.csv")
    parser.add_argument("--overwrite", action="store_true", help="Re-tune methods and overwrite FROZEN.json.")
    args = parser.parse_args()

    _DEVICE = args.device
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same validation data and topology as script 31 (validation split only).
    traces = get_validation_traces()
    topo_files = sorted(Path("results/topologies").glob(f"ward_{args.nodes}_*.json"))
    topology = (
        Topology.load_json(topo_files[0])
        if topo_files
        else generate_ward_topology(n_nodes=args.nodes, seed=args.seed)
    )
    logger.info(f"{len(traces)} validation traces | topology {args.nodes} nodes | device {_DEVICE}")

    objective = TuningObjective()
    lagrangian = load_rl_lcr_lagrangian()

    for method in args.methods:
        if method == "rl_lcr":
            convert_rl_lcr(out_dir, args.overwrite)
            continue
        fixed = dict(lagrangian) if method in LCR_METHODS else {}
        tune_method(
            method,
            topology=topology,
            traces=traces,
            objective=objective,
            budget_trials=args.trials,
            duration_s=args.duration,
            seed=args.seed,
            fixed=fixed,
            out_dir=out_dir,
            overwrite=args.overwrite,
        )

    missing = [m for m in ALL_METHODS if not (out_dir / f"{m}_best.json").exists()]
    if missing:
        logger.warning(f"Not freezing: missing tuning files for {missing}")
        return

    write_summary(out_dir, Path(args.summary))

    frozen_path = out_dir / FROZEN_PATH.name
    if frozen_path.exists() and not args.overwrite:
        logger.warning(f"{frozen_path} already exists; not overwriting frozen parameters (use --overwrite).")
        return
    manifest = write_frozen(
        {m: out_dir / f"{m}_best.json" for m in ALL_METHODS},
        out_path=frozen_path,
        extra={
            "split": "val",
            "budget_trials": args.trials,
            "seed": args.seed,
            "duration_s": args.duration,
            "objective": asdict(objective),
        },
    )
    logger.info(f"Froze {len(manifest['methods'])} methods -> {frozen_path}")


if __name__ == "__main__":
    main()
