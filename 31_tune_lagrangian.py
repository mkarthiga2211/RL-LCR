"""Script 31: Hyperparameter tuning for RL-LCR Lagrangian solver and fixed multipliers.

Tunes RL-LCR's fixed multipliers and solver settings exclusively on the VALIDATION split
using an exact trial budget matching the baseline protocols (configs/baselines.yaml budget_trials = 40).
Saves the optimal parameter configuration to results/models/best_lagrangian_params.json and
configs/lagrangian_tuned.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from rllcr.config import load_config
from rllcr.eval.tuning_objective import (
    TuningObjective,
    evaluate_tuning_trial,
    get_validation_traces,
    sample_lagrangian_params,
)
from rllcr.sim.topology import Topology, generate_ward_topology
from rllcr.utils.logging import setup_logging
from rllcr.utils.seed import set_seed


def tune_lagrangian(
    budget_trials: int = 40,
    duration_s: float = 20.0,
    n_nodes: int = 20,
    seed: int = 2026,
    device_pref: str = "auto",
    qtable_path: Path | str = "results/models/qtable.npz",
    output_path: Path | str = "results/models/best_lagrangian_params.json",
    tuned_config_path: Path | str = "configs/lagrangian_tuned.yaml",
) -> dict[str, Any]:
    """Execute hyperparameter tuning for RL-LCR over the validation split.

    Args:
        budget_trials: Total number of tuning trials (identical to baseline budget).
        duration_s: Simulation clock duration per trial in seconds.
        n_nodes: Topology node count.
        seed: Random seed for parameter sampling and simulation.
        device_pref: Compute device preference ('auto', 'mps', 'cpu').
        qtable_path: Path to pre-trained Q-table model archive (.npz).
        output_path: Destination path for optimal parameters JSON summary.
        tuned_config_path: Destination path for tuned YAML configuration.

    Returns:
        dict[str, Any]: Best parameter configuration, performance metrics, and history.
    """
    t_start = time.perf_counter()
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Load validation traces strictly isolated from test cohort
    val_traces = get_validation_traces()
    logger.info(f"Loaded {len(val_traces)} validation trace files strictly from validation split.")

    # 2. Load or generate validation Ward topology
    topo_files = sorted(Path("results/topologies").glob(f"ward_{n_nodes}_*.json"))
    if topo_files:
        # Select first validation topology deterministically
        val_topo_path = topo_files[0]
        topo = Topology.load_json(val_topo_path)
        logger.info(f"Loaded validation topology: {val_topo_path.name}")
    else:
        topo = generate_ward_topology(n_nodes=n_nodes, seed=seed)
        logger.info(f"Generated validation topology with {n_nodes} nodes (seed={seed})")

    # 3. Initialize objective and trial tracking
    objective = TuningObjective()
    trial_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    logger.info(
        f"Starting RL-LCR tuning over {budget_trials} trials "
        f"(duration={duration_s}s, device={device_pref})"
    )

    # Default baseline configuration evaluated as trial 0
    default_params: dict[str, Any] = {
        "lr": 0.01,
        "iters": 50,
        "rho": 1.0,
        "omega_tx": 1.0,
        "k_candidates": 3,
        "lam_delay_init": 1.0,
        "lam_energy_init": 1.0,
    }

    for trial_idx in range(budget_trials):
        trial_t0 = time.perf_counter()

        # Trial 0 evaluates default baseline; subsequent trials sample parameter space
        if trial_idx == 0:
            params = default_params
        else:
            params = sample_lagrangian_params(rng=rng)

        # Derive reproducible seed per trial
        trial_seed = int(seed + trial_idx * 17)

        # Execute trial simulation
        result = evaluate_tuning_trial(
            params=params,
            topology=topo,
            trace_paths=val_traces,
            duration_s=duration_s,
            seed=trial_seed,
            device_pref=device_pref,
            qtable_path=Path(qtable_path),
            objective=objective,
        )

        trial_duration = time.perf_counter() - trial_t0
        result["trial"] = trial_idx
        result["wall_time_s"] = trial_duration
        trial_records.append(result)

        is_new_best = False
        if best_record is None or result["score"] < best_record["score"]:
            best_record = result
            is_new_best = True

        marker = "★ [NEW BEST]" if is_new_best else ""
        logger.info(
            f"Trial {trial_idx + 1:02d}/{budget_trials:02d} | "
            f"Score: {result['score']:.4f} | "
            f"PDR: {result['pdr'] * 100:.2f}% | "
            f"DVR: {result['dvr'] * 100:.2f}% | "
            f"Delay: {result['mean_delay_ms']:.2f} ms | "
            f"Energy: {result['total_energy_j']:.2f} J | "
            f"Time: {trial_duration:.2f} s {marker}"
        )

    t_total = time.perf_counter() - t_start
    assert best_record is not None

    # 4. Save results to JSON artifact
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "best_trial": best_record["trial"],
        "best_score": best_record["score"],
        "best_params": best_record["params"],
        "metrics": {
            "pdr": best_record["pdr"],
            "dvr": best_record["dvr"],
            "mean_delay_ms": best_record["mean_delay_ms"],
            "total_energy_j": best_record["total_energy_j"],
            "total_packets": best_record["total_packets"],
            "delivered": best_record["delivered"],
        },
        "budget_trials": budget_trials,
        "total_wall_time_s": t_total,
        "seed": seed,
        "history": [
            {
                "trial": r["trial"],
                "score": r["score"],
                "pdr": r["pdr"],
                "dvr": r["dvr"],
                "mean_delay_ms": r["mean_delay_ms"],
                "total_energy_j": r["total_energy_j"],
                "params": r["params"],
                "wall_time_s": r["wall_time_s"],
            }
            for r in trial_records
        ],
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"Saved tuning summary to {out_file}")

    # 5. Save optimal configuration to YAML
    tuned_cfg_file = Path(tuned_config_path)
    tuned_cfg_file.parent.mkdir(parents=True, exist_ok=True)
    yaml_payload = {
        "lagrangian": {
            "lr": float(best_record["params"]["lr"]),
            "iters": int(best_record["params"]["iters"]),
            "rho": float(best_record["params"]["rho"]),
            "lam_delay_init": float(best_record["params"]["lam_delay_init"]),
            "lam_energy_init": float(best_record["params"]["lam_energy_init"]),
            "omega_tx": float(best_record["params"]["omega_tx"]),
            "k_candidates": int(best_record["params"]["k_candidates"]),
        }
    }
    with tuned_cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)
    logger.info(f"Saved tuned YAML configuration to {tuned_cfg_file}")

    # Print summary table
    print("\n" + "=" * 65)
    print("RL-LCR HYPERPARAMETER TUNING COMPLETE (VALIDATION SPLIT)")
    print("=" * 65)
    print(f"Trials Completed:     {budget_trials}")
    print(f"Total Wall Time:      {t_total:.2f} s (Mean: {t_total / budget_trials:.2f} s/trial)")
    print(f"Optimal Trial:        #{best_record['trial'] + 1}")
    print(f"Best Objective Loss:  {best_record['score']:.4f}")
    print(f"Validation PDR:       {best_record['pdr'] * 100:.2f}%")
    print(f"Validation DVR:       {best_record['dvr'] * 100:.2f}%")
    print(f"Validation Latency:   {best_record['mean_delay_ms']:.2f} ms")
    print(f"Validation Energy:    {best_record['total_energy_j']:.2f} J")
    print("-" * 65)
    print("Optimal Hyperparameters:")
    for k, v in best_record["params"].items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.5f}")
        else:
            print(f"  {k:18s}: {v}")
    print("=" * 65 + "\n")

    return summary_payload


def main() -> None:
    """CLI entrypoint."""
    setup_logging("31_tune_lagrangian")
    cfg = load_config()
    default_budget = cfg.baselines.tuning.get("budget_trials", 40)

    parser = argparse.ArgumentParser(
        description="Tune RL-LCR Lagrangian multipliers and solver settings on validation split."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=default_budget,
        help=f"Number of tuning trials (default: {default_budget}).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration per trial in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=20,
        help="Number of nodes in validation topology (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed (default: 2026).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device preference ('auto', 'mps', 'cpu', default: 'auto').",
    )
    parser.add_argument(
        "--qtable",
        type=str,
        default="results/models/qtable.npz",
        help="Path to pre-trained Q-table model archive (.npz).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/models/best_lagrangian_params.json",
        help="Destination path for best parameters JSON.",
    )
    parser.add_argument(
        "--tuned-config",
        type=str,
        default="configs/lagrangian_tuned.yaml",
        help="Destination path for tuned YAML configuration.",
    )
    args = parser.parse_args()

    tune_lagrangian(
        budget_trials=args.budget,
        duration_s=args.duration,
        n_nodes=args.nodes,
        seed=args.seed,
        device_pref=args.device,
        qtable_path=args.qtable,
        output_path=args.output,
        tuned_config_path=args.tuned_config,
    )


if __name__ == "__main__":
    main()
