"""Script 14: Partition patients into stratified train/val/test splits without data leakage.

Outputs:
    - data/splits/mimic_patient_split.json
    - data/processed/traces_index.parquet (updates 'split' column)
    - data/processed/traces/<profile>/traces_index.parquet
    - reports/eda/split_summary.csv
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from rllcr.config import load_config
from rllcr.data.schemas import TRACES_INDEX_SCHEMA, validate_dataframe
from rllcr.data.splits import make_patient_split
from rllcr.utils.io import sha256_file, write_json


def main() -> None:
    """Entry point for patient-level stratified dataset partitioning."""
    parser = argparse.ArgumentParser(
        description="Split MIMIC cohort at patient level (70/15/15) to prevent patient leakage."
    )
    parser.add_argument("--windows-index", default="data/processed/windows/windows_index.parquet")
    parser.add_argument("--labels", default="data/processed/events/window_labels.parquet")
    parser.add_argument("--traces-index", default="data/processed/traces_index.parquet")
    parser.add_argument("--traces-dir", default="data/processed/traces")
    parser.add_argument("--out-split", default="data/splits/mimic_patient_split.json")
    parser.add_argument("--summary-csv", default="reports/eda/split_summary.csv")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing patient split file (test set must never change).",
    )
    args = parser.parse_args()

    cfg = load_config()
    split_cfg = cfg.data.split

    out_split_path = Path(args.out_split)
    if out_split_path.exists() and not args.force:
        logger.error(
            "Existing patient split file found at {}. "
            "Refusing to overwrite without --force! "
            "The test set must never change after experiments start.",
            out_split_path,
        )
        return
    elif out_split_path.exists() and args.force:
        logger.warning(
            "⚠️ FORCING OVERWRITE of existing split file at {}! "
            "Test set boundaries are being modified.",
            out_split_path,
        )

    windows_index_path = Path(args.windows_index)
    labels_path = Path(args.labels)

    if not windows_index_path.exists() or not labels_path.exists():
        logger.error(
            "Prerequisites missing: windows_index ({}) or labels ({})",
            windows_index_path.exists(),
            labels_path.exists(),
        )
        return

    df_windows = pd.read_parquet(windows_index_path)
    df_labels = pd.read_parquet(labels_path)

    # 1. Compute SHA-256 hash of input files
    hash_windows = sha256_file(windows_index_path)
    hash_labels = sha256_file(labels_path)
    sha256_inputs = f"windows_index:{hash_windows}|window_labels:{hash_labels}"

    # 2. Partition patients
    logger.info("Computing stratified patient-level partition with seed {}", split_cfg.seed)
    splits = make_patient_split(
        df_windows=df_windows,
        df_labels=df_labels,
        train_ratio=split_cfg.train,
        val_ratio=split_cfg.val,
        test_ratio=split_cfg.test,
        seed=split_cfg.seed,
    )

    # 3. Write data/splits/mimic_patient_split.json
    out_split_path.parent.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "seed": split_cfg.seed,
        "created": datetime.now(UTC).isoformat(),
        "sha256_of_inputs": sha256_inputs,
    }
    write_json(split_payload, out_split_path)
    logger.success("Saved patient partition to {}", out_split_path)

    # 4. Fill the 'split' column in traces_index.parquet
    patient_to_split: dict[int, str] = {}
    for sp_name, pids in splits.items():
        for p in pids:
            patient_to_split[p] = sp_name

    traces_index_path = Path(args.traces_index)
    if traces_index_path.exists():
        df_traces = pd.read_parquet(traces_index_path)
        df_traces["split"] = df_traces["subject_id"].map(
            lambda sid: patient_to_split.get(int(sid), "train")
        )
        validate_dataframe(df_traces, TRACES_INDEX_SCHEMA)
        df_traces.to_parquet(traces_index_path, index=False)
        logger.success("Updated 'split' column in {}", traces_index_path)

    # Also update per-profile traces_index if existing
    traces_dir = Path(args.traces_dir)
    if traces_dir.exists():
        for prof_dir in traces_dir.iterdir():
            if prof_dir.is_dir():
                prof_idx = df_traces[df_traces["scenario_profile"] == prof_dir.name]
                if not prof_idx.empty:
                    prof_idx_path = prof_dir / "traces_index.parquet"
                    prof_idx.to_parquet(prof_idx_path, index=False)

    # 5. Build detailed per-split report
    df_merged = df_windows.merge(df_labels, on="window_id")
    df_merged["split"] = df_merged["subject_id"].map(
        lambda sid: patient_to_split.get(int(sid), "train")
    )
    valid_merged = df_merged[df_merged["quality_ok"] == True]  # noqa: E712

    summary_rows = []
    classes = ["ROUTINE", "DESATURATION", "HYPOTENSION", "TACHY_BRADY", "ARRHYTHMIA"]

    for sp_name in ("train", "val", "test"):
        sp_valid = valid_merged[valid_merged["split"] == sp_name]
        sp_all = df_merged[df_merged["split"] == sp_name]

        n_pats = len(splits[sp_name])
        n_recs = int(sp_all["record_name"].nunique())
        tot_hours = float((sp_all["end_s"].sum() - sp_all["start_s"].sum()) / 3600.0)
        n_wins = len(sp_valid)
        n_crit = int((sp_valid["is_critical"] == True).sum())  # noqa: E712
        crit_pct = (n_crit / n_wins * 100) if n_wins > 0 else 0.0

        row_dict: dict[str, str | int | float] = {
            "split": sp_name,
            "patients": n_pats,
            "records": n_recs,
            "hours": round(tot_hours, 2),
            "windows": n_wins,
            "critical_windows": n_crit,
            "critical_pct": round(crit_pct, 2),
        }

        # Class counts
        for cls in classes:
            row_dict[cls] = int((sp_valid["primary_class"] == cls).sum())

        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = Path(args.summary_csv)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_csv_path, index=False)
    logger.success("Saved split summary report to {}", summary_csv_path)

    # 6. Check pool sizes against scenario needs
    test_patients = len(splits["test"])
    logger.info(
        "Checking pool sizes against scenario needs: "
        "Ward (20/35/50 nodes), Cohort (100/150/200 nodes), Ambulatory (20/35/50 nodes)"
    )
    if test_patients < 200:
        logger.warning(
            "Cohort-scale scenario requires 200 nodes; test split contains {} patients. "
            "Window-level sampling with replacement across distinct 5-minute windows "
            "will be utilized during 200-node simulation runs.",
            test_patients,
        )

    # 7. Print summary table
    print("\n" + "=" * 90)
    print("                    PATIENT-LEVEL STRATIFIED SPLIT SUMMARY")
    print("=" * 90)
    print(
        f"{'Split':<8} | {'Patients':<8} | {'Records':<8} | {'Hours':<8} | "
        f"{'Windows':<8} | {'Crit Win':<9} | {'Crit %':<8} | {'Routine':<8} | {'Acute (D/H/T)':<12}"
    )
    print("-" * 90)
    for _, r in df_summary.iterrows():
        acute_cnt = r["DESATURATION"] + r["HYPOTENSION"] + r["TACHY_BRADY"] + r["ARRHYTHMIA"]
        print(
            f"{r['split']:<8} | {r['patients']:<8} | {r['records']:<8} | {r['hours']:>6.2f}h | "
            f"{r['windows']:<8} | {r['critical_windows']:<9} | {r['critical_pct']:>6.2f}% | "
            f"{r['ROUTINE']:<8} | {acute_cnt:<12}"
        )
    print("-" * 90)
    tot_pats = sum(df_summary["patients"])
    tot_wins = sum(df_summary["windows"])
    tot_crit = sum(df_summary["critical_windows"])
    tot_pct = (tot_crit / tot_wins * 100) if tot_wins > 0 else 0.0
    print(
        f"{'TOTAL':<8} | {tot_pats:<8} | {df_merged['record_name'].nunique():<8} | "
        f"{sum(df_summary['hours']):>6.2f}h | {tot_wins:<8} | {tot_crit:<9} | {tot_pct:>6.2f}% | — | —"
    )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
