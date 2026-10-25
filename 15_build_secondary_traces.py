"""Script 15: Generate zero-shot evaluation packet traces for secondary datasets.

Builds evaluation traces for:
    - BIDMC (53 ICU records at 125 Hz)
    - PPG-DaLiA (15 ambulatory subjects with wrist BVP + chest ECG)
    - eICU (sample of ICU stays with 5-minute periodic vitals)

Outputs:
    - data/processed/secondary/<dataset>/traces/*.parquet (schema packets)
    - data/processed/secondary/<dataset>/traces_index.parquet (schema traces_index)
    - reports/eda/secondary_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from rllcr.config import load_config
from rllcr.data.secondary import (
    build_bidmc_traces,
    build_eicu_traces,
    build_ppg_dalia_traces,
    check_dataset_status,
)


def main() -> None:
    """Entry point for secondary zero-shot trace generation."""
    parser = argparse.ArgumentParser(
        description="Convert secondary evaluation datasets into canonical packet traces."
    )
    parser.add_argument("--bidmc-dir", default="data/raw/bidmc")
    parser.add_argument("--dalia-dir", default="data/raw/ppg_dalia")
    parser.add_argument(
        "--eicu-parquet", default="data/interim/clinical_parquet/eicu_vitalperiodic.parquet"
    )
    parser.add_argument("--out-base", default="data/processed/secondary")
    parser.add_argument("--reports-dir", default="reports/eda")
    parser.add_argument("--eicu-stays", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    out_base = Path(args.out_base)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    # 1. BIDMC
    bidmc_status = check_dataset_status("bidmc", args.bidmc_dir)
    if bidmc_status == "READY":
        logger.info("Processing BIDMC zero-shot evaluation dataset...")
        bidmc_idx, bidmc_sum = build_bidmc_traces(
            raw_dir=args.bidmc_dir,
            out_dir=out_base / "bidmc",
            config=cfg,
        )
        logger.success("BIDMC: Generated {} traces", len(bidmc_idx))
        tot_hrs = float(bidmc_sum["hours"].sum())
        mean_load = float(bidmc_sum["offered_load_kbps"].mean())
        mean_crit = float(bidmc_sum["crit_fraction"].mean())

        # Class counts
        cls_counts = bidmc_sum["primary_class"].value_counts().to_dict()
        summary_records.append(
            {
                "dataset": "BIDMC",
                "subjects": len(bidmc_idx),
                "hours": round(tot_hrs, 2),
                "offered_load_kbps": round(mean_load, 2),
                "crit_fraction": round(mean_crit, 4),
                "ROUTINE": cls_counts.get("ROUTINE", 0),
                "DESATURATION": cls_counts.get("DESATURATION", 0),
                "HYPOTENSION": cls_counts.get("HYPOTENSION", 0),
                "TACHY_BRADY": cls_counts.get("TACHY_BRADY", 0),
                "ARRHYTHMIA": cls_counts.get("ARRHYTHMIA", 0),
            }
        )
    else:
        logger.warning("BIDMC dataset not ready: status {}", bidmc_status)

    # 2. PPG-DaLiA
    dalia_status = check_dataset_status("ppg_dalia", args.dalia_dir)
    if dalia_status == "READY":
        logger.info("Processing PPG-DaLiA zero-shot ambulatory dataset...")
        dalia_idx, dalia_sum = build_ppg_dalia_traces(
            raw_dir=args.dalia_dir,
            out_dir=out_base / "ppg_dalia",
            config=cfg,
        )
        logger.success("PPG-DaLiA: Generated {} traces", len(dalia_idx))
        tot_hrs = float(dalia_sum["hours"].sum())
        mean_load = float(dalia_sum["offered_load_kbps"].mean())
        mean_crit = float(dalia_sum["crit_fraction"].mean())
        cls_counts = dalia_sum["primary_class"].value_counts().to_dict()
        summary_records.append(
            {
                "dataset": "PPG-DaLiA",
                "subjects": len(dalia_idx),
                "hours": round(tot_hrs, 2),
                "offered_load_kbps": round(mean_load, 2),
                "crit_fraction": round(mean_crit, 4),
                "ROUTINE": cls_counts.get("ROUTINE", 0),
                "DESATURATION": cls_counts.get("DESATURATION", 0),
                "HYPOTENSION": cls_counts.get("HYPOTENSION", 0),
                "TACHY_BRADY": cls_counts.get("TACHY_BRADY", 0),
                "ARRHYTHMIA": cls_counts.get("ARRHYTHMIA", 0),
            }
        )
    else:
        logger.warning("PPG-DaLiA dataset not ready: status {}", dalia_status)

    # 3. eICU
    eicu_parquet_path = Path(args.eicu_parquet)
    if eicu_parquet_path.exists():
        logger.info("Processing eICU zero-shot bedside vitals dataset...")
        eicu_idx, eicu_sum = build_eicu_traces(
            vital_parquet=eicu_parquet_path,
            out_dir=out_base / "eicu",
            config=cfg,
            n_stays=args.eicu_stays,
        )
        logger.success("eICU: Generated {} traces", len(eicu_idx))
        tot_hrs = float(eicu_sum["hours"].sum())
        mean_load = float(eicu_sum["offered_load_kbps"].mean())
        mean_crit = float(eicu_sum["crit_fraction"].mean())
        cls_counts = eicu_sum["primary_class"].value_counts().to_dict()
        summary_records.append(
            {
                "dataset": "eICU",
                "subjects": len(eicu_idx),
                "hours": round(tot_hrs, 2),
                "offered_load_kbps": round(mean_load, 2),
                "crit_fraction": round(mean_crit, 4),
                "ROUTINE": cls_counts.get("ROUTINE", 0),
                "DESATURATION": cls_counts.get("DESATURATION", 0),
                "HYPOTENSION": cls_counts.get("HYPOTENSION", 0),
                "TACHY_BRADY": cls_counts.get("TACHY_BRADY", 0),
                "ARRHYTHMIA": cls_counts.get("ARRHYTHMIA", 0),
            }
        )
    else:
        logger.warning("eICU parquet not found: {}", eicu_parquet_path)

    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = reports_dir / "secondary_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.success("Saved secondary summary to {}", summary_csv_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("               SECONDARY ZERO-SHOT EVALUATION DATASET SUMMARY")
    print("=" * 90)
    print(
        f"{'Dataset':<15} | {'Subjects':<10} | {'Hours':<8} | {'Load (kbps)':<12} | "
        f"{'Crit Frac':<10} | {'Routine':<8} | {'Acute Events':<12}"
    )
    print("-" * 90)
    for _, r in df_summary.iterrows():
        acute = r["DESATURATION"] + r["HYPOTENSION"] + r["TACHY_BRADY"] + r["ARRHYTHMIA"]
        print(
            f"{r['dataset']:<15} | {r['subjects']:<10} | {r['hours']:>6.1f}h | "
            f"{r['offered_load_kbps']:>10.2f} kbps | {r['crit_fraction']:>9.4f} | "
            f"{r['ROUTINE']:<8} | {acute:<12}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
