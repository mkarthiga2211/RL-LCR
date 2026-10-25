"""Script 13: Build packet traces for ICU full and ambulatory profiles.

Outputs:
    - data/processed/traces/<profile>/<record>.parquet (schema packets)
    - data/processed/traces_index.parquet (schema traces_index)
    - reports/eda/traffic_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from rllcr.data.schemas import PACKETS_SCHEMA, TRACES_INDEX_SCHEMA, validate_dataframe
from rllcr.data.traffic import build_trace_for_record
from rllcr.utils.hw import perf_cores


def process_single_trace(
    record_name: str,
    subject_id: int,
    df_rec_windows: pd.DataFrame,
    df_rec_labels: pd.DataFrame,
    profile: str,
    out_dir: Path,
) -> dict[str, str | int | float]:
    """Build and save packet trace for a single record and scenario profile.

    Args:
        record_name: Name of patient record.
        subject_id: Subject ID.
        df_rec_windows: Windows index rows for this record.
        df_rec_labels: Window labels rows for this record.
        profile: Scenario profile identifier.
        out_dir: Output directory for this profile's traces.

    Returns:
        dict[str, Any]: Trace summary statistics row.
    """
    df_trace, stats = build_trace_for_record(
        record_name=record_name,
        subject_id=subject_id,
        df_record_windows=df_rec_windows,
        df_record_labels=df_rec_labels,
        profile=profile,
    )

    validate_dataframe(df_trace, PACKETS_SCHEMA)
    trace_path = out_dir / f"{record_name}.parquet"
    df_trace.to_parquet(trace_path, index=False)

    return stats


def main() -> None:
    """Entry point for packet trace generation script."""
    parser = argparse.ArgumentParser(description="Generate data-grounded packet simulation traces.")
    parser.add_argument("--windows-index", default="data/processed/windows/windows_index.parquet")
    parser.add_argument("--labels", default="data/processed/events/window_labels.parquet")
    parser.add_argument("--traces-dir", default="data/processed/traces")
    parser.add_argument("--reports-dir", default="reports/eda")
    args = parser.parse_args()

    traces_base_dir = Path(args.traces_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_windows = pd.read_parquet(args.windows_index)
    df_labels = pd.read_parquet(args.labels)

    profiles = ["icu_full", "ambulatory_subsampled"]
    n_workers = perf_cores()
    all_trace_stats = []

    logger.info("Building traces for profiles: {} across {} workers", profiles, n_workers)

    distinct_records = df_windows["record_name"].unique()

    for prof in profiles:
        prof_dir = traces_base_dir / prof
        prof_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Processing profile '{}' for {} records...", prof, len(distinct_records))

        results = Parallel(n_jobs=n_workers)(
            delayed(process_single_trace)(
                record_name=rec,
                subject_id=int(
                    df_windows.loc[df_windows["record_name"] == rec, "subject_id"].iloc[0]
                ),
                df_rec_windows=df_windows[df_windows["record_name"] == rec],
                df_rec_labels=df_labels[df_labels["window_id"].isin(
                    df_windows.loc[df_windows["record_name"] == rec, "window_id"]
                )],
                profile=prof,
                out_dir=prof_dir,
            )
            for rec in distinct_records
        )
        all_trace_stats.extend(results)

    # Build traces index conforming to TRACES_INDEX_SCHEMA
    df_stats = pd.DataFrame(all_trace_stats)
    # Default split placeholder (train/val/test assigned per patient partition)
    # 70/15/15 deterministic assignment by subject_id
    splits = []
    for s_id in df_stats["subject_id"]:
        # Deterministic split hashing
        val = (int(s_id) * 2026) % 100
        if val < 70:
            splits.append("train")
        elif val < 85:
            splits.append("val")
        else:
            splits.append("test")

    df_traces_index = pd.DataFrame(
        {
            "trace_id": df_stats["trace_id"],
            "subject_id": df_stats["subject_id"].astype(int),
            "split": splits,
            "scenario_profile": df_stats["scenario_profile"],
            "n_packets": df_stats["n_packets"].astype(int),
            "duration_s": df_stats["duration_s"].astype(float),
            "crit_fraction": df_stats["crit_fraction"].astype(float),
        }
    )

    validate_dataframe(df_traces_index, TRACES_INDEX_SCHEMA)
    traces_index_path = traces_base_dir.parent / "traces_index.parquet"
    df_traces_index.to_parquet(traces_index_path, index=False)
    logger.success("Saved traces index to {}", traces_index_path)

    # Compute summary metrics per scenario profile
    summary_records = []
    for prof in profiles:
        sub_stats = df_stats[df_stats["scenario_profile"] == prof]
        mean_load = float(sub_stats["offered_load_kbps"].mean())
        mean_crit_frac = float(sub_stats["crit_fraction"].mean())
        mean_burstiness = float(sub_stats["burstiness"].mean())

        # Check IEEE 802.15.4 saturation threshold (50 kbps)
        saturation_warning = "ALERT (>50 kbps)" if mean_load > 50.0 else "OK (<50 kbps)"

        summary_records.append(
            {
                "profile": prof,
                "n_traces": len(sub_stats),
                "total_packets": int(sub_stats["n_packets"].sum()),
                "mean_offered_load_kbps": round(mean_load, 2),
                "mean_crit_fraction": round(mean_crit_frac, 4),
                "mean_burstiness": round(mean_burstiness, 2),
                "status_802_15_4": saturation_warning,
            }
        )

    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = reports_dir / "traffic_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.success("Saved traffic summary to {}", summary_csv_path)

    # Print summary table
    print("\n" + "=" * 85)
    print("                  TRAFFIC GENERATION SUMMARY PER PROFILE")
    print("=" * 85)
    print(
        f"{'Profile':<25} | {'Traces':<8} | {'Packets':<10} | {'Offered (kbps)':<15} | "
        f"{'Crit Frac':<10} | {'Burstiness':<10}"
    )
    print("-" * 85)
    for _, r in df_summary.iterrows():
        print(
            f"{r['profile']:<25} | {r['n_traces']:<8} | {r['total_packets']:<10} | "
            f"{r['mean_offered_load_kbps']:>13.2f} kbps | {r['mean_crit_fraction']:>9.4f} | "
            f"{r['mean_burstiness']:>9.2f}"
        )
    print("-" * 85)
    for _, r in df_summary.iterrows():
        if r["mean_offered_load_kbps"] > 50.0:
            print(f"WARNING: Profile '{r['profile']}' offered load > 50 kbps (saturates 802.15.4)!")
        else:
            print(
                f"Status '{r['profile']}': {r['mean_offered_load_kbps']:.2f} kbps "
                "[Nominal 802.15.4 WBAN load]"
            )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
