"""Script 12: Extract clinical events, label window criticality, and assign delay budgets.

Outputs:
    - data/processed/events/mimic_events.parquet (schema events)
    - data/processed/events/window_labels.parquet (schema window_labels)
    - reports/eda/event_prevalence.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from rllcr.config import load_config
from rllcr.data.criticality import label_windows_criticality
from rllcr.data.events import extract_events_from_numerics
from rllcr.data.schemas import EVENTS_SCHEMA, WINDOW_LABELS_SCHEMA, validate_dataframe


def main() -> None:
    """Entry point for clinical event extraction and window labeling."""
    parser = argparse.ArgumentParser(
        description="Extract clinical events from numerics/charts and label window criticality."
    )
    parser.add_argument("--windows-index", default="data/processed/windows/windows_index.parquet")
    parser.add_argument("--numerics-dir", default="data/interim/windows")
    parser.add_argument(
        "--chartevents", default="data/interim/clinical_parquet/chartevents_subset.parquet"
    )
    parser.add_argument("--out-dir", default="data/processed/events")
    parser.add_argument("--reports-dir", default="reports/eda")
    args = parser.parse_args()

    cfg = load_config()
    delay_budgets = cfg.clinical_constraints.delay_budget_ms

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    windows_index_path = Path(args.windows_index)
    if not windows_index_path.exists():
        logger.error("Windows index not found: {}", windows_index_path)
        return

    df_windows = pd.read_parquet(windows_index_path)
    logger.info("Loaded {} windows from {}", len(df_windows), windows_index_path)

    numerics_dir = Path(args.numerics_dir)
    chartevents_path = Path(args.chartevents)

    if chartevents_path.exists():
        logger.info("Found clinical chart events at {}", chartevents_path)
    else:
        logger.info(
            "chartevents_subset.parquet not detected. "
            "Executing Phase 2.C in NUMERICS-ONLY event detection mode."
        )

    # Process all records present in df_windows
    distinct_records = df_windows["record_name"].unique()
    all_events_list = []

    for rec in distinct_records:
        rec_subj_id = int(df_windows.loc[df_windows["record_name"] == rec, "subject_id"].iloc[0])
        num_file = numerics_dir / f"{rec}_numerics.parquet"

        if not num_file.exists():
            logger.warning("Numerics file missing for record {}", rec)
            continue

        df_num = pd.read_parquet(num_file)
        if df_num.empty:
            continue

        # Extract numerics threshold events
        df_rec_events = extract_events_from_numerics(df_num, subject_id=rec_subj_id)
        if not df_rec_events.empty:
            all_events_list.append(df_rec_events)

    if all_events_list:
        df_all_events = pd.concat(all_events_list, ignore_index=True)
    else:
        df_all_events = pd.DataFrame(
            columns=["subject_id", "charttime_s", "event_type", "source", "duration_s", "value"]
        )

    validate_dataframe(df_all_events, EVENTS_SCHEMA)
    events_out_path = out_dir / "mimic_events.parquet"
    df_all_events.to_parquet(events_out_path, index=False)
    logger.success("Saved {} events to {}", len(df_all_events), events_out_path)

    # Label window criticality
    logger.info("Labeling criticality across {} windows...", len(df_windows))
    df_labels = label_windows_criticality(
        df_windows=df_windows,
        df_events=df_all_events,
        delay_budget_ms=delay_budgets,
    )

    validate_dataframe(df_labels, WINDOW_LABELS_SCHEMA)
    labels_out_path = out_dir / "window_labels.parquet"
    df_labels.to_parquet(labels_out_path, index=False)
    logger.success("Saved window labels to {}", labels_out_path)

    # Compute prevalence statistics per alarm class (windows & patients)
    # Merge subject_id for patient-level counting
    df_merged = df_labels.merge(
        df_windows[["window_id", "subject_id", "quality_ok"]], on="window_id", how="left"
    )

    prevalence_records = []
    all_classes = list(delay_budgets.keys())
    total_valid_windows = int((df_merged["quality_ok"] == True).sum())  # noqa: E712
    total_patients = int(df_merged["subject_id"].nunique())

    for cls_name in all_classes:
        # Windows where this class is the primary class
        cls_windows = df_merged[
            (df_merged["primary_class"] == cls_name) & (df_merged["quality_ok"] == True)  # noqa: E712
        ]
        n_win = len(cls_windows)
        pct_win = (n_win / total_valid_windows * 100) if total_valid_windows > 0 else 0.0
        n_pat = int(cls_windows["subject_id"].nunique())
        pct_pat = (n_pat / total_patients * 100) if total_patients > 0 else 0.0

        prevalence_records.append(
            {
                "alarm_class": cls_name,
                "dmax_ms": delay_budgets[cls_name],
                "windows_count": n_win,
                "windows_pct": round(pct_win, 2),
                "patients_count": n_pat,
                "patients_pct": round(pct_pat, 2),
            }
        )

    df_prev = pd.DataFrame(prevalence_records)
    csv_prev_path = reports_dir / "event_prevalence.csv"
    df_prev.to_csv(csv_prev_path, index=False)
    logger.success("Saved event prevalence table to {}", csv_prev_path)

    # Print class prevalence table to terminal
    print("\n" + "=" * 75)
    print("                    CLINICAL ALARM CLASS PREVALENCE")
    print("=" * 75)
    print(
        f"{'Alarm Class':<15} | {'Dmax (ms)':<10} | {'Windows':<10} | {'Win %':<8} | "
        f"{'Patients':<10} | {'Pat %':<8}"
    )
    print("-" * 75)
    for _, r in df_prev.iterrows():
        print(
            f"{r['alarm_class']:<15} | {r['dmax_ms']:<10.0f} | {r['windows_count']:<10} | "
            f"{r['windows_pct']:>6.2f}% | {r['patients_count']:<10} | {r['patients_pct']:>6.2f}%"
        )
    print("-" * 75)
    print(
        f"{'TOTAL VALID':<15} | {'—':<10} | {total_valid_windows:<10} | {'100.00%':>8} | "
        f"{total_patients:<10} | {'100.00%':>8}"
    )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
