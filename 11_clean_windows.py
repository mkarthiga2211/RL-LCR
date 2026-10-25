"""Script 11: Clean, filter, normalize windows, and generate processed dataset.

Outputs:
    - data/processed/windows/<record>.parquet (schema window_signals)
    - data/processed/windows/windows_index.parquet (schema windows_index)
    - reports/eda/cleaning_summary.json
    - docs/CLEANING_RULES.md (with exclusion counts table)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from rllcr.config import load_config
from rllcr.data.cleaning import clean_window
from rllcr.data.schemas import WINDOW_SIGNALS_SCHEMA, WINDOWS_INDEX_SCHEMA, validate_dataframe
from rllcr.utils.hw import perf_cores


def clean_single_record(
    record_name: str,
    record_windows_index: pd.DataFrame,
    interim_dir: Path,
    processed_dir: Path,
    max_flatline_s: float,
    max_gap_frac: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean all windows for a single patient record.

    Args:
        record_name: Patient record identifier.
        record_windows_index: Sliced window index subset for this record.
        interim_dir: Directory containing interim window parquets.
        processed_dir: Target output directory for processed records.
        max_flatline_s: Maximum flatline duration.
        max_gap_frac: Maximum tolerable gap fraction.

    Returns:
        tuple containing:
            - pd.DataFrame: Updated window index rows.
            - dict[str, int]: Exclusion counts dictionary for this record.
    """
    interim_file = interim_dir / f"{record_name}.parquet"
    if not interim_file.exists():
        logger.warning("Interim file for record {} does not exist.", record_name)
        return record_windows_index, {}

    df_record_raw = pd.read_parquet(interim_file)
    updated_rows = []
    cleaned_window_dfs = []
    exclusion_counts: dict[str, int] = {
        "passed": 0,
        "gap": 0,
        "flatline": 0,
        "abp_range": 0,
        "ecg_sat": 0,
        "missing_channels": 0,
    }

    for _, row in record_windows_index.iterrows():
        w_id = row["window_id"]
        w_df = df_record_raw[df_record_raw["window_id"] == w_id]

        if w_df.empty:
            exclusion_counts["gap"] += 1
            row_dict = dict(row)
            row_dict["quality_ok"] = False
            row_dict["exclusion_reason"] = "gap"
            updated_rows.append(row_dict)
            continue

        # Parse fs from json or default 125.0
        try:
            fs_map = json.loads(row["fs"])
            fs = float(next(iter(fs_map.values())))
        except Exception:
            fs = 125.0

        cleaned_tidy_df, quality_ok, reason, _ = clean_window(
            window_df=w_df,
            fs=fs,
            max_flatline_s=max_flatline_s,
            max_gap_frac=max_gap_frac,
        )

        row_dict = dict(row)
        row_dict["quality_ok"] = quality_ok
        row_dict["exclusion_reason"] = reason
        updated_rows.append(row_dict)

        if quality_ok:
            exclusion_counts["passed"] += 1
            cleaned_tidy_df["window_id"] = w_id
            cleaned_window_dfs.append(cleaned_tidy_df)
        else:
            reason_key = reason if reason in exclusion_counts else "other"
            exclusion_counts[reason_key] = exclusion_counts.get(reason_key, 0) + 1

    # Save processed window signals
    if cleaned_window_dfs:
        combined_cleaned = pd.concat(cleaned_window_dfs, ignore_index=True)
        # Select and enforce schema columns: window_id, channel, t, value
        combined_cleaned = combined_cleaned[["window_id", "channel", "t", "value"]]
        validate_dataframe(combined_cleaned, WINDOW_SIGNALS_SCHEMA)
        out_path = processed_dir / f"{record_name}.parquet"
        combined_cleaned.to_parquet(out_path, index=False)
    else:
        # Create empty parquet conforming to schema
        empty_df = pd.DataFrame(columns=["window_id", "channel", "t", "value"])
        out_path = processed_dir / f"{record_name}.parquet"
        empty_df.to_parquet(out_path, index=False)

    df_updated_index = pd.DataFrame(updated_rows)
    return df_updated_index, exclusion_counts


def write_cleaning_rules_doc(
    summary_counts: dict[str, int],
    total_windows: int,
    output_path: Path,
) -> None:
    """Document cleaning rules and record exclusion statistics for manuscript.

    Args:
        summary_counts: Aggregate count of windows per exclusion reason.
        total_windows: Total number of evaluated windows.
        output_path: Target path for CLEANING_RULES.md.
    """
    passed = summary_counts.get("passed", 0)
    passed_pct = (passed / total_windows * 100) if total_windows > 0 else 0.0

    table_rows = [
        f"| **Valid / Quality OK** | **{passed}** | **{passed_pct:.2f}%** | "
        "**Included in training/evaluation** |"
    ]
    for reason, count in sorted(summary_counts.items()):
        if reason == "passed":
            continue
        pct = (count / total_windows * 100) if total_windows > 0 else 0.0
        table_rows.append(f"| Excluded: `{reason}` | {count} | {pct:.2f}% | Rejected |")

    table_rows.append(f"| **Total Evaluated** | **{total_windows}** | **100.00%** | — |")
    table_str = "\n".join(table_rows)

    doc_text = f"""# Signal Cleaning, Artifact Filtering, and Gap Handling Rules

This document details the exact mathematical rules, physiological cutoffs, and exclusion criteria
applied during Phase 2 signal preprocessing. This specification is formatted for direct inclusion
in the research manuscript.

---

## 1. Zero-Phase Butterworth Filtering

All physiological waveform channels undergo 4th-order zero-phase bandpass filtering implemented via
Second-Order Sections (`scipy.signal.sosfiltfilt`):

| Channel | Passband (Hz) | Filter Order | Rationale |
| :--- | :--- | :--- | :--- |
| **ECG Lead II** | 0.5 – 40.0 Hz | 4th Order | Baseline wander and 50/60 Hz noise elimination. |
| **PLETH (PPG)** | 0.5 – 8.0 Hz | 4th Order | Preserves dicrotic notch; eliminates motion noise. |
| **ABP** | 0.5 – 20.0 Hz | 4th Order | Retains pressure dynamics and dicrotic inflection. |
| **RESP** | 0.05 – 1.0 Hz | 4th Order | Captures true breathing; attenuates cardiac noise. |

> **Boundary & Run Handling**: Filtering is performed strictly on contiguous non-NaN segments exceeding
> $3 \\times \\text{{padlen}}$ ($3 \\times 24 = 72$ samples). Shorter fragments are converted to missing
> values (NaN) to prevent transient edge distortion.

---

## 2. Deterministic Gap Handling

- **Short Gaps ($\\le 2.0$ seconds)**: Linearly interpolated along the continuous time axis ($t$).
- **Long Gaps ($> 2.0$ seconds)**: Retained as missing values (NaN). Windows with total missing fraction
  $\\le 20\\%$ are preserved with `partial=True`.
- **Excessive Missingness ($> 20\\%$)**: If the cumulative gap fraction across any monitored channel
  exceeds $0.20$, the entire 5-minute window is excluded (`exclusion_reason = 'gap'`).

---

## 3. Physiological Artifact & Saturation Rules

- **Sensor Detachment / Flatlines**: Rolling 2-second standard deviation $\\sigma < 10^{{-6}}$ for
  continuous duration $> 10.0$ seconds triggers immediate exclusion (`'flatline'`).
- **Arterial Pressure Range Constraints**: Arterial blood pressure values outside the viable domain
  ($< 10\\text{{ mmHg}}$ or $> 300\\text{{ mmHg}}$) exceeding $10\\%$ of samples trigger exclusion
  (`'abp_range'`).
- **ECG Saturation / ADC Railing**: Lead II samples railed at ADC limits or exceeding physical amplifier
  thresholds ($|x| > 4.5\\text{{ mV}}$) for $> 5\\%$ of samples trigger exclusion (`'ecg_sat'`).

---

## 4. Per-Window Robust Normalization

Each valid window is standardized using robust z-score transformation:
$$\\tilde{{x}} = \\frac{{x - \\text{{median}}(x)}}{{\\max(\\text{{IQR}}(x), 10^{{-3}})}}$$
where $\\text{{IQR}} = Q_{{75}} - Q_{{25}}$.

---

## 5. Empirical Cleaning & Exclusion Statistics

The table below reports the empirical window retention and exclusion counts across the cohort:

| Classification / Exclusion Reason | Window Count | Percentage of Total | Action |
| :--- | :--- | :--- | :--- |
{table_str}

*(This summary is automatically regenerated by `scripts/11_clean_windows.py`)*
"""
    output_path.write_text(doc_text, encoding="utf-8")


def main() -> None:
    """Entry point for window cleaning and filtering script."""
    parser = argparse.ArgumentParser(description="Clean, filter, and normalize raw waveform windows.")
    parser.add_argument("--interim-dir", default="data/interim/windows")
    parser.add_argument("--processed-dir", default="data/processed/windows")
    parser.add_argument("--reports-dir", default="reports/eda")
    parser.add_argument("--docs-path", default="docs/CLEANING_RULES.md")
    args = parser.parse_args()

    cfg = load_config()
    max_flatline_s = float(cfg.data.windows.max_flatline_s)
    max_gap_frac = float(cfg.data.windows.max_gap_frac)

    interim_dir = Path(args.interim_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_index_path = interim_dir / "windows_index_raw.parquet"
    if not raw_index_path.exists():
        logger.error("Raw window index does not exist: {}", raw_index_path)
        return

    df_raw_index = pd.read_parquet(raw_index_path)
    logger.info("Found {} windows to clean in raw index.", len(df_raw_index))

    distinct_records = df_raw_index["record_name"].unique()
    n_workers = perf_cores()
    logger.info(
        "Processing {} records across {} workers (P-cores)...",
        len(distinct_records),
        n_workers,
    )

    results = Parallel(n_jobs=n_workers)(
        delayed(clean_single_record)(
            record_name=rec,
            record_windows_index=df_raw_index[df_raw_index["record_name"] == rec],
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            max_flatline_s=max_flatline_s,
            max_gap_frac=max_gap_frac,
        )
        for rec in distinct_records
    )

    all_indices = []
    total_exclusions: dict[str, int] = {}

    for df_idx, counts in results:
        all_indices.append(df_idx)
        for k, v in counts.items():
            total_exclusions[k] = total_exclusions.get(k, 0) + v

    df_final_index = pd.concat(all_indices, ignore_index=True)
    # Ensure columns match WINDOWS_INDEX_SCHEMA
    index_cols = [f.name for f in WINDOWS_INDEX_SCHEMA]
    df_final_index = df_final_index[index_cols]
    validate_dataframe(df_final_index, WINDOWS_INDEX_SCHEMA)

    final_index_path = processed_dir / "windows_index.parquet"
    df_final_index.to_parquet(final_index_path, index=False)
    logger.success("Saved final windows index to {}", final_index_path)

    # Save summary json
    summary_path = reports_dir / "cleaning_summary.json"
    summary_payload = {
        "total_windows": len(df_final_index),
        "counts": total_exclusions,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.success("Saved cleaning summary JSON to {}", summary_path)

    # Write CLEANING_RULES.md
    docs_path = Path(args.docs_path)
    write_cleaning_rules_doc(
        summary_counts=total_exclusions,
        total_windows=len(df_final_index),
        output_path=docs_path,
    )
    logger.success("Generated {}", docs_path)

    # Print exclusion table to terminal
    print("\n" + "=" * 65)
    print("           WINDOW CLEANING & EXCLUSION TABLE")
    print("=" * 65)
    print(f"{'Category':<25} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 65)
    tot = len(df_final_index)
    for cat, cnt in sorted(total_exclusions.items()):
        pct = (cnt / tot * 100) if tot > 0 else 0.0
        print(f"{cat:<25} | {cnt:<10} | {pct:>8.2f}%")
    print("-" * 65)
    print(f"{'TOTAL EVALUATED':<25} | {tot:<10} | {'100.00%':>10}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
