"""Script 10: Slice raw MIMIC-III waveforms into interim 5-minute windows and numerics streams.

Outputs:
    - data/interim/windows/windows_index_raw.parquet
    - data/interim/windows/<record_name>.parquet
    - data/interim/windows/<record_name>_numerics.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from rllcr.config import load_config
from rllcr.data.waveforms import load_wfdb_numerics, load_wfdb_record, slice_record_windows
from rllcr.utils.hw import perf_cores


def process_single_record(
    row: pd.Series,
    raw_dir: Path,
    interim_dir: Path,
    window_len_s: float,
    max_gap_frac: float,
    max_flatline_s: float,
) -> pd.DataFrame | None:
    """Process a single patient record: slice windows and extract numerics.

    Args:
        row: Record metadata row from selected_records.csv.
        raw_dir: Base directory containing raw records.
        interim_dir: Output directory for interim windows and numerics.
        window_len_s: Length of each window in seconds.
        max_gap_frac: Maximum tolerable gap fraction.
        max_flatline_s: Maximum tolerable flatline duration in seconds.

    Returns:
        pd.DataFrame | None: Slice metadata for the record, or None if failed.
    """
    rec_name = str(row["record_name"])
    patient_dir = str(row["patient_dir"])
    subject_id = int(str(row["patient_id"]).replace("p", ""))

    rec_path = raw_dir / patient_dir / rec_name
    logger.info("Processing record: {}", rec_name)

    try:
        p_signal, sig_names, fs, _ = load_wfdb_record(rec_path)
    except Exception as e:
        logger.error("Failed to load WFDB record {}: {}", rec_name, e)
        return None

    if p_signal.shape[0] == 0:
        logger.warning("Record {} has 0 samples, skipping.", rec_name)
        return None

    # Slice windows
    df_index, window_data_dict = slice_record_windows(
        p_signal=p_signal,
        sig_names=sig_names,
        fs=fs,
        record_name=rec_name,
        subject_id=subject_id,
        window_len_s=window_len_s,
        max_gap_frac=max_gap_frac,
        max_flatline_s=max_flatline_s,
    )

    if not window_data_dict:
        logger.warning("No windows produced for record {}", rec_name)
        return None

    # Combine all windows for this record into a single interim parquet
    record_dfs = []
    for w_id, w_df in window_data_dict.items():
        w_df["window_id"] = w_id
        record_dfs.append(w_df)

    combined_df = pd.concat(record_dfs, ignore_index=True)
    out_record_path = interim_dir / f"{rec_name}.parquet"
    combined_df.to_parquet(out_record_path, index=False)

    # Extract and save periodic numerics (1 Hz)
    num_path = raw_dir / patient_dir / f"{rec_name}n"
    df_numerics = load_wfdb_numerics(num_path)
    df_numerics["record_name"] = rec_name
    df_numerics["subject_id"] = subject_id
    out_num_path = interim_dir / f"{rec_name}_numerics.parquet"
    df_numerics.to_parquet(out_num_path, index=False)

    logger.success(
        "Record {}: {} windows sliced, numerics saved ({} rows)",
        rec_name,
        len(df_index),
        len(df_numerics),
    )
    return df_index


def main() -> None:
    """Entry point for raw waveform slicing script."""
    parser = argparse.ArgumentParser(description="Slice raw waveforms into 5-minute interim windows.")
    parser.add_argument("--catalog", default="data/interim/catalog/selected_records.csv")
    parser.add_argument("--raw-dir", default="data/raw/mimic3wdb_matched")
    parser.add_argument("--out-dir", default="data/interim/windows")
    args = parser.parse_args()

    cfg = load_config()
    window_len_s = cfg.data.windows.length_min * 60.0
    max_gap_frac = cfg.data.windows.max_gap_frac
    max_flatline_s = float(cfg.data.windows.max_flatline_s)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        logger.error("Selected records catalog not found: {}", catalog_path)
        return

    df_catalog = pd.read_csv(catalog_path)
    logger.info("Found {} records in catalog to slice.", len(df_catalog))

    n_workers = perf_cores()
    logger.info("Running parallel window slicing with {} workers (P-cores)", n_workers)

    results = Parallel(n_jobs=n_workers)(
        delayed(process_single_record)(
            row=row,
            raw_dir=raw_dir,
            interim_dir=out_dir,
            window_len_s=window_len_s,
            max_gap_frac=max_gap_frac,
            max_flatline_s=max_flatline_s,
        )
        for _, row in df_catalog.iterrows()
    )

    valid_indices = [r for r in results if r is not None and not r.empty]
    if not valid_indices:
        logger.error("No valid windows generated from any records.")
        return

    df_all_index = pd.concat(valid_indices, ignore_index=True)
    raw_index_path = out_dir / "windows_index_raw.parquet"
    df_all_index.to_parquet(raw_index_path, index=False)

    logger.success(
        "Successfully sliced {} total windows across {} records. Saved to {}",
        len(df_all_index),
        len(valid_indices),
        raw_index_path,
    )


if __name__ == "__main__":
    main()
