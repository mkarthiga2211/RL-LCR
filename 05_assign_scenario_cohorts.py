"""Assign patient records and channel configurations to network scenarios.

Reads selected MIMIC-III records, groups records into simulation scenario pools
(ward, cohort, ambulatory), evaluates distinct patient counts against scale
requirements, writes data/interim/catalog/scenario_pools.json, and produces
docs/SCENARIOS.md documentation.
"""

import argparse
from pathlib import Path

import pandas as pd

from rllcr.config import load_config
from rllcr.data.splits import assign_scenarios
from rllcr.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Assign patient records to network scenarios and generate documentation."
    )
    parser.add_argument(
        "--records",
        type=str,
        default="data/interim/catalog/selected_records.csv",
        help="Path to selected_records.csv catalog.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/interim/catalog/scenario_pools.json",
        help="Output path for scenario_pools.json.",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="docs/SCENARIOS.md",
        help="Output markdown file summarizing scenarios.",
    )
    return parser.parse_args()


def generate_scenarios_markdown(
    scenario_pools: dict[str, any],
    docs_path: Path | str,
) -> None:
    """Generate Markdown documentation table for network scenarios.

    Args:
        scenario_pools: Dictionary returned by assign_scenarios.
        docs_path: Destination path for docs/SCENARIOS.md.
    """
    docs_file = Path(docs_path)
    docs_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for sc_name, sc_data in scenario_pools.items():
        node_counts = "/".join(str(n) for n in sc_data["nodes"])
        channels = ", ".join(sc_data["channels"])
        rate_str = sc_data["sampling_rate"]
        if "target_fs" in sc_data:
            fs_parts = [f"{k}={v}Hz" for k, v in sc_data["target_fs"].items()]
            rate_str += f" ({', '.join(fs_parts)})"

        total_pts = sc_data["total_patients"]
        total_recs = sc_data["total_records"]
        splits_str = (
            f"Train: {sc_data['splits']['train']['patient_count']} pts | "
            f"Val: {sc_data['splits']['val']['patient_count']} pts | "
            f"Test: {sc_data['splits']['test']['patient_count']} pts"
        )
        rows.append(
            f"| **{sc_name.capitalize()}** | {node_counts} | `{channels}` | {rate_str} | "
            f"{total_recs} records ({total_pts} patients) | {splits_str} |"
        )

    table_content = "\n".join(rows)

    md_text = f"""# Network Simulation Scenarios

This document specifies the network scenarios used in the RL-LCR evaluation suite.
Every simulated node receives a realistic physiological traffic profile drawn from a real patient record.

## Scenario Summary Table

| Scenario | Node Counts | Channels | Sampling Rate | Total Pool Size | Partitioned Patient Pools |
| :--- | :--- | :--- | :--- | :--- | :--- |
{table_content}

## Detailed Scenario Profiles

### 1. Ward Deployment (`ward`)
- **Node Configurations**: 20, 35, 50 nodes.
- **Hardware Architecture**: Bedside mains-assisted nodes (2000 mAh battery, 3.7 V, 90% mains-powered).
- **Signal Channels**: Full-rate ICU multi-lead signals (`[II, PLETH, ABP, RESP]`) at native rate (125 Hz).
- **Traffic Nature**: Continuous monitoring with acute clinical event triggers (Arrhythmia, Desat, Hypo).

### 2. Scalability Cohort (`cohort`)
- **Node Configurations**: 100, 150, 200 nodes.
- **Hardware Architecture**: Bedside mains-assisted nodes covering expanded multi-bed hospital wards.
- **Signal Channels**: Full-rate ICU multi-lead signals (`[II, PLETH, ABP, RESP]`) at native sampling rate.
- **Sampling Strategy**:
  - Full cohort evaluation with up to 200 nodes.
  - When distinct patient count in the test split is < 200, traffic is sampled with replacement
    across non-overlapping 5-minute patient signal windows without cross-patient leakage.

### 3. Ambulatory Monitoring (`ambulatory`)
- **Node Configurations**: 20, 35, 50 nodes.
- **Hardware Architecture**: Battery-only wearable sensors (300 mAh, 3.7 V, 0% mains, lifetime >= 72h).
- **Signal Channels**: Resampled low-power subset (`[II, PLETH]`).
- **Resampling Specifications**:
  - Lead II (ECG): Downsampled to **62.5 Hz**.
  - PLETH (SpO2): Downsampled to **25.0 Hz**.
- **Mobility**: Mobile nodes under Random Waypoint mobility within ward boundaries (0.5 - 1.5 m/s).

## Cohort-Scale Sampling Policy
- **Threshold**: Requires >= 200 distinct patients in the test split for sampling without replacement.
- **Fall-back Policy**: When fewer distinct patients are available, temporal windowing (sampling distinct
  5-minute epochs with replacement from the test pool) is activated and logged.
"""

    docs_file.write_text(md_text, encoding="utf-8")


def main() -> None:
    """Execute scenario cohort assignment pipeline."""
    setup_logging("05_assign_scenario_cohorts")
    args = parse_args()
    cfg = load_config()

    records_file = Path(args.records)
    if not records_file.exists():
        raise FileNotFoundError(
            f"Prerequisite catalog file not found: {records_file}. "
            f"Please verify data/interim/catalog/selected_records.csv exists."
        )

    records_df = pd.read_csv(records_file)
    print(f"Loaded {len(records_df)} selected records from {records_file}")

    scenario_pools = assign_scenarios(
        records_df=records_df,
        output_path=args.out,
        config=cfg,
    )

    generate_scenarios_markdown(
        scenario_pools=scenario_pools,
        docs_path=args.docs,
    )
    print(f"Generated scenario documentation at {args.docs}")

    # Print summary table to stdout
    print("\n==========================================================================")
    print("Scenario Pools Assignment Summary")
    print("==========================================================================")
    for sc, data in scenario_pools.items():
        print(f"Scenario: {sc.upper()}")
        print(f"  Description    : {data['description']}")
        print(f"  Nodes          : {data['nodes']}")
        print(f"  Channels       : {data['channels']}")
        print(f"  Rate           : {data['sampling_rate']}")
        print(f"  Eligible Recs  : {data['total_records']} (Patients: {data['total_patients']})")
        print(f"  Split Pools    : Train={data['splits']['train']['patient_count']} pts, "
              f"Val={data['splits']['val']['patient_count']} pts, "
              f"Test={data['splits']['test']['patient_count']} pts")
        print("--------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
