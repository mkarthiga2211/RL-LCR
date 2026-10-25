"""Download and prepare secondary zero-shot evaluation datasets.

Coordinates ingestion and cataloging for BIDMC, PPG-DaLiA, WESAD (optional),
and eICU. These datasets are strictly held out for zero-shot testing and
must NEVER be used during model training or hyperparameter tuning.
"""

import argparse
import os
import subprocess

from rllcr.config import load_config
from rllcr.data.secondary import (
    catalog_bidmc,
    catalog_ppg_dalia,
    catalog_wesad,
    check_dataset_status,
    download_bidmc,
    generate_manifest,
    process_eicu,
)
from rllcr.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI options for secondary dataset preparation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download and catalog secondary zero-shot datasets."
    )
    parser.add_argument(
        "--include-wesad",
        action="store_true",
        default=True,
        help="Include optional WESAD wearable stress dataset.",
    )
    parser.add_argument(
        "--eicu-stays",
        type=int,
        default=3000,
        help="Maximum ICU stays to sample from eICU (>= 6h vitals).",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip computing SHA256 file manifests if already present.",
    )
    return parser.parse_args()


def print_instructions_and_wget() -> None:
    """Print download instructions and credential commands for user reference."""
    print("\n--- Credentialed / License Dataset Download References ---")
    print("1. PPG-DaLiA (UCI License):")
    print("   Place PPG_FieldStudy.zip into data/raw/ppg_dalia/ and unzip, OR use Kaggle CLI:")
    print("   kaggle datasets download -d ameersifat53/ppg-dalia-dataset -p data/raw/ppg_dalia --unzip")
    print("\n2. eICU Collaborative Research Database (v2.0, PhysioNet Credentialed):")
    print("   wget -r -N -c -np --user <physionet_username> --ask-password \\")
    print("     https://physionet.org/files/eicu-crd/2.0/vitalPeriodic.csv.gz")
    print("   wget -r -N -c -np --user <physionet_username> --ask-password \\")
    print("     https://physionet.org/files/eicu-crd/2.0/vitalAperiodic.csv.gz")
    print("   wget -r -N -c -np --user <physionet_username> --ask-password \\")
    print("     https://physionet.org/files/eicu-crd/2.0/patient.csv.gz")
    print("----------------------------------------------------------\n")


def main() -> None:
    """Run end-to-end secondary dataset preparation pipeline."""
    setup_logging("06_download_secondary")
    args = parse_args()
    cfg = load_config()

    raw_root = cfg.paths.raw
    catalog_dir = cfg.paths.interim / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    clinical_pq_dir = cfg.paths.interim / "clinical_parquet"
    clinical_pq_dir.mkdir(parents=True, exist_ok=True)

    print_instructions_and_wget()

    statuses: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # 1. BIDMC Dataset
    # -------------------------------------------------------------------------
    bidmc_dir = raw_root / "bidmc"
    print("Processing BIDMC...")
    try:
        download_bidmc(dl_dir=bidmc_dir)
        catalog_bidmc(bidmc_dir=bidmc_dir, out_path=catalog_dir / "bidmc_catalog.parquet")
        if not args.skip_manifests and not (bidmc_dir / "MANIFEST.csv").exists():
            generate_manifest(bidmc_dir)
        statuses["BIDMC"] = check_dataset_status("bidmc", bidmc_dir)
    except Exception as exc:
        print(f"Error processing BIDMC: {exc}")
        statuses["BIDMC"] = "MISSING"

    # -------------------------------------------------------------------------
    # 2. PPG-DaLiA Dataset
    # -------------------------------------------------------------------------
    dalia_dir = raw_root / "ppg_dalia"
    print("Processing PPG-DaLiA...")
    status_dalia = check_dataset_status("ppg_dalia", dalia_dir)
    if status_dalia == "MISSING":
        # Check if Kaggle credentials are set to attempt download
        k_user = os.environ.get("KAGGLE_USERNAME")
        k_key = os.environ.get("KAGGLE_KEY")
        if k_user and k_key:
            print("Attempting Kaggle CLI download for PPG-DaLiA...")
            try:
                subprocess.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        "ameersifat53/ppg-dalia-dataset",
                        "-p",
                        str(dalia_dir),
                        "--unzip",
                    ],
                    check=True,
                )
            except Exception as e:
                print(f"Kaggle download failed: {e}")

    try:
        catalog_ppg_dalia(dalia_dir=dalia_dir, out_path=catalog_dir / "ppg_dalia_catalog.parquet")
        if not args.skip_manifests and not (dalia_dir / "MANIFEST.csv").exists():
            generate_manifest(dalia_dir)
        statuses["PPG-DaLiA"] = check_dataset_status("ppg_dalia", dalia_dir)
    except Exception as exc:
        print(f"Error processing PPG-DaLiA: {exc}")
        statuses["PPG-DaLiA"] = check_dataset_status("ppg_dalia", dalia_dir)

    # -------------------------------------------------------------------------
    # 3. WESAD Dataset (Optional)
    # -------------------------------------------------------------------------
    if args.include_wesad:
        wesad_dir = raw_root / "wesad"
        print("Processing WESAD...")
        try:
            catalog_wesad(wesad_dir=wesad_dir, out_path=catalog_dir / "wesad_catalog.parquet")
            if not args.skip_manifests and not (wesad_dir / "MANIFEST.csv").exists():
                generate_manifest(wesad_dir)
            statuses["WESAD"] = check_dataset_status("wesad", wesad_dir)
        except Exception as exc:
            print(f"Error processing WESAD: {exc}")
            statuses["WESAD"] = check_dataset_status("wesad", wesad_dir)

    # -------------------------------------------------------------------------
    # 4. eICU Dataset
    # -------------------------------------------------------------------------
    eicu_dir = raw_root / "eicu"
    print("Processing eICU...")
    status_eicu = check_dataset_status("eicu", eicu_dir)
    if status_eicu == "READY":
        try:
            process_eicu(
                eicu_dir=eicu_dir,
                out_dir=clinical_pq_dir,
                n_stays=args.eicu_stays,
                seed=cfg.experiments.base_seed,
            )
            if not args.skip_manifests and not (eicu_dir / "MANIFEST.csv").exists():
                generate_manifest(eicu_dir)
            statuses["eICU"] = "READY"
        except Exception as exc:
            print(f"Error processing eICU Parquet extraction: {exc}")
            statuses["eICU"] = "READY (Processing Error)"
    else:
        statuses["eICU"] = status_eicu

    # -------------------------------------------------------------------------
    # Summary Status Report
    # -------------------------------------------------------------------------
    print("\n==========================================================================")
    print("Secondary Zero-Shot Datasets Status Report")
    print("==========================================================================")
    for name, st in statuses.items():
        print(f"  {name:15s}: {st}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
