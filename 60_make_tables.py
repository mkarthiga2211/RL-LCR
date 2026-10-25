"""Script 60: Generate all manuscript tables from Phase 5 results.

Usage:
    python scripts/60_make_tables.py [--results-dir results] [--output-dir reports/tables]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rllcr.eval.tables import export_all_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript evaluation tables directly from results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results parquets.")
    parser.add_argument("--output-dir", type=str, default="reports/tables", help="Output directory for generated tables.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print(f"Reading experimental results from: {results_dir.resolve()}")
    print(f"Generating manuscript tables into: {output_dir.resolve()}")

    tables = export_all_tables(results_dir, output_dir)

    # Verify all expected files exist
    expected_files = [
        output_dir / "table1.tex",
        output_dir / "table1.csv",
        output_dir / "table2.tex",
        output_dir / "table2.csv",
        output_dir / "table3.tex",
        output_dir / "table3.csv",
        output_dir / "table4.tex",
        output_dir / "table4.csv",
        output_dir / "table5.tex",
        output_dir / "table5.csv",
        output_dir / "all_tables.md",
        output_dir / "test_tables.tex",
    ]

    all_exist = all(f.exists() for f in expected_files)
    if not all_exist:
        missing = [str(f) for f in expected_files if not f.exists()]
        print(f"Error: Missing generated files: {missing}", file=sys.stderr)
        sys.exit(1)

    print("\nSuccessfully generated all manuscript tables:")
    for f in expected_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    # LaTeX Compilation Check
    latexmk_path = shutil.which("latexmk")
    pdflatex_path = shutil.which("pdflatex")

    if latexmk_path:
        print("\nCompiling test_tables.tex with latexmk...")
        res = subprocess.run([latexmk_path, "-pdf", "test_tables.tex"], cwd=output_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print("LaTeX compilation SUCCESSFUL: reports/tables/test_tables.pdf created.")
        else:
            print("latexmk warning/failure:\n", res.stderr[-500:])
    elif pdflatex_path:
        print("\nCompiling test_tables.tex with pdflatex...")
        res = subprocess.run([pdflatex_path, "-interaction=nonstopmode", "test_tables.tex"], cwd=output_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print("LaTeX compilation SUCCESSFUL: reports/tables/test_tables.pdf created.")
        else:
            print("pdflatex warning/failure:\n", res.stderr[-500:])
    else:
        print("\n[NOTE] No local LaTeX engine found (neither 'latexmk' nor 'pdflatex' installed in PATH).")
        print("To compile these tables into a publication-ready PDF:")
        print("1. In Overleaf (Recommended):")
        print("   - Create a new project in Overleaf.")
        print(f"   - Upload the files from '{output_dir}/':")
        print("     test_tables.tex, table1.tex, table2.tex, table3.tex, table4.tex, table5.tex")
        print("   - Set main document to 'test_tables.tex' and click Recompile.")
        print("2. In local MacTeX / TeX Live:")
        print(f"   cd \"{output_dir.resolve()}\" && pdflatex test_tables.tex")


if __name__ == "__main__":
    main()
