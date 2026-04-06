"""
average_mtl_epochs.py

Takes a directory of CSV files, averages the dbmtl column for Steps 189-199
(inclusive), prints the average per metric (csv file), and saves results to a
new directory with '_average' appended to the input directory name.

Usage:
    python average_mtl_epochs.py <input_directory>
"""

import sys
import pandas as pd
from pathlib import Path


def average_epochs(input_dir: str):
    input_path = Path(input_dir).resolve()

    if not input_path.is_dir():
        print(f"Error: '{input_path}' is not a valid directory.")
        sys.exit(1)

    output_path = input_path.parent / (input_path.name + "_average")
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{input_path}'.")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV file(s) in '{input_path}'.")
    print(f"Output directory: '{output_path}'")
    print()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "Step" not in df.columns:
            print(f"  WARNING: No 'Step' column found in '{csv_file.name}'. Skipping.")
            continue

        subset = df[df["Step"].between(189, 199)]

        if subset.empty:
            print(f"  WARNING: No rows for Steps 189-199 in '{csv_file.name}'. Skipping.")
            continue

        # Keep only the core dbmtl column (exclude __MIN and __MAX variants)
        dbmtl_cols = [
            c for c in df.columns
            if "dbmtl" in c
            and not c.endswith("__MIN")
            and not c.endswith("__MAX")
        ]

        if not dbmtl_cols:
            print(f"  WARNING: No dbmtl column found in '{csv_file.name}'. Skipping.")
            continue

        averaged = subset[dbmtl_cols].mean(numeric_only=True).to_frame().T

        out_file = output_path / csv_file.name
        averaged.to_csv(out_file, index=False)

        # Print metric name (csv filename without extension) and the average value
        metric_name = csv_file.stem
        avg_value = averaged[dbmtl_cols[0]].iloc[0]
        print(f"  {metric_name}: {avg_value:.6f}")

    print()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python average_mtl_epochs.py <input_directory>")
        sys.exit(1)

    average_epochs(sys.argv[1])