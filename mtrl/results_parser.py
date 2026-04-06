"""
average_mtrl_seeds.py

Reads eval.log files from MTRL experiments structured as:
  <parent_dir>/
    metaworld-mt10_<method>_state_sac_seed_1/eval.log
    metaworld-mt10_<method>_state_sac_seed_2/eval.log
    ...

For each seed, finds the best-checkpoint success (max 'success' across all
eval steps). Then reports mean ± SEM across seeds, and saves a summary CSV
to a sibling directory named <parent_dir>_mtrl_average/.

Usage:
    python average_mtrl_seeds.py <parent_directory>
"""

import sys
import json
import math
import pandas as pd
from pathlib import Path


def read_eval_log(filepath: Path) -> pd.DataFrame:
    """Read a JSONL eval.log file into a DataFrame."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(records)


def sem(values):
    """Standard error of the mean."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def process_parent_dir(parent_dir: str):
    parent_path = Path(parent_dir).resolve()

    if not parent_path.is_dir():
        print(f"Error: '{parent_path}' is not a valid directory.")
        sys.exit(1)

    output_path = parent_path.parent / (parent_path.name + "_mtrl_average")
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all seed subdirectories containing eval.log
    seed_dirs = sorted([
        d for d in parent_path.iterdir()
        if d.is_dir() and (d / "eval.log").exists()
    ])

    if not seed_dirs:
        print(f"No subdirectories with eval.log found in '{parent_path}'.")
        sys.exit(1)

    print(f"Found {len(seed_dirs)} seed(s) in '{parent_path}'.")
    print()

    per_seed_results = []

    for seed_dir in seed_dirs:
        eval_log = seed_dir / "eval.log"
        df = read_eval_log(eval_log)

        if "success" not in df.columns:
            print(f"  WARNING: No 'success' column in '{eval_log}'. Skipping.")
            continue

        # Filter to eval rows only (in case train rows are mixed in)
        if "mode" in df.columns:
            df = df[df["mode"] == "eval"]

        best_idx = df["success"].idxmax()
        best_row = df.loc[best_idx]
        best_success = best_row["success"]
        best_step = best_row.get("step", "?")
        best_episode = best_row.get("episode", "?")

        print(f"  {seed_dir.name}")
        print(f"    Best success: {best_success:.4f}  (step={best_step}, episode={best_episode})")

        per_seed_results.append({
            "seed_dir": seed_dir.name,
            "best_success": best_success,
            "best_step": best_step,
            "best_episode": best_episode,
        })

    if not per_seed_results:
        print("No valid seeds found.")
        sys.exit(1)

    successes = [r["best_success"] for r in per_seed_results]
    mean_success = sum(successes) / len(successes)
    sem_success = sem(successes)

    print()
    print(f"  {'─' * 50}")
    print(f"  Seeds:           {len(successes)}")
    print(f"  Mean success:    {mean_success:.4f}")
    print(f"  SEM:             {sem_success:.4f}")
    print(f"  Result:          {mean_success:.3f} ± {sem_success:.3f}")
    print()

    # Save per-seed results
    results_df = pd.DataFrame(per_seed_results)
    results_df.loc[len(results_df)] = {
        "seed_dir": "MEAN ± SEM",
        "best_success": mean_success,
        "best_step": "",
        "best_episode": f"± {sem_success:.4f}",
    }

    out_file = output_path / "best_checkpoint_success.csv"
    results_df.to_csv(out_file, index=False)
    print(f"  Saved summary to '{out_file}'")
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python average_mtrl_seeds.py <parent_directory>")
        sys.exit(1)

    process_parent_dir(sys.argv[1])