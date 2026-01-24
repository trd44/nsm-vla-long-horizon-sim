#!/usr/bin/env python3
"""
plot_task_progress.py
---------------------
Visualise how many runs have reached each sub-task.

*  y-axis – number of runs
*  x-axis – sub-tasks completed (0, 1, 2 …)

The script detects how many *rows* your file actually uses
(stops at the first completely-empty row).

Usage
-----
    python plot_task_progress.py data/vla_exports/score.csv \
                                 --out progress.png
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_matrix(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, header=0)
    # drop completely-empty rows at bottom if they exist
    df = df.dropna(how="all")
    return df.apply(pd.to_numeric, errors="coerce")

def cumulative_counts(mat: pd.DataFrame):
    """
    mat: rows = sub-task index, cols = runs, cells = sub-task id or NaN
    Return two equal-length lists:
        tasks_list  – [0, 1, 2, …]
        run_counts  – [count that reached ≥0, ≥1, ≥2 …]
    """
    counts = []
    for i, row in mat.iterrows():
        counts.append(int(row.notna().sum()))
    tasks = list(range(len(counts)))
    return tasks, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="score.csv exported from W&B")
    ap.add_argument("--out", default="task_progress.png",
                    help="output image file (default: task_progress.png)")
    args = ap.parse_args()

    mat = load_matrix(args.csv)
    tasks, counts = cumulative_counts(mat)

    plt.figure(figsize=(8, 5))
    plt.step(tasks, counts, where="post")
    plt.xlabel("Sub‑tasks completed")
    plt.ylabel("Number of runs")
    plt.title("Task progression across runs")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # --- custom tick spacing & limits ----------------------------------------
    import numpy as np
    plt.xlim(0, 14)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 14, 1))   # 0‑13 inclusive
    plt.yticks(np.arange(0, 101, 5))  # 0‑100 in steps of 5

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"✔ Plot saved to {args.out}")

if __name__ == "__main__":
    main()