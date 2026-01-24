
#!/usr/bin/env python3
"""
download_wandb_data.py
----------------------
Download **all runs** from a Weights & Biases project and save one **wide‑format**
CSV per metric key you request.

Each CSV has:
    • rows → logged timesteps
    • cols → run names / ids

Example
-------
    python download_wandb_data.py \
        --project tim-duggan93-tufts-university/TRUE_FINAL_pi0_hanoi_300_one_task_3_blocks_random_selection \
        --metrics "*" \
        --out-dir data/vla_exports/One_Task_3_Blocks_Random_Selection

Arguments
---------
--project      entity/project  (mandatory)
--metrics      Comma‑separated list of metric keys,
               or '*' to grab every key present in any run.
--out-dir      Output directory (default: current working dir)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------------- #
def download_wandb_csvs(project: str,
                        metric_keys: List[str],
                        out_dir: Path) -> None:
    """Stream `metric_keys` for every run and write one wide CSV per metric."""
    try:
        import wandb
    except ModuleNotFoundError:
        raise SystemExit("❌  `wandb` not installed.  Run `pip install wandb` and retry.")

    api = wandb.Api()
    runs = api.runs(project)
    if not runs:
        raise SystemExit(f"❌  No runs found for project '{project}'")

    # Wildcard "*" → collect every key that appears in the runs
    if metric_keys == ["*"]:
        metric_set = set()
        for run in tqdm(runs, desc="Scanning runs", unit="run"):
            # scan_history(keys=[]) gives dicts of *all* keys
            for row in run.scan_history(keys=[]):
                metric_set.update(row.keys())
        metric_keys = sorted(metric_set)
        print(f"Found {len(metric_keys)} distinct metric keys")

    # initialise one empty wide DF per metric
    dfs: Dict[str, pd.DataFrame] = {m: pd.DataFrame() for m in metric_keys}

    print(f"Downloading {len(runs)} runs × {len(metric_keys)} metrics …")
    for run in tqdm(runs, desc="Runs", unit="run"):
        run_name = run.name or run.id
        # scan_history can omit the "_step" key if the metric wasn't logged
        # in that particular row.  We fall back to a monotonically increasing
        # index (`idx`) to avoid KeyError issues.
        # enumerate rows so we always have a fallback row‑index
        for idx, row in enumerate(run.scan_history(keys=metric_keys)):
            step = row.get("_step")
            if step is None:
                step = idx  # graceful fallback when _step key is absent
            for m in metric_keys:
                if m not in row:
                    continue
                dfs[m].loc[step, run_name] = row[m]  # align on row‑index

    # sort index (step) numerically
    for df in dfs.values():
        df.sort_index(inplace=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for m, df in dfs.items():
        if df.empty:
            print(f"⚠️  No data for key '{m}', skipping.")
            continue
        fn = out_dir / (m.replace("/", "_") + ".csv")
        df.to_csv(fn, index=False)
        print(f"✔️  Saved {fn}  (rows={df.shape[0]}, runs={df.shape[1]})")


# ----------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="Project path 'entity/project'")
    ap.add_argument("--metrics", required=True,
                    help="Comma‑separated metric keys, or '*' for all")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="Directory to save CSVs")
    args = ap.parse_args()

    metric_keys = [m.strip() for m in args.metrics.split(",")]
    download_wandb_csvs(args.project, metric_keys, args.out_dir)


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
