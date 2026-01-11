#!/usr/bin/env python3
"""
Download W&B system metrics (Events Stream) to match UI Chart Exports.

FINAL WORKING VERSION:
  1. Auto-corrects key names (handles "system/" vs "system." mismatch).
  2. Fetches high-density 'events' stream (run.history(stream="events")).
  3. Resamples to 15s intervals to generate Mean/Min/Max columns.
  4. Exports exact UI-style CSVs.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

try:
    import pandas as pd
    import wandb
    import tyro
except ImportError:
    print("❌ Missing dependencies. Run: pip install pandas wandb tyro")
    sys.exit(1)


@dataclasses.dataclass
class Args:
    project: str = "openpi"
    run_name: str = "hanoi_50_ja_no_mask_50ah"
    metrics: str = "system/gpu.0.powerWatts"
    out_dir: Path = Path(f"data/finetuning_exports/{run_name}/")
    interval: str = "15s"
    debug: bool = True


def get_system_event_series(run, metric_key: str, interval: str) -> pd.DataFrame | None:
    """
    Fetches the full 'events' stream, handles key mismatch, and resamples.
    """
    # 1. Fetch ALL events
    try:
        # Request 100k samples to ensure we get the full dense stream
        df = run.history(stream="events", samples=100000)
    except Exception as e:
        print(f"  ❌ Error fetching events: {e}")
        return None

    if df.empty:
        return None
        
    # 2. AUTO-CORRECT KEY (The Fix)
    target_key = metric_key
    if target_key not in df.columns:
        # Try replacing / with .
        alt_key = metric_key.replace("/", ".")
        if alt_key in df.columns:
            print(f"  ℹ️  Mapping '{metric_key}' -> '{alt_key}'")
            target_key = alt_key
        else:
            # Try replacing . with /
            alt_key_2 = metric_key.replace(".", "/")
            if alt_key_2 in df.columns:
                print(f"  ℹ️  Mapping '{metric_key}' -> '{alt_key_2}'")
                target_key = alt_key_2

    if target_key not in df.columns:
        if args.debug:
            print(f"  ⚠️  Metric '{metric_key}' (or variants) not found.")
        return None

    # 3. Handle Time (Use _runtime for Relative Time)
    if "_runtime" in df.columns:
        df = df.sort_values("_runtime")
        time_index = pd.to_timedelta(df["_runtime"], unit="s")
    elif "_timestamp" in df.columns:
        df = df.sort_values("_timestamp")
        start_time = df["_timestamp"].min()
        rel_time = df["_timestamp"] - start_time
        time_index = pd.to_timedelta(rel_time, unit="s")
    else:
        print("  ❌ Error: No time column found.")
        return None

    df.index = time_index
    
    # 4. Clean Data
    series = df[target_key].dropna()
    series = pd.to_numeric(series, errors='coerce').dropna()

    if series.empty:
        return None

    # 5. Resample and Aggregate (Mean, Min, Max)
    # This exactly matches the W&B UI Export format
    resampled = series.resample(interval).agg(['mean', 'min', 'max'])
    resampled = resampled.ffill()

    resampled.index.name = "Relative Time (Process)"
    return resampled


def download_wandb_history(
    project: str,
    metric_keys: list[str],
    out_dir: Path,
    run_name_filter: str,
    interval: str,
    debug: bool,
) -> None:
    
    api = wandb.Api()
    runs = api.runs(project)
    
    if run_name_filter:
        runs = [r for r in runs if run_name_filter in (r.name, r.id)]
    
    if not runs:
        print(f"❌ No runs found matching '{run_name_filter}' in '{project}'")
        return

    print(f"Found {len(runs)} runs. Processing metrics: {metric_keys}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metric_keys:
        print(f"\n--- Processing Metric: {metric} ---")
        all_run_dfs = []

        for run in runs:
            r_name = run.name or run.id
            if debug:
                print(f"Scanning run: {r_name}...")
            
            stats_df = get_system_event_series(run, metric, interval)
            
            if stats_df is not None:
                # Use the original requested name for columns to keep CSV pretty
                # e.g., "RunName - system/gpu.0.powerWatts"
                base_col = f"{r_name} - {metric}"
                stats_df.columns = [base_col, f"{base_col}__MIN", f"{base_col}__MAX"]
                all_run_dfs.append(stats_df)
                print(f"  ✔️  Captured {len(stats_df)} intervals")
            else:
                if debug:
                    print(f"  -> No data.")

        if not all_run_dfs:
            print("  ⚠️ No data captured for this metric.")
            continue

        # Merge and align
        merged_df = pd.concat(all_run_dfs, axis=1)
        
        # Convert Timedelta Index to Seconds (Float) for CSV
        merged_df.index = merged_df.index.total_seconds()
        merged_df.index.name = "Relative Time (Process)"
        merged_df = merged_df.sort_index()

        safe_name = metric.replace("/", "_") + ".csv"
        safe_name = "gpu_power.csv"
        out_path = out_dir / safe_name
        merged_df.to_csv(out_path)
        print(f"✔️  Saved {safe_name} | Shape: {merged_df.shape}")


# Need to make args available to helper function, simple way:
args = None 

def main() -> None:
    global args
    args = tyro.cli(Args)
    metric_keys = [m.strip() for m in args.metrics.split(",")]
    
    download_wandb_history(
        project=args.project,
        metric_keys=metric_keys,
        out_dir=args.out_dir,
        run_name_filter=args.run_name,
        interval=args.interval,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()