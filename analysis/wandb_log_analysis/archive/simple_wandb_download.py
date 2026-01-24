#!/usr/bin/env python3
"""
Simple W&B data downloader that avoids pandas/NumPy compatibility issues.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def download_wandb_data(project, metric_keys, out_dir):
    """Download W&B data using only basic Python libraries."""
    try:
        import wandb
    except ModuleNotFoundError:
        print("❌ `wandb` not installed. Run `pip install wandb` and retry.")
        sys.exit(1)

    api = wandb.Api()
    runs = api.runs(project)
    if not runs:
        print(f"❌ No runs found for project '{project}'")
        sys.exit(1)

    print(f"Found {len(runs)} runs in project '{project}'")

    # If wildcard, discover all available metrics
    if metric_keys == ["*"]:
        print("Scanning runs to discover available metrics...")
        metric_set = set()
        for run in runs:
            try:
                for row in run.scan_history(keys=[]):
                    metric_set.update(row.keys())
            except Exception as e:
                print(f"Warning: Could not scan run {run.name or run.id}: {e}")
        metric_keys = sorted(metric_set)
        print(f"Found {len(metric_keys)} distinct metric keys:")
        for key in metric_keys:
            print(f"  - {key}")

    # Collect data for each metric
    metric_data = defaultdict(lambda: defaultdict(dict))  # metric -> run -> step -> value
    
    print(f"Downloading data for {len(metric_keys)} metrics...")
    for run in runs:
        run_name = run.name or run.id
        print(f"Processing run: {run_name}")
        
        try:
            for idx, row in enumerate(run.scan_history(keys=metric_keys)):
                step = row.get("_step", idx)
                for metric in metric_keys:
                    if metric in row:
                        metric_data[metric][run_name][step] = row[metric]
        except Exception as e:
            print(f"Warning: Could not process run {run_name}: {e}")

    # Save each metric to CSV
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for metric, run_data in metric_data.items():
        if not run_data:
            print(f"⚠️  No data for key '{metric}', skipping.")
            continue
            
        # Get all unique steps across all runs
        all_steps = set()
        for run_steps in run_data.values():
            all_steps.update(run_steps.keys())
        all_steps = sorted(all_steps)
        
        # Get all run names
        run_names = sorted(run_data.keys())
        
        # Create CSV
        filename = out_dir / (metric.replace("/", "_") + ".csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Step'] + run_names)
            
            # Write data rows
            for step in all_steps:
                row = [step]
                for run_name in run_names:
                    value = run_data[run_name].get(step, '')
                    row.append(value)
                writer.writerow(row)
        
        print(f"✔️  Saved {filename} (rows={len(all_steps)}, runs={len(run_names)})")

def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_wandb_download.py <project> <metrics> [out_dir]")
        print("Example: python simple_wandb_download.py entity/project 'metric1,metric2' data/")
        print("Use '*' for all metrics")
        sys.exit(1)
    
    project = sys.argv[1]
    metrics = [m.strip() for m in sys.argv[2].split(",")]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
    
    download_wandb_data(project, metrics, out_dir)

if __name__ == "__main__":
    main()
