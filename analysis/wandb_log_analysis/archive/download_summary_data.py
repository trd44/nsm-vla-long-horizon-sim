#!/usr/bin/env python3
"""
Download W&B summary data (not time-series) from all runs.

python download_summary_data.py \
    "tim-duggan93-tufts-university/TRUE_FINAL_pi0_hanoi_300_subtasks_3_blocks_random_selection" \
    data/vla_exports/TRUE_FINAL_pi0_hanoi_300_subtasks_3_blocks_random_selection
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def download_wandb_summaries(project, out_dir):
    """Download summary data from all W&B runs."""
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

    # Collect all unique summary keys
    all_keys = set()
    run_data = {}
    
    print("Scanning runs for summary data...")
    for run in runs:
        run_name = run.name or run.id
        run_data[run_name] = {}
        
        try:
            summary = run.summary
            for key, value in summary.items():
                # Skip internal wandb keys
                if not key.startswith('_wandb'):
                    all_keys.add(key)
                    run_data[run_name][key] = value
        except Exception as e:
            print(f"Warning: Could not get summary for run {run_name}: {e}")

    all_keys = sorted(all_keys)
    print(f"Found {len(all_keys)} summary metrics:")
    for key in all_keys:
        print(f"  - {key}")

    # Save each metric to CSV
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one CSV with all summary data
    filename = out_dir / "all_summary_data.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['run_name'] + all_keys)
        
        # Write data rows
        for run_name, data in run_data.items():
            row = [run_name]
            for key in all_keys:
                value = data.get(key, '')
                row.append(value)
            writer.writerow(row)
    
    print(f"✔️  Saved {filename} (runs={len(run_data)}, metrics={len(all_keys)})")
    
    # Also save individual metric files
    for key in all_keys:
        filename = out_dir / (key.replace("/", "_") + ".csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_name', key])
            for run_name, data in run_data.items():
                value = data.get(key, '')
                writer.writerow([run_name, value])
        print(f"✔️  Saved {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_summary_data.py <project> [out_dir]")
        print("Example: python download_summary_data.py entity/project data/")
        sys.exit(1)
    
    project = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    download_wandb_summaries(project, out_dir)

if __name__ == "__main__":
    main()
