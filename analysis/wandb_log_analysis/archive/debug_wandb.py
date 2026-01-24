#!/usr/bin/env python3
"""
Debug script to see what's actually in the W&B runs
"""

import wandb

def debug_wandb_runs(project):
    api = wandb.Api()
    runs = api.runs(project)
    
    print(f"Found {len(runs)} runs in project '{project}'")
    
    for i, run in enumerate(runs[:3]):  # Check first 3 runs
        print(f"\n--- Run {i+1}: {run.name or run.id} ---")
        print(f"Run ID: {run.id}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        
        # Try to get summary
        try:
            summary = run.summary
            print(f"Summary keys: {list(summary.keys())}")
        except Exception as e:
            print(f"Error getting summary: {e}")
        
        # Try to get config
        try:
            config = run.config
            print(f"Config keys: {list(config.keys())}")
        except Exception as e:
            print(f"Error getting config: {e}")
        
        # Try to scan history
        try:
            history = list(run.scan_history(keys=[]))
            print(f"History rows: {len(history)}")
            if history:
                print(f"First row keys: {list(history[0].keys())}")
                print(f"Sample first row: {history[0]}")
        except Exception as e:
            print(f"Error scanning history: {e}")

if __name__ == "__main__":
    project = "tim-duggan93-tufts-university/FINAL_pi0_hanoi_300_one_task_3_blocks_random_selection"
    debug_wandb_runs(project)
