#!/usr/bin/env python3
"""
Analyze power consumption for the first 30,000 training steps.

This script correlates step data from subtask_finetuning_loss.csv with 
time-based power data from subtask_finetuning_power.csv to calculate
power statistics for only the first 30,000 steps.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import argparse

def parse_time_col(series: pd.Series) -> pd.Series:
    """Convert 'Relative Time (Process)' to seconds (float)."""
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float)
    # assume HH:MM:SS[.ms]
    return pd.to_timedelta(series).dt.total_seconds()

def effective_deltas(t: np.ndarray, gap_mult: float = 10.0):
    """Clamp large gaps so 'paused' periods are excluded."""
    dt = np.diff(t)
    med = np.median(dt)
    threshold = gap_mult * med
    # The first sample contributes no time; append median to keep lengths equal
    return np.minimum(dt, threshold), threshold

def correlate_steps_with_time(loss_df, power_df, max_steps=30000):
    """
    Correlate step data with time data to find the time range for first max_steps.
    
    Returns:
        tuple: (start_time, end_time) in seconds for the first max_steps
    """
    # Filter loss data to first max_steps
    loss_filtered = loss_df[loss_df['Step'] <= max_steps].copy()
    
    if len(loss_filtered) == 0:
        raise ValueError(f"No data found for steps up to {max_steps}")
    
    # Get the step range we're interested in
    min_step = loss_filtered['Step'].min()
    max_step = loss_filtered['Step'].max()
    
    print(f"Analyzing steps {min_step} to {max_step} (total: {len(loss_filtered)} data points)")
    
    # Since we don't have direct step-time correlation, we'll estimate based on:
    # 1. Assuming linear relationship between steps and time
    # 2. Using the total time range and scaling by step ratio
    
    # Parse power timestamps
    power_times = parse_time_col(power_df["Relative Time (Process)"]).values
    
    # Estimate time per step based on total time and total steps
    total_time = power_times[-1] - power_times[0]
    total_steps_in_loss = loss_df['Step'].max()
    time_per_step = total_time / total_steps_in_loss
    
    # Calculate time range for our target steps
    start_time = power_times[0] + (min_step * time_per_step)
    end_time = power_times[0] + (max_step * time_per_step)
    
    print(f"Estimated time range: {start_time:.1f}s to {end_time:.1f}s")
    print(f"Total estimated duration: {end_time - start_time:.1f}s")
    
    return start_time, end_time

def analyze_power_for_time_range(power_df, start_time, end_time, gap_mult=30.0):
    """
    Analyze power consumption for a specific time range.
    """
    # Parse timestamps
    power_times = parse_time_col(power_df["Relative Time (Process)"]).values
    
    # Filter power data to the time range
    time_mask = (power_times >= start_time) & (power_times <= end_time)
    power_filtered = power_df[time_mask].copy()
    
    if len(power_filtered) == 0:
        raise ValueError(f"No power data found in time range {start_time:.1f}s to {end_time:.1f}s")
    
    print(f"Found {len(power_filtered)} power measurements in target time range")
    
    # Parse timestamps for filtered data
    t = parse_time_col(power_filtered["Relative Time (Process)"]).values
    dt_eff, thresh = effective_deltas(t, gap_mult)
    dt_eff = np.insert(dt_eff, 0, 0.0)  # first sample has zero duration
    
    results = {}
    
    # Analyze each power column
    for col in power_filtered.columns:
        if col == "Relative Time (Process)" or col.endswith("__MIN") or col.endswith("__MAX"):
            continue
            
        values = power_filtered[col].values
        
        if "powerWatts" in col:
            # Calculate energy and power statistics
            energy_j = np.sum(values * dt_eff)  # watt-seconds (joules)
            active_time = dt_eff.sum()
            
            if active_time > 0:
                avg_power = energy_j / active_time
                results["gpu_power_mean_W"] = avg_power
                results["gpu_energy_kWh"] = energy_j / (3.6e6)
                results["gpu_power_max_W"] = np.max(values)
                results["gpu_power_min_W"] = np.min(values)
                results["active_time_s"] = active_time
            else:
                print(f"Warning: No active time calculated for {col}")
        else:
            # For other metrics, just calculate basic stats
            stats = {
                "mean": np.mean(values),
                "max": np.max(values),
                "min": np.min(values)
            }
            key = col.split("/")[-1].split(".")[0]
            results[f"{key}_mean"] = stats["mean"]
            results[f"{key}_max"] = stats["max"]
            results[f"{key}_min"] = stats["min"]
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze power consumption for first 30k steps")
    parser.add_argument("--loss-file", default="subtask_finetuning_loss.csv",
                       help="Path to loss CSV file")
    parser.add_argument("--power-file", default="subtask_finetuning_power.csv", 
                       help="Path to power CSV file")
    parser.add_argument("--max-steps", type=int, default=30000,
                       help="Maximum number of steps to analyze")
    parser.add_argument("--gap-mult", type=float, default=30.0,
                       help="Gap multiplier for detecting paused periods")
    parser.add_argument("--out", help="Optional CSV path to save summary")
    
    args = parser.parse_args()
    
    print("Loading data...")
    loss_df = pd.read_csv(args.loss_file)
    power_df = pd.read_csv(args.power_file)
    
    print(f"Loss data: {len(loss_df)} rows, steps {loss_df['Step'].min()} to {loss_df['Step'].max()}")
    print(f"Power data: {len(power_df)} rows")
    
    # Correlate steps with time
    print(f"\nCorrelating steps with time for first {args.max_steps} steps...")
    start_time, end_time = correlate_steps_with_time(loss_df, power_df, args.max_steps)
    
    # Analyze power for the time range
    print(f"\nAnalyzing power consumption...")
    results = analyze_power_for_time_range(power_df, start_time, end_time, args.gap_mult)
    
    # Display results
    print("\n" + "="*50)
    print(f"POWER ANALYSIS FOR FIRST {args.max_steps} STEPS")
    print("="*50)
    
    if "active_time_s" in results:
        active_time_td = timedelta(seconds=results["active_time_s"])
        print(f"Active training time: {active_time_td}")
    
    if "gpu_power_mean_W" in results:
        print(f"Average GPU power: {results['gpu_power_mean_W']:.1f} W")
        print(f"Max GPU power: {results['gpu_power_max_W']:.1f} W")
        print(f"Min GPU power: {results['gpu_power_min_W']:.1f} W")
        print(f"Total GPU energy: {results['gpu_energy_kWh']:.3f} kWh")
    
    # Show other metrics
    for key, value in results.items():
        if not key.startswith(("gpu_power", "gpu_energy", "active_time")):
            print(f"{key:<20}: {value:.2f}")
    
    # Save results if requested
    if args.out:
        pd.DataFrame([results]).to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    main()
