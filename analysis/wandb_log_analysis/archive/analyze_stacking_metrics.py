#!/usr/bin/env python3
"""
Analyze stacking metrics from W&B data.
Calculates mean stacking duration and mean stacking length steps for successful stackings only.

A stacking is defined as an odd subtask + even subtask pair (1&2, 3&4, 5&6, etc.).
Only stackings where both subtasks were completed successfully are counted.

python analyze_stacking_metrics.py \
    data/vla_exports/TRUE_FINAL_pi0_hanoi_300_subtasks_3_blocks_random_selection \
    SUBTASK_3_Blocks_stacking_analysis.csv
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def load_wandb_data(data_dir):
    """Load all W&B summary data."""
    data_dir = Path(data_dir)
    
    # Load the main summary data
    summary_file = data_dir / "all_summary_data.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    runs = []
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    
    print(f"Loaded {len(runs)} runs from {summary_file}")
    return runs

def calculate_stacking_metrics(runs):
    """Calculate stacking duration and step metrics for successful stackings only."""
    results = {}
    
    # Track successful stackings
    successful_stacking_durations = []
    successful_stacking_steps = []
    total_successful_stackings = 0
    total_attempted_stackings = 0
    
    # Stacking pairs: (odd_step, even_step)
    stacking_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    
    print("Analyzing stacking metrics...")
    for i, run in enumerate(runs):
        run_name = run['run_name']
        
        # Parse numeric values
        try:
            runtime = float(run.get('_runtime', 0))
            total_steps = int(run.get('_step', 0))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse data for run {run_name}: {e}")
            continue
        
        # Check each stacking pair
        for odd_step, even_step in stacking_pairs:
            total_attempted_stackings += 1
            
            # Check if both subtasks in the stacking were completed
            try:
                odd_completions = int(run.get(f'step{odd_step}_completions', 0) or 0)
                even_completions = int(run.get(f'step{even_step}_completions', 0) or 0)
            except (ValueError, TypeError):
                odd_completions = 0
                even_completions = 0
            
            # A stacking is successful only if both subtasks were completed
            if odd_completions > 0 and even_completions > 0:
                total_successful_stackings += 1
                
                # For duration, we need to estimate when the stacking was completed
                # We'll use the total runtime as a proxy, but this could be refined
                # by looking at task completion steps if available
                successful_stacking_durations.append(runtime)
                
                # For steps, we'll use the total steps as a proxy
                # This could be refined by looking at specific task completion steps
                successful_stacking_steps.append(total_steps)
                
                print(f"  Run {run_name}: Stacking {odd_step}&{even_step} successful (odd: {odd_completions}, even: {even_completions})")
    
    # Calculate metrics
    results['total_attempted_stackings'] = total_attempted_stackings
    results['total_successful_stackings'] = total_successful_stackings
    results['stacking_success_rate'] = (total_successful_stackings / total_attempted_stackings * 100) if total_attempted_stackings > 0 else 0
    
    # Mean stacking duration (only for successful stackings)
    results['mean_stacking_duration'] = sum(successful_stacking_durations) / len(successful_stacking_durations) if successful_stacking_durations else 0
    
    # Mean stacking length steps (only for successful stackings)
    results['mean_stacking_length_steps'] = sum(successful_stacking_steps) / len(successful_stacking_steps) if successful_stacking_steps else 0
    
    # Additional statistics
    results['successful_stacking_durations'] = successful_stacking_durations
    results['successful_stacking_steps'] = successful_stacking_steps
    
    return results

def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("STACKING METRICS ANALYSIS")
    print("="*60)
    
    print(f"\n📊 STACKING SUCCESS METRICS:")
    print(f"  Total attempted stackings: {results['total_attempted_stackings']}")
    print(f"  Total successful stackings: {results['total_successful_stackings']}")
    print(f"  Stacking success rate: {results['stacking_success_rate']:.1f}%")
    
    print(f"\n⏱️  STACKING DURATION METRICS:")
    if results['mean_stacking_duration'] > 0:
        print(f"  Mean stacking duration: {results['mean_stacking_duration']:.2f} seconds")
        print(f"  (Based on {len(results['successful_stacking_durations'])} successful stackings)")
    else:
        print(f"  Mean stacking duration: No successful stackings found")
    
    print(f"\n🔄 STACKING STEP METRICS:")
    if results['mean_stacking_length_steps'] > 0:
        print(f"  Mean stacking length steps: {results['mean_stacking_length_steps']:.0f} steps")
        print(f"  (Based on {len(results['successful_stacking_steps'])} successful stackings)")
    else:
        print(f"  Mean stacking length steps: No successful stackings found")

def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerow(['Total Attempted Stackings', results['total_attempted_stackings'], 'count'])
        writer.writerow(['Total Successful Stackings', results['total_successful_stackings'], 'count'])
        writer.writerow(['Stacking Success Rate', f"{results['stacking_success_rate']:.1f}", '%'])
        writer.writerow(['Mean Stacking Duration', f"{results['mean_stacking_duration']:.2f}", 'seconds'])
        writer.writerow(['Mean Stacking Length Steps', f"{results['mean_stacking_length_steps']:.0f}", 'steps'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_stacking_metrics.py <data_dir> [output_file]")
        print("Example: python analyze_stacking_metrics.py data/vla_exports/TRUE_FINAL_pi0_hanoi_300_subtasks_3_blocks_random_selection stacking_results.csv")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "stacking_analysis_results.csv"
    
    print(f"Loading data from {data_dir}")
    print(f"Output will be saved to {output_file}")
    
    try:
        runs = load_wandb_data(data_dir)
        results = calculate_stacking_metrics(runs)
        print_results(results)
        save_results_to_csv(results, output_file)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
