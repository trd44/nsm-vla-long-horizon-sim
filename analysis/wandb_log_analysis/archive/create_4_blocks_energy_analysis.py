#!/usr/bin/env python3
"""
Create combined stacking analysis with energy metrics for 4_blocks data.
This combines the stacking metrics with energy calculations similar to the 3_blocks analysis.
"""

import csv
import sys
from pathlib import Path

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

def calculate_stacking_metrics_with_energy(runs):
    """Calculate stacking metrics with energy calculations."""
    results = {}
    
    # Track successful stackings
    successful_stacking_durations = []
    successful_stacking_steps = []
    successful_stacking_energies = []
    total_successful_stackings = 0
    total_attempted_stackings = 0
    
    # Stacking pairs: (odd_step, even_step)
    stacking_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    
    print("Analyzing stacking metrics with energy...")
    for i, run in enumerate(runs):
        run_name = run['run_name']
        
        # Parse numeric values
        try:
            total_runtime = float(run.get('_runtime', 0))
            total_steps = int(run.get('_step', 0))
            episode_energy = float(run.get('episode_energy_wh', 0))
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
                # Get the completion step for the even subtask (when the stacking was completed)
                try:
                    even_completion_step = int(run.get(f'task_{even_step}_completed_step', 0) or 0)
                except (ValueError, TypeError):
                    even_completion_step = 0
                
                if even_completion_step > 0:
                    total_successful_stackings += 1
                    
                    # Calculate duration based on the completion step
                    if total_steps > 0:
                        duration_per_step = total_runtime / total_steps
                        stacking_duration = even_completion_step * duration_per_step
                        stacking_steps = even_completion_step
                        
                        # Calculate energy for this stacking
                        # Estimate energy proportionally based on steps completed
                        if episode_energy > 0:
                            energy_per_step = episode_energy / total_steps
                            stacking_energy = even_completion_step * energy_per_step
                        else:
                            stacking_energy = 0
                        
                        successful_stacking_durations.append(stacking_duration)
                        successful_stacking_steps.append(stacking_steps)
                        successful_stacking_energies.append(stacking_energy)
                        
                        print(f"  Run {run_name}: Stacking {odd_step}&{even_step} completed at step {even_completion_step} (duration: {stacking_duration:.2f}s, steps: {stacking_steps}, energy: {stacking_energy:.3f}Wh)")
                    else:
                        print(f"  Run {run_name}: Stacking {odd_step}&{even_step} successful but no step data available")
    
    # Calculate metrics
    results['total_attempted_stackings'] = total_attempted_stackings
    results['total_successful_stackings'] = total_successful_stackings
    results['stacking_success_rate'] = (total_successful_stackings / total_attempted_stackings * 100) if total_attempted_stackings > 0 else 0
    
    # Mean stacking duration (only for successful stackings)
    results['mean_stacking_duration'] = sum(successful_stacking_durations) / len(successful_stacking_durations) if successful_stacking_durations else 0
    
    # Mean stacking length steps (only for successful stackings)
    results['mean_stacking_length_steps'] = sum(successful_stacking_steps) / len(successful_stacking_steps) if successful_stacking_steps else 0
    
    # Mean energy per successful stacking
    results['mean_energy_per_successful_stacking'] = sum(successful_stacking_energies) / len(successful_stacking_energies) if successful_stacking_energies else 0
    
    return results

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
        writer.writerow(['Mean Energy per Successful Stacking', f"{results['mean_energy_per_successful_stacking']:.4f}", 'Wh'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_4_blocks_energy_analysis.py <data_dir> [output_file]")
        print("Example: python create_4_blocks_energy_analysis.py data/vla_exports/FINAL_pi0_hanoi_300_subtasks_4_blocks SUBTASK_4_Blocks_stacking_analysis_with_energy.csv")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "SUBTASK_4_Blocks_stacking_analysis_with_energy.csv"
    
    print(f"Loading data from {data_dir}")
    print(f"Output will be saved to {output_file}")
    
    try:
        runs = load_wandb_data(data_dir)
        results = calculate_stacking_metrics_with_energy(runs)
        save_results_to_csv(results, output_file)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
