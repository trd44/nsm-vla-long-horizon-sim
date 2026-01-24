#!/usr/bin/env python3
"""
Analyze inference results from W&B data.
Calculates success rates, advancement rates, durations, and other metrics.

python analyze_inference_results.py \
    data/vla_exports/TRUE_FINAL_pi0_hanoi_300_subtasks_3_blocks_random_selection \
    SUBTASK_3_Blocks_analysis_results_complete.csv
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

def calculate_metrics(runs):
    """Calculate all requested metrics."""
    results = {}
    
    # Basic counts
    total_runs = len(runs)
    successful_runs = 0  # score == 14 (max observed score)
    total_tasks_available = 0
    total_tasks_completed = 0
    total_subtasks_completed = 0
    
    # Advancement rate calculation
    total_stacking_attempts = 0
    total_stacking_successes = 0
    
    # Duration and step data
    successful_durations = []
    successful_steps = []
    all_durations = []
    all_steps = []
    
    # Power data (if available) - these will be empty for inference runs
    gpu_power_values = []
    cpu_power_values = []
    
    # Energy calculation variables
    total_energy_per_episode = 0
    energy_values = []
    
    print("Analyzing runs...")
    for i, run in enumerate(runs):
        run_name = run['run_name']
        
        # Parse numeric values
        try:
            score = int(run.get('score', 0))
            runtime = float(run.get('_runtime', 0))
            steps = int(run.get('_step', 0))
            tasks_available = int(run.get('total_tasks_available', 0))
            tasks_completed = int(run.get('total_tasks_completed', 0))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse data for run {run_name}: {e}")
            continue
            
        # Calculate subtasks completed (sum of all step completions)
        subtasks_completed = 0
        episode_stacking_attempts = 0
        episode_stacking_successes = 0
        
        for key, value in run.items():
            if key.startswith('step') and key.endswith('_completions'):
                try:
                    completions = int(value or 0)
                    subtasks_completed += completions
                    episode_stacking_successes += completions
                    # Each step can be attempted multiple times, but we only count successful completions
                    # For advancement rate, we assume each step was attempted at least once
                    episode_stacking_attempts += max(1, completions)  # At least 1 attempt per step
                except (ValueError, TypeError):
                    pass
        
        # Track totals
        total_tasks_available += tasks_available
        total_tasks_completed += tasks_completed
        total_subtasks_completed += subtasks_completed
        
        # Track stacking attempts and successes for advancement rate
        total_stacking_attempts += episode_stacking_attempts
        total_stacking_successes += episode_stacking_successes
        
        # Track durations and steps
        all_durations.append(runtime)
        all_steps.append(steps)
        
        # Check if successful (score == 14, which is the maximum observed)
        if score == 14:
            successful_runs += 1
            successful_durations.append(runtime)
            successful_steps.append(steps)
        
        # Look for power data in the run data
        if 'gpu_power_watts' in run:
            try:
                gpu_power_values.append(float(run['gpu_power_watts']))
            except (ValueError, TypeError):
                pass
        
        if 'cpu_power_watts' in run:
            try:
                cpu_power_values.append(float(run['cpu_power_watts']))
            except (ValueError, TypeError):
                pass
        
        # Look for energy data
        if 'episode_energy_wh' in run:
            try:
                energy_values.append(float(run['episode_energy_wh']))
            except (ValueError, TypeError):
                pass
    
    # Calculate metrics
    results['total_runs'] = total_runs
    results['successful_runs'] = successful_runs
    results['success_rate'] = (successful_runs / total_runs * 100) if total_runs > 0 else 0
    
    # Advancement rate: stacking successes / stacking attempts averaged over all episodes
    results['advancement_rate'] = (total_stacking_successes / total_stacking_attempts * 100) if total_stacking_attempts > 0 else 0
    
    # Duration metrics
    results['mean_duration_all'] = sum(all_durations) / len(all_durations) if all_durations else 0
    results['mean_duration_successful'] = sum(successful_durations) / len(successful_durations) if successful_durations else 0
    
    # Step metrics
    results['mean_steps_all'] = sum(all_steps) / len(all_steps) if all_steps else 0
    results['mean_steps_successful'] = sum(successful_steps) / len(successful_steps) if successful_steps else 0
    
    # Power metrics (will be 0 if no power data available)
    results['mean_gpu_power'] = sum(gpu_power_values) / len(gpu_power_values) if gpu_power_values else 0
    results['mean_cpu_power'] = sum(cpu_power_values) / len(cpu_power_values) if cpu_power_values else 0
    
    # Energy calculation from episode data
    results['total_energy_per_episode'] = sum(energy_values) / len(energy_values) if energy_values else 0
    
    # Calculate energy per successful stacking (only for episodes with score 14)
    successful_energy_values = []
    for run in runs:
        try:
            score = int(run.get('score', 0))
            energy = float(run.get('episode_energy_wh', 0))
            if score == 14 and energy > 0:  # Only perfect score episodes
                successful_energy_values.append(energy)
        except (ValueError, TypeError):
            pass
    
    # Assuming 7 stackings required for score 14
    stackings_per_perfect_score = 7
    if successful_energy_values:
        avg_energy_successful = sum(successful_energy_values) / len(successful_energy_values)
        results['energy_per_successful_stacking'] = avg_energy_successful / stackings_per_perfect_score
        results['successful_episodes_count'] = len(successful_energy_values)
    else:
        results['energy_per_successful_stacking'] = 0
        results['successful_episodes_count'] = 0
    
    # Additional useful metrics
    results['total_tasks_available'] = total_tasks_available
    results['total_tasks_completed'] = total_tasks_completed
    results['total_subtasks_completed'] = total_subtasks_completed
    results['task_completion_rate'] = (total_tasks_completed / total_tasks_available * 100) if total_tasks_available > 0 else 0
    
    # Stacking statistics
    results['total_stacking_attempts'] = total_stacking_attempts
    results['total_stacking_successes'] = total_stacking_successes
    results['mean_stacking_attempts_per_episode'] = total_stacking_attempts / total_runs if total_runs > 0 else 0
    results['mean_stacking_successes_per_episode'] = total_stacking_successes / total_runs if total_runs > 0 else 0
    
    return results

def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("INFERENCE RESULTS ANALYSIS")
    print("="*60)
    
    print(f"\n📊 SUCCESS METRICS:")
    print(f"  Total runs: {results['total_runs']}")
    print(f"  Successful runs (score=14): {results['successful_runs']}")
    print(f"  Success rate: {results['success_rate']:.1f}%")
    
    print(f"\n📈 ADVANCEMENT METRICS:")
    print(f"  Total tasks available: {results['total_tasks_available']}")
    print(f"  Total tasks completed: {results['total_tasks_completed']}")
    print(f"  Task completion rate: {results['task_completion_rate']:.1f}%")
    print(f"  Total subtasks completed: {results['total_subtasks_completed']}")
    print(f"  Advancement rate: {results['advancement_rate']:.1f}%")
    print(f"  Total stacking attempts: {results['total_stacking_attempts']}")
    print(f"  Total stacking successes: {results['total_stacking_successes']}")
    print(f"  Mean stacking attempts per episode: {results['mean_stacking_attempts_per_episode']:.1f}")
    print(f"  Mean stacking successes per episode: {results['mean_stacking_successes_per_episode']:.1f}")
    
    print(f"\n⏱️  DURATION METRICS:")
    print(f"  Mean duration (all runs): {results['mean_duration_all']:.2f} seconds")
    print(f"  Mean duration (successful): {results['mean_duration_successful']:.2f} seconds")
    
    print(f"\n🔄 STEP METRICS:")
    print(f"  Mean steps (all runs): {results['mean_steps_all']:.0f} steps")
    print(f"  Mean steps (successful): {results['mean_steps_successful']:.0f} steps")
    
    print(f"\n⚡ POWER METRICS:")
    if results['mean_gpu_power'] > 0:
        print(f"  Mean GPU power: {results['mean_gpu_power']:.1f} W")
    else:
        print(f"  Mean GPU power: Not available (inference runs)")
    
    if results['mean_cpu_power'] > 0:
        print(f"  Mean CPU power: {results['mean_cpu_power']:.1f} W")
    else:
        print(f"  Mean CPU power: Not available (inference runs)")
    
    if results['total_energy_per_episode'] > 0:
        print(f"  Total energy per episode: {results['total_energy_per_episode']:.3f} Wh")
    else:
        print(f"  Total energy per episode: Not available (requires power data)")
    
    if results['energy_per_successful_stacking'] > 0:
        print(f"  Energy per successful stacking: {results['energy_per_successful_stacking']:.3f} Wh")
        print(f"  (Based on {results['successful_episodes_count']} episodes with perfect score)")
    else:
        print(f"  Energy per successful stacking: Not available")

def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerow(['Total Runs', results['total_runs'], 'count'])
        writer.writerow(['Successful Runs', results['successful_runs'], 'count'])
        writer.writerow(['Success Rate', f"{results['success_rate']:.1f}", '%'])
        writer.writerow(['Advancement Rate', f"{results['advancement_rate']:.1f}", '%'])
        writer.writerow(['Task Completion Rate', f"{results['task_completion_rate']:.1f}", '%'])
        writer.writerow(['Mean Duration (All)', f"{results['mean_duration_all']:.2f}", 'seconds'])
        writer.writerow(['Mean Duration (Successful)', f"{results['mean_duration_successful']:.2f}", 'seconds'])
        writer.writerow(['Mean Steps (All)', f"{results['mean_steps_all']:.0f}", 'steps'])
        writer.writerow(['Mean Steps (Successful)', f"{results['mean_steps_successful']:.0f}", 'steps'])
        writer.writerow(['Mean GPU Power', f"{results['mean_gpu_power']:.1f}", 'W'])
        writer.writerow(['Mean CPU Power', f"{results['mean_cpu_power']:.1f}", 'W'])
        writer.writerow(['Total Energy per Episode', f"{results['total_energy_per_episode']:.3f}", 'Wh'])
        writer.writerow(['Energy per Successful Stacking', f"{results['energy_per_successful_stacking']:.3f}", 'Wh'])
        writer.writerow(['Successful Episodes Count', results['successful_episodes_count'], 'count'])
        writer.writerow(['Total Tasks Available', results['total_tasks_available'], 'count'])
        writer.writerow(['Total Tasks Completed', results['total_tasks_completed'], 'count'])
        writer.writerow(['Total Subtasks Completed', results['total_subtasks_completed'], 'count'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_inference_results.py <data_dir> [output_file]")
        print("Example: python analyze_inference_results.py data/vla_exports/One_Task_3_Blocks_Random_Selection results.csv")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "inference_analysis_results.csv"
    
    print(f"Loading data from {data_dir}")
    print(f"Output will be saved to {output_file}")
    
    try:
        runs = load_wandb_data(data_dir)
        results = calculate_metrics(runs)
        print_results(results)
        save_results_to_csv(results, output_file)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
