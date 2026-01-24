#!/usr/bin/env python3
"""
Simple power analysis for first 30,000 steps using only built-in Python libraries.
This avoids NumPy/pandas compatibility issues.
"""

import csv
import sys
from datetime import timedelta

def parse_time_to_seconds(time_str):
    """Convert time string to seconds (float)."""
    try:
        # Try parsing as float first
        return float(time_str)
    except ValueError:
        # Try parsing as HH:MM:SS format
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Unable to parse time: {time_str}")

def load_loss_data(filename):
    """Load loss data and return list of (step, loss) tuples."""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['Step'])
            loss = float(row['pi_hanoi_300_subtasks_no_masking - loss'])
            data.append((step, loss))
    return data

def load_power_data(filename):
    """Load power data and return list of (time, power) tuples."""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_str = row['Relative Time (Process)']
            time_sec = parse_time_to_seconds(time_str)
            # Handle different column names for different datasets
            if 'pi_hanoi_300_subtasks_no_masking - system/gpu.0.powerWatts' in row:
                power = float(row['pi_hanoi_300_subtasks_no_masking - system/gpu.0.powerWatts'])
            elif 'ft1 - system/gpu.0.powerWatts' in row:
                power = float(row['ft1 - system/gpu.0.powerWatts'])
            else:
                # Try to find any column with 'powerWatts' in the name
                power_col = None
                for col in row:
                    if 'powerWatts' in col and not col.endswith('__MIN') and not col.endswith('__MAX'):
                        power_col = col
                        break
                if power_col:
                    power = float(row[power_col])
                else:
                    raise ValueError(f"Could not find power column in {filename}")
            data.append((time_sec, power))
    return data

def find_time_range_for_steps(loss_data, power_data, max_steps):
    """Find the time range corresponding to the first max_steps."""
    # Filter loss data to first max_steps
    filtered_loss = [(step, loss) for step, loss in loss_data if step <= max_steps]
    
    if not filtered_loss:
        raise ValueError(f"No loss data found for steps up to {max_steps}")
    
    min_step = min(step for step, _ in filtered_loss)
    max_step = max(step for step, _ in filtered_loss)
    
    print(f"Analyzing steps {min_step} to {max_step} (total: {len(filtered_loss)} data points)")
    
    # Get time range from power data
    power_times = [time for time, _ in power_data]
    total_time = power_times[-1] - power_times[0]
    
    # Get total steps from loss data
    total_steps = max(step for step, _ in loss_data)
    
    # Estimate time per step
    time_per_step = total_time / total_steps
    
    # Calculate time range for our target steps
    start_time = power_times[0] + (min_step * time_per_step)
    end_time = power_times[0] + (max_step * time_per_step)
    
    print(f"Estimated time range: {start_time:.1f}s to {end_time:.1f}s")
    print(f"Total estimated duration: {end_time - start_time:.1f}s")
    
    return start_time, end_time

def analyze_power_detailed(power_data, start_time, end_time):
    """Detailed power analysis with comprehensive statistics."""
    # Filter power data to time range
    filtered_power = [(time, power) for time, power in power_data 
                     if start_time <= time <= end_time]
    
    if not filtered_power:
        raise ValueError(f"No power data found in time range {start_time:.1f}s to {end_time:.1f}s")
    
    print(f"Found {len(filtered_power)} power measurements in target time range")
    
    powers = [power for _, power in filtered_power]
    times = [time for time, _ in filtered_power]
    
    # Basic statistics
    avg_power = sum(powers) / len(powers)
    max_power = max(powers)
    min_power = min(powers)
    
    # Sort powers for percentile calculations
    sorted_powers = sorted(powers)
    n = len(sorted_powers)
    
    # Calculate percentiles
    p50 = sorted_powers[int(0.5 * n)] if n > 0 else 0
    p90 = sorted_powers[int(0.9 * n)] if n > 0 else 0
    p95 = sorted_powers[int(0.95 * n)] if n > 0 else 0
    p99 = sorted_powers[int(0.99 * n)] if n > 0 else 0
    
    # Calculate energy more accurately using trapezoidal rule
    total_energy_j = 0
    active_time = 0
    
    if len(times) > 1:
        for i in range(len(times) - 1):
            dt = times[i+1] - times[i]
            power_avg = (powers[i] + powers[i+1]) / 2  # Trapezoidal rule
            total_energy_j += power_avg * dt
            active_time += dt
    
    total_energy_kwh = total_energy_j / (3.6e6)
    
    # Calculate power efficiency metrics
    if active_time > 0:
        effective_avg_power = total_energy_j / active_time
    else:
        effective_avg_power = avg_power
    
    # Find periods of high/low power usage
    high_power_threshold = p90
    low_power_threshold = p10 = sorted_powers[int(0.1 * n)] if n > 0 else 0
    
    high_power_count = sum(1 for p in powers if p >= high_power_threshold)
    low_power_count = sum(1 for p in powers if p <= low_power_threshold)
    
    return {
        'avg_power_w': avg_power,
        'effective_avg_power_w': effective_avg_power,
        'max_power_w': max_power,
        'min_power_w': min_power,
        'p50_power_w': p50,
        'p90_power_w': p90,
        'p95_power_w': p95,
        'p99_power_w': p99,
        'total_energy_kwh': total_energy_kwh,
        'active_time_s': active_time,
        'num_measurements': len(filtered_power),
        'high_power_count': high_power_count,
        'low_power_count': low_power_count,
        'high_power_pct': (high_power_count / len(powers)) * 100,
        'low_power_pct': (low_power_count / len(powers)) * 100
    }

def save_results_to_csv(results, filename):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerow(['Average Power', f"{results['avg_power_w']:.2f}", 'W'])
        writer.writerow(['Effective Average Power', f"{results['effective_avg_power_w']:.2f}", 'W'])
        writer.writerow(['Max Power', f"{results['max_power_w']:.2f}", 'W'])
        writer.writerow(['Min Power', f"{results['min_power_w']:.2f}", 'W'])
        writer.writerow(['Median Power (P50)', f"{results['p50_power_w']:.2f}", 'W'])
        writer.writerow(['P90 Power', f"{results['p90_power_w']:.2f}", 'W'])
        writer.writerow(['P95 Power', f"{results['p95_power_w']:.2f}", 'W'])
        writer.writerow(['P99 Power', f"{results['p99_power_w']:.2f}", 'W'])
        writer.writerow(['Total Energy', f"{results['total_energy_kwh']:.3f}", 'kWh'])
        writer.writerow(['Active Time', f"{results['active_time_s']:.1f}", 'seconds'])
        writer.writerow(['Number of Measurements', f"{results['num_measurements']}", 'count'])
        writer.writerow(['High Power Count', f"{results['high_power_count']}", 'count'])
        writer.writerow(['Low Power Count', f"{results['low_power_count']}", 'count'])
        writer.writerow(['High Power Percentage', f"{results['high_power_pct']:.1f}", '%'])
        writer.writerow(['Low Power Percentage', f"{results['low_power_pct']:.1f}", '%'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_power_analysis.py <power_file> [max_steps] [output_file]")
        print("Example: python simple_power_analysis.py onetask_finetuning_power.csv 30000 results.csv")
        print("Note: If you have exactly 30k steps, you can omit max_steps or set it to 30000")
        sys.exit(1)
    
    power_file = sys.argv[1]
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"power_analysis_{max_steps}_steps.csv"
    
    print(f"Loading power data from {power_file}")
    print(f"Analyzing first {max_steps} steps")
    print(f"Output will be saved to {output_file}")
    
    # Load power data
    power_data = load_power_data(power_file)
    print(f"Power data: {len(power_data)} rows")
    
    # For a 30k step run, we can analyze the entire dataset
    # or just the first portion if it's longer
    if max_steps == 30000:
        print("Analyzing entire dataset (assuming 30k steps)")
        start_time = power_data[0][0]  # First timestamp
        end_time = power_data[-1][0]   # Last timestamp
        print(f"Time range: {start_time:.1f}s to {end_time:.1f}s")
        print(f"Total duration: {end_time - start_time:.1f}s")
    else:
        # If we need to estimate based on steps, we'd need loss data
        print("Error: For step-based filtering, you need both loss and power data")
        print("Use: python simple_power_analysis.py <loss_file> <power_file> <max_steps>")
        sys.exit(1)
    
    # Analyze power consumption
    print(f"\nAnalyzing power consumption...")
    results = analyze_power_detailed(power_data, start_time, end_time)
    
    # Display results
    print("\n" + "="*70)
    print(f"DETAILED POWER ANALYSIS FOR {max_steps} STEPS")
    print("="*70)
    
    if results['active_time_s'] > 0:
        active_time_td = timedelta(seconds=results['active_time_s'])
        print(f"Active training time: {active_time_td}")
    
    print(f"\nPower Statistics:")
    print(f"  Average power: {results['avg_power_w']:.1f} W")
    print(f"  Effective average power: {results['effective_avg_power_w']:.1f} W")
    print(f"  Max power: {results['max_power_w']:.1f} W")
    print(f"  Min power: {results['min_power_w']:.1f} W")
    
    print(f"\nPower Distribution:")
    print(f"  Median (P50): {results['p50_power_w']:.1f} W")
    print(f"  P90: {results['p90_power_w']:.1f} W")
    print(f"  P95: {results['p95_power_w']:.1f} W")
    print(f"  P99: {results['p99_power_w']:.1f} W")
    
    print(f"\nEnergy Consumption:")
    print(f"  Total energy: {results['total_energy_kwh']:.3f} kWh")
    
    print(f"\nPower Usage Patterns:")
    print(f"  High power measurements: {results['high_power_count']} ({results['high_power_pct']:.1f}%)")
    print(f"  Low power measurements: {results['low_power_count']} ({results['low_power_pct']:.1f}%)")
    print(f"  Total measurements: {results['num_measurements']}")
    
    # Save results
    save_results_to_csv(results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
