#!/bin/bash

# Ensure script exits on first error and catches undefined variables
set -euo pipefail

# Parse command-line arguments
ARG1=""
ARG2=""
ARG3=""
LOG_DIR=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -arg1) ARG1="$2"; shift ;;  # Environment
        -arg2) ARG2="$2"; shift ;;  # Operator
        -arg3) ARG3="$2"; shift ;;  # Seed
        -path) LOG_DIR="$2"; shift ;;  # Log directory path
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$ARG1" || -z "$ARG2" || -z "$ARG3" || -z "$LOG_DIR" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 -arg1 <environment> -arg2 <operator> -arg3 <seed> -path <log_directory>"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Define log file paths
LOG_FILE="$LOG_DIR/perf_eval.log"
GPU_LOG="$LOG_DIR/gpu_log.txt"
CPU_POWER_LOG="$LOG_DIR/cpu_power_log.txt"
PERF_LOG="$LOG_DIR/perf_metrics.log"
GPU_COMPUTATION_LOG="$LOG_DIR/gpu_computation_log.txt"
SUM_LOG="$LOG_DIR/summary_log.txt"

# Clean previous logs
> "$LOG_FILE"
> "$GPU_LOG"
> "$CPU_POWER_LOG"
> "$PERF_LOG"
> "$GPU_COMPUTATION_LOG"
> "$SUM_LOG"

# Start NVIDIA GPU monitoring in the background (track memory, utilization, etc.)
nvidia-smi dmon -o DT -f "$GPU_LOG" &
NVIDIA_PID=$!

# Start CPU power monitoring with rapl-read (No sudo required)
./uarch-configure/rapl-read/rapl-read > "$CPU_POWER_LOG" &
RAPL_PID=$!

# Ensure cleanup of background processes
cleanup() {
    echo "Cleaning up..."

    # Append logs to the final log file
    cat "$GPU_LOG" >> "$LOG_FILE"
    cat "$PERF_LOG" >> "$LOG_FILE"
    cat "$GPU_COMPUTATION_LOG" >> "$LOG_FILE"

    # Generate human-friendly summary
    echo -e "\n==================== Performance Summary ====================" >> "$SUM_LOG"

    # Extract total run time from /usr/bin/time output in PERF_LOG
    RUN_TIME=$(grep "Elapsed (wall clock) time" "$PERF_LOG" | awk '{print $8}')
    echo -e "⏱️  Total Run Time: $RUN_TIME" >> "$SUM_LOG"



    # Extract CPU power usage summary
    if [[ -f "$CPU_POWER_LOG" ]]; then
        CPU_ENERGY=$(awk '{sum += $1} END {print sum}' "$CPU_POWER_LOG")
        echo -e "🖥️  Total CPU Energy: ${CPU_ENERGY} Joules" >> "$SUM_LOG"
    fi

    # Extract GPU power usage summary
    if [[ -f "$GPU_LOG" ]]; then
        GPU_ENERGY=$(awk '{sum += $2} END {print sum}' "$GPU_LOG")
        echo -e "🎮 Total GPU Energy: ${GPU_ENERGY} Joules" >> "$SUM_LOG"
    fi

    # Extract key CPU performance metrics from perf stat
    echo -e "\n💾 CPU Performance Metrics:" >> "$SUM_LOG"
    grep -E "mem-loads|mem-stores|cache-references|cache-misses|cpu-cycles|instructions|branch-instructions|branch-misses" "$PERF_LOG" >> "$SUM_LOG"

    if [[ ! -s "$GPU_COMPUTATION_LOG" ]]; then
        echo "Error: $GPU_COMPUTATION_LOG is empty or not found."
        exit 1
    fi

    # Extract GPU usage information (skipping the header row)
    GPU_USAGE=$(grep -E "^[0-9]" "$GPU_LOG" | tail -n +2)

    # Check if GPU_USAGE has content before appending
    if [[ -n "$GPU_USAGE" ]]; then
        echo -e "\n🎮 GPU Usage / Computation / Memory:" >> "$SUM_LOG"
        echo "$GPU_USAGE" >> "$SUM_LOG"
    else
        echo "No GPU usage data available." >> "$SUM_LOG"
    fi


    # Initialize variables to accumulate metrics
    total_gpu_memory=0
    total_gpu_time_duration=0
    total_gpu_occupancy=0
    total_gpu_cycles_active=0
    total_gpu_warps_active=0
    count=0

    # Iterate through the GPU computation log to sum values
    while IFS= read -r line; do
        if [[ "$line" =~ gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed ]]; then
            total_gpu_memory=$(echo "$total_gpu_memory + $(echo "$line" | awk '{print $3}')" | bc)
            ((count++))
        elif [[ "$line" =~ gpu__time_duration\.sum ]]; then
            total_gpu_time_duration=$(echo "$total_gpu_time_duration + $(echo "$line" | awk '{print $3}')" | bc)
        elif [[ "$line" =~ launch__occupancy_per_block_size ]]; then
            total_gpu_occupancy=$(echo "$total_gpu_occupancy + $(echo "$line" | awk '{print $3}')" | bc)
        elif [[ "$line" =~ sm__cycles_active\.avg ]]; then
            total_gpu_cycles_active=$(echo "$total_gpu_cycles_active + $(echo "$line" | awk '{print $3}')" | bc)
        elif [[ "$line" =~ sm__warps_active\.avg\.per_cycle_active ]]; then
            total_gpu_warps_active=$(echo "$total_gpu_warps_active + $(echo "$line" | awk '{print $3}')" | bc)
        fi
    done < "$GPU_COMPUTATION_LOG"

    # Calculate averages (if needed)
    if [[ $count -gt 0 ]]; then
        avg_gpu_memory=$(echo "$total_gpu_memory / $count" | bc -l)
    fi

    # Append summed or averaged metrics to the summary log
    echo -e "\n🎮 GPU Computational Metrics:" >> "$SUM_LOG"
    echo -e "💡 Memory Throughput Efficiency (Average): ${avg_gpu_memory}%" >> "$SUM_LOG"
    echo -e "⏳ Total GPU Time Duration: ${total_gpu_time_duration} microseconds" >> "$SUM_LOG"
    echo -e "💻 Occupancy per Block Size (Sum): ${total_gpu_occupancy}" >> "$SUM_LOG"
    echo -e "💥 Active Compute Cycles (Sum): ${total_gpu_cycles_active} cycles" >> "$SUM_LOG"
    echo -e "🔄 Warps Active per Cycle (Sum): ${total_gpu_warps_active} warps" >> "$SUM_LOG"


    # Extract GPU computational expenses (FLOPs) and other relevant metrics
    GPU_COMPUTATION=$(grep -E "flop_count|sm__warps_active|sm__cycles_active|gpu__time_duration|gpu__compute_memory_throughput|launch__occupancy" "$GPU_COMPUTATION_LOG")
    echo -e "\n🎮 GPU Computation and Metrics:" >> "$SUM_LOG"
    echo "$GPU_COMPUTATION" >> "$SUM_LOG"

    echo -e "\n==================== End of Log ====================" >> "$SUM_LOG"

    # Kill the background processes
    kill $NVIDIA_PID 2>/dev/null || true
    kill $RAPL_PID 2>/dev/null || true
    
    # Wait for background processes to terminate before proceeding
    wait $NVIDIA_PID 2>/dev/null || true
    wait $RAPL_PID 2>/dev/null || true

}

trap cleanup EXIT INT TERM SIGINT

export DISPLAY=:99
export MUJOCO_GL=egl
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate lxm
python --version
python -c "import torch; print(torch.cuda.is_available())"
echo "Python path: $(which python)"

# Run the Python script with profiling tools and capture all metrics in one command
/usr/bin/time -v perf stat \
    -e mem-loads,mem-stores,cache-references,cache-misses,cpu-cycles,instructions,branch-instructions,branch-misses \
    ncu --metrics sm__warps_active.avg.per_cycle_active,sm__cycles_active.avg,gpu__time_duration.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__occupancy_per_block_size --log-file "$GPU_COMPUTATION_LOG" \
    python learning/baselines/eval_rl.py --vision --env "$ARG1" --op "$ARG2" --seed "$ARG3" \
    &> "$PERF_LOG"

# Stop monitoring
cleanup
