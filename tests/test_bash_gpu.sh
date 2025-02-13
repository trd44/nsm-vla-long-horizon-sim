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

# Clean previous logs
> "$LOG_FILE"
> "$GPU_LOG"
> "$CPU_POWER_LOG"
> "$PERF_LOG"

# Start NVIDIA GPU monitoring in the background
nvidia-smi dmon -s pum -o DT -f "$GPU_LOG" &
NVIDIA_PID=$!

# Ensure cleanup of background processes
cleanup() {
    kill $NVIDIA_PID 2>/dev/null || true
    wait $NVIDIA_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DISPLAY=:99
export MUJOCO_GL=egl

# Start CPU power monitoring with turbostat
sudo turbostat --quiet --Summary --interval 1 > "$CPU_POWER_LOG" & 
TURBO_PID=$!

# Run the Python script with perf profiling
/usr/bin/time -v perf stat -e \
    mem-loads,mem-stores,cache-references,cache-misses,cpu-cycles,instructions,branch-instructions,branch-misses \
    python learning/baselines/eval_rl.py --vision --env "$ARG1" --op "$ARG2" --seed "$ARG3" \
    &> "$PERF_LOG"

# Stop monitoring
kill $TURBO_PID 2>/dev/null || true
wait $TURBO_PID 2>/dev/null || true
cleanup

# Append GPU monitoring logs to the final log file
cat "$GPU_LOG" >> "$LOG_FILE"

# Append perf profiling logs
cat "$PERF_LOG" >> "$LOG_FILE"

# Generate human-friendly summary
echo -e "\n==================== Performance Summary ====================" >> "$LOG_FILE"

# Extract CPU power usage summary
if [[ -f "$CPU_POWER_LOG" ]]; then
    CPU_ENERGY=$(awk '{sum += $1} END {print sum}' "$CPU_POWER_LOG")
    echo -e "Total CPU Energy: ${CPU_ENERGY} Joules" >> "$LOG_FILE"
fi

# Extract GPU power usage summary
if [[ -f "$GPU_LOG" ]]; then
    GPU_ENERGY=$(awk '{sum += $2} END {print sum}' "$GPU_LOG")
    echo -e "Total GPU Energy: ${GPU_ENERGY} Joules" >> "$LOG_FILE"
fi

# Extract key CPU performance metrics from perf stat
echo -e "\n CPU Performance Metrics:" >> "$LOG_FILE"
grep -E "mem-loads|mem-stores|cache-references|cache-misses|cpu-cycles|instructions|branch-instructions|branch-misses" "$PERF_LOG" >> "$LOG_FILE"

# Extract time & memory usage from /usr/bin/time
echo -e "\n Execution Time Metrics:" >> "$LOG_FILE"
grep -E "Elapsed|User time|System time|Maximum resident set size" "$PERF_LOG" >> "$LOG_FILE"

echo -e "Performance logging completed. Logs saved in: $LOG_DIR" >> "$LOG_FILE"
