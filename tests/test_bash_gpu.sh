#!/bin/bash

# Ensure script exits on first error and catches undefined variables
set -euo pipefail

# Parse command-line arguments
ARG1=""
ARG2=""
ARG3=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -arg1) ARG1="$2"; shift ;;  # Environment
        -arg2) ARG2="$2"; shift ;;  # Operator
        -arg3) ARG3="$2"; shift ;;  # Seed
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$ARG1" || -z "$ARG2" || -z "$ARG3" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 -arg1 <environment> -arg2 <operator> -arg3 <seed>"
    exit 1
fi

# Define log files
LOG_FILE="perf_eval.log"
GPU_LOG="gpu_log.txt"

# Start NVIDIA GPU monitoring in the background
nvidia-smi dmon -s pum -o DT -f "$GPU_LOG" &
NVIDIA_PID=$!

# Ensure cleanup of background processes
cleanup() {
    kill $NVIDIA_PID 2>/dev/null || true
    wait $NVIDIA_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Run the Python script with both CPU and GPU profiling
/usr/bin/time -v \
/usr/local/cuda/bin/nsys profile --trace=cuda,opengl,osrt,openacc \
perf stat -e \
    mem-loads,mem-stores,cache-references,cache-misses,cpu-cycles,instructions,branch-instructions,branch-misses,power/energy-cores/,power/energy-pkg/ \
    python learning/baselines/eval_rl.py --vision --env "$ARG1" --op "$ARG2" --seed "$ARG3" \
    &> "$LOG_FILE"



# Stop GPU monitoring
cleanup

# Append GPU power logs to the final log file
cat "$GPU_LOG" >> "$LOG_FILE"

echo "Performance logging completed. Logs saved to $LOG_FILE."
