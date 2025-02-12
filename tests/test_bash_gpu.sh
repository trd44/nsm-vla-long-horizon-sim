#!/bin/bash

# Ensure script exits on first error
set -e

# Define log files
LOG_FILE="perf_eval.log"
GPU_LOG="gpu_log.txt"
FULL_LOG="full_log.txt"

# Start NVIDIA GPU monitoring in the background
nvidia-smi dmon -s pum -o DT -f "$GPU_LOG" &
NVIDIA_PID=$!

# Ensure the background process is killed when script exits
trap "kill $NVIDIA_PID 2>/dev/null" EXIT

# Run CPU profiling (time, power, FLOPs) and log output
/usr/bin/time -v perf stat \
    -e power/energy-cores/,power/energy-ram/,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double \
    python tests/keyboard_kitchen.py --render --seed 2 | tee "$FULL_LOG"

# Stop GPU monitoring before running `nvprof`
kill $NVIDIA_PID 2>/dev/null
wait $NVIDIA_PID 2>/dev/null || true  # Ignore errors if the process is already stopped

# Run GPU FLOP profiling separately and log output
nvprof --metrics flop_count_sp,flop_count_dp \
    python learning/baselines/eval_rl.py --eval_freq 100 --n_eval_episodes 2 \
    &>> "$FULL_LOG"

# Append GPU monitoring logs to the final log
cat "$GPU_LOG" >> "$FULL_LOG"
