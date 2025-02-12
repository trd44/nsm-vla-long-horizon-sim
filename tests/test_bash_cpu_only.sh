#!/bin/bash

# Ensure script exits on first error
set -e

# Define log file
LOG_FILE="perf_eval.log"

# List available perf events (logged for debugging)
echo "Available perf events:" > perf_events.log
perf list >> perf_events.log

# Run CPU profiling with time, power, and FLOPs
/usr/bin/time -v perf stat \
    -e power/energy-cores/,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double \
    python learning/baselines/eval_rl.py --eval_freq 100 --n_eval_episodes 2 \
    &> "$LOG_FILE"
