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

    # Extract total run time from /usr/bin/time output
    RUN_TIME=$(grep "Elapsed (wall clock) time" "$LOG_FILE" | awk '{print $4}')
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

    # Extract GPU usage information
    GPU_USAGE=$(grep -E "^(GPU|Processes)" "$GPU_LOG" | tail -n +2)
    echo -e "\n🎮 GPU Usage / Computation / Memory:" >> "$SUM_LOG"
    echo "$GPU_USAGE" >> "$SUM_LOG"

    # Extract GPU computational expenses (FLOPs)
    GPU_COMPUTATION=$(grep -E "flop_count" "$GPU_COMPUTATION_LOG")
    echo -e "\n🎮 GPU Computation (FLOPs):" >> "$SUM_LOG"
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
    ncu --metrics flop_count_sp --log-file "$GPU_COMPUTATION_LOG" \
    python learning/baselines/eval_rl.py --vision --env "$ARG1" --op "$ARG2" --seed "$ARG3" \
    &> "$PERF_LOG"

# Stop monitoring
cleanup

# Append logs to the final log file
cat "$GPU_LOG" >> "$LOG_FILE"
cat "$PERF_LOG" >> "$LOG_FILE"
cat "$GPU_COMPUTATION_LOG" >> "$LOG_FILE"

# Generate human-friendly summary
echo -e "\n==================== Performance Summary ====================" >> "$LOG_FILE"

# Extract total run time from /usr/bin/time output
RUN_TIME=$(grep "Elapsed (wall clock) time" "$LOG_FILE" | awk '{print $4}')
echo -e "⏱️  Total Run Time: $RUN_TIME" >> "$LOG_FILE"

# Extract CPU power usage summary
if [[ -f "$CPU_POWER_LOG" ]]; then
    CPU_ENERGY=$(awk '{sum += $1} END {print sum}' "$CPU_POWER_LOG")
    echo -e "🖥️  Total CPU Energy: ${CPU_ENERGY} Joules" >> "$LOG_FILE"
fi

# Extract GPU power usage summary
if [[ -f "$GPU_LOG" ]]; then
    GPU_ENERGY=$(awk '{sum += $2} END {print sum}' "$GPU_LOG")
    echo -e "🎮 Total GPU Energy: ${GPU_ENERGY} Joules" >> "$LOG_FILE"
fi

# Extract key CPU performance metrics from perf stat
echo -e "\n💾 CPU Performance Metrics:" >> "$LOG_FILE"
grep -E "mem-loads|mem-stores|cache-references|cache-misses|cpu-cycles|instructions|branch-instructions|branch-misses" "$PERF_LOG" >> "$LOG_FILE"

# Extract GPU usage information
GPU_USAGE=$(grep -E "^(GPU|Processes)" "$GPU_LOG" | tail -n +2)
echo -e "\n🎮 GPU Usage / Computation / Memory:" >> "$LOG_FILE"
echo "$GPU_USAGE" >> "$LOG_FILE"

# Extract GPU computational expenses (FLOPs)
GPU_COMPUTATION=$(grep -E "flop_count" "$GPU_COMPUTATION_LOG")
echo -e "\n🎮 GPU Computation (FLOPs):" >> "$LOG_FILE"
echo "$GPU_COMPUTATION" >> "$LOG_FILE"

echo -e "\n==================== End of Log ====================" >> "$LOG_FILE"
