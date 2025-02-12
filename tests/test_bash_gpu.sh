# Start NVIDIA GPU monitoring in the background
nvidia-smi dmon -s pum -o DT -f gpu_log.txt &
NVIDIA_PID=$!

# Run CPU profiling (time, power, FLOPs)
# Ensure correct Python script is used
/usr/bin/time -v perf stat -e power/energy-cores/,power/energy-ram/,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double python tests/keyboard_kitchen.py --render --seed 2 | tee full_log.txt

# Wait for GPU monitoring to finish before killing it
wait $NVIDIA_PID
kill $NVIDIA_PID

# Run GPU FLOP profiling separately
nvprof --metrics flop_count_sp,flop_count_dp python tests/keyboard_kitchen.py --render

# Append GPU logs to the final log
cat gpu_log.txt >> full_log.txt
