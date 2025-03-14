import subprocess
import time
import argparse
import os

def run_nvidia_smi(command):
    """Run an nvidia-smi command and return its output as a list of lines."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.splitlines()
    except Exception as e:
        print(f"Error running {command}: {e}")
        return []

def parse_dmon_output(dmon_lines):
    """Parse the output of nvidia-smi dmon"""
    parsed_data = []
    for line in dmon_lines:
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit():  # Ignore headers, look for numeric GPU indices
            parsed_data.append({
                "gpu_id": parts[0],
                "sm_util": parts[1],  # Streaming multiprocessor usage (%)
                "mem_util": parts[2],  # Memory usage (%)
                "enc_util": parts[3],  # Encoder usage (%)
                "dec_util": parts[4],  # Decoder usage (%)
                "power_w": parts[5]  # Power consumption (W)
            })
    return parsed_data

def parse_pmon_output(pmon_lines):
    """Parse the output of nvidia-smi pmon"""
    parsed_data = []
    for line in pmon_lines:
        parts = line.split()
        if len(parts) >= 6 and parts[0].isdigit():  # Ignore headers, look for numeric PIDs
            parsed_data.append({
                "gpu_id": parts[0],
                "pid": parts[1],
                "type": parts[2],  # Compute (C) or Graphics (G)
                "sm_util": parts[3],  # GPU compute usage (%)
                "mem_util": parts[4],  # GPU memory usage (MB)
                "enc_util": parts[5],  # Encoder usage (%)
                "dec_util": parts[6]   # Decoder usage (%)
            })
    return parsed_data

def log_gpu_usage(log_path):
    """Continuously logs GPU usage to a file."""
    with open(log_path, "w") as log_file:
        log_file.write("timestamp,gpu_id,sm_util,mem_util,enc_util,dec_util,power_w,pid,type,sm_util_process,mem_util_process,enc_util_process,dec_util_process\n")

    print(f"Logging GPU usage to {log_path}... Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            dmon_lines = run_nvidia_smi("nvidia-smi dmon -s pum -c 1")  # Track Power, Utilization, Memory
            pmon_lines = run_nvidia_smi("nvidia-smi pmon -s um -c 1")  # Track process-level usage

            dmon_data = parse_dmon_output(dmon_lines)
            pmon_data = parse_pmon_output(pmon_lines)

            with open(log_path, "a") as log_file:
                for d in dmon_data:
                    log_file.write(f"{timestamp},{d['gpu_id']},{d['sm_util']},{d['mem_util']},{d['enc_util']},{d['dec_util']},{d['power_w']},,,,," + "\n")
                
                for p in pmon_data:
                    log_file.write(f"{timestamp},{p['gpu_id']},,,,,,{p['pid']},{p['type']},{p['sm_util']},{p['mem_util']},{p['enc_util']},{p['dec_util']}" + "\n")

            time.sleep(1)  # Log every second

    except KeyboardInterrupt:
        print("\nStopped logging GPU usage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log GPU usage from nvidia-smi dmon and pmon to a file.")
    parser.add_argument("--log", type=str, default=os.path.join(os.getcwd(), "gpu_log.csv"),
                        help="Path to log file (default: gpu_log.csv in the running folder).")
    args = parser.parse_args()

    log_gpu_usage(args.log)
