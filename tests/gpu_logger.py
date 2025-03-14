import subprocess
import time
import argparse
import os
import psutil

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
            pid = int(parts[1])
            cmd_name = get_process_name(pid)
            parsed_data.append({
                "gpu_id": parts[0],
                "pid": pid,
                "cmd_name": cmd_name,
                "type": parts[2],  # Compute (C) or Graphics (G)
                "sm_util": parts[3],  # GPU compute usage (%)
                "mem_usage_mb": parts[4],  # GPU memory usage (MB)
                "enc_util": parts[5],  # Encoder usage (%)
                "dec_util": parts[6]   # Decoder usage (%)
            })
    return parsed_data

def get_process_name(pid):
    """Get the command name of a process given its PID"""
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "Unknown"

def log_gpu_usage(dmon_log_path, pmon_log_path):
    """Continuously logs GPU usage to separate dmon and pmon files."""
    with open(dmon_log_path, "w") as dmon_log, open(pmon_log_path, "w") as pmon_log:
        dmon_log.write("timestamp,gpu_id,sm_util,mem_util,enc_util,dec_util,power_w\n")
        pmon_log.write("timestamp,gpu_id,pid,cmd_name,type,sm_util,mem_usage_mb,enc_util,dec_util\n")

    print(f"Logging GPU usage to {dmon_log_path} (device) and {pmon_log_path} (process)... Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            dmon_lines = run_nvidia_smi("nvidia-smi dmon -s pum -c 1")  # Power, Utilization, Memory
            pmon_lines = run_nvidia_smi("nvidia-smi pmon -s um -c 1")  # Process-level usage

            dmon_data = parse_dmon_output(dmon_lines)
            pmon_data = parse_pmon_output(pmon_lines)

            with open(dmon_log_path, "a") as dmon_log:
                for d in dmon_data:
                    dmon_log.write(f"{timestamp},{d['gpu_id']},{d['sm_util']},{d['mem_util']},{d['enc_util']},{d['dec_util']},{d['power_w']}\n")

            with open(pmon_log_path, "a") as pmon_log:
                for p in pmon_data:
                    pmon_log.write(f"{timestamp},{p['gpu_id']},{p['pid']},{p['cmd_name']},{p['type']},{p['sm_util']},{p['mem_usage_mb']},{p['enc_util']},{p['dec_util']}\n")

            time.sleep(1)  # Log every second

    except KeyboardInterrupt:
        print("\nStopped logging GPU usage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log GPU usage from nvidia-smi dmon and pmon to separate files.")
    parser.add_argument("--dmon_log", type=str, default=os.path.join(os.getcwd(), "gpu_dmon_log.csv"),
                        help="Path to device-level GPU log file (default: gpu_dmon_log.csv).")
    parser.add_argument("--pmon_log", type=str, default=os.path.join(os.getcwd(), "gpu_pmon_log.csv"),
                        help="Path to process-level GPU log file (default: gpu_pmon_log.csv).")
    args = parser.parse_args()

    log_gpu_usage(args.dmon_log, args.pmon_log)
