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
        if len(parts) >= 11 and parts[0].isdigit():  # Ensure we have numeric GPU indices
            parsed_data.append({
                "gpu_id": parts[0],
                "power_w": parts[1],   # Power consumption (W)
                "gpu_temp_c": parts[2], # GPU temperature (°C)
                "mem_temp_c": parts[3], # Memory temperature (°C) or '-'
                "sm_util": parts[4],    # Streaming multiprocessor usage (%)
                "mem_util": parts[5],   # Memory utilization (%)
                "enc_util": parts[6],   # Encoder usage (%)
                "dec_util": parts[7],   # Decoder usage (%)
                "jpg_util": parts[8],   # JPEG decoding (%)
                "ofa_util": parts[9],   # Optical Flow Accelerator (%)
                "fb_mem_mb": parts[10], # Framebuffer memory (MB)
                "bar1_mem_mb": parts[11], # BAR1 memory (MB)
                "ccpm_mem_mb": parts[12] # Compute Process Memory (MB)
            })
    return parsed_data

def parse_pmon_output(pmon_lines):
    """Parse the output of nvidia-smi pmon"""
    parsed_data = []
    for line in pmon_lines:
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit():  # Ensure we have numeric GPU indices
            pid = int(parts[1])
            cmd_line = get_process_cmdline(pid)
            parsed_data.append({
                "gpu_id": parts[0],
                "pid": pid,
                "type": parts[2],  # Compute (C), Graphics (G), or Both (C+G)
                "sm_util": parts[3],  # GPU compute usage (%)
                "mem_util": parts[4],  # GPU memory usage (%)
                "enc_util": parts[5],  # Encoder usage (%)
                "dec_util": parts[6],  # Decoder usage (%)
                "fb_mem_mb": parts[7], # GPU memory usage (MB)
                "ccpm_mem_mb": parts[8], # Compute Process Memory (MB)
                "cmd_line": cmd_line  # Full command line (script + args)
            })
    return parsed_data

def get_process_cmdline(pid):
    """Get the full command line of a process given its PID"""
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "Unknown"

def log_gpu_usage(dmon_log_path, pmon_log_path):
    """Continuously logs GPU usage to separate dmon and pmon files."""
    with open(dmon_log_path, "w") as dmon_log, open(pmon_log_path, "w") as pmon_log:
        dmon_log.write("timestamp,gpu_id,power_w,gpu_temp_c,mem_temp_c,sm_util,mem_util,enc_util,dec_util,jpg_util,ofa_util,fb_mem_mb,bar1_mem_mb,ccpm_mem_mb\n")
        pmon_log.write("timestamp,gpu_id,pid,type,sm_util,mem_util,enc_util,dec_util,fb_mem_mb,ccpm_mem_mb,cmd_line\n")

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
                    dmon_log.write(f"{timestamp},{d['gpu_id']},{d['power_w']},{d['gpu_temp_c']},{d['mem_temp_c']},{d['sm_util']},{d['mem_util']},{d['enc_util']},{d['dec_util']},{d['jpg_util']},{d['ofa_util']},{d['fb_mem_mb']},{d['bar1_mem_mb']},{d['ccpm_mem_mb']}\n")

            with open(pmon_log_path, "a") as pmon_log:
                for p in pmon_data:
                    pmon_log.write(f"{timestamp},{p['gpu_id']},{p['pid']},{p['type']},{p['sm_util']},{p['mem_util']},{p['enc_util']},{p['dec_util']},{p['fb_mem_mb']},{p['ccpm_mem_mb']},{p['cmd_line']}\n")

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
