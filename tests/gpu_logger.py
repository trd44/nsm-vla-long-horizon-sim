import subprocess
import time
import argparse
import os
import psutil
import glob

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
        if len(parts) >= 11 and parts[0].isdigit():
            parsed_data.append({
                "gpu_id": parts[0],
                "power_w": parts[1],
                "gpu_temp_c": parts[2],
                "mem_temp_c": parts[3],
                "sm_util": parts[4],
                "mem_util": parts[5],
                "enc_util": parts[6],
                "dec_util": parts[7],
                "jpg_util": parts[8],
                "ofa_util": parts[9],
                "fb_mem_mb": parts[10],
                "bar1_mem_mb": parts[11],
                "ccpm_mem_mb": parts[12]
            })
    return parsed_data

def parse_pmon_output(pmon_lines):
    """Parse the output of nvidia-smi pmon"""
    parsed_data = []
    for line in pmon_lines:
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit():
            pid = int(parts[1])
            cmd_line = get_process_cmdline(pid)
            parsed_data.append({
                "gpu_id": parts[0],
                "pid": pid,
                "type": parts[2],
                "sm_util": parts[3],
                "mem_util": parts[4],
                "enc_util": parts[5],
                "dec_util": parts[6],
                "fb_mem_mb": parts[7],
                "ccpm_mem_mb": parts[8],
                "cmd_line": cmd_line
            })
    return parsed_data

def get_process_cmdline(pid):
    """Get the full command line of a process given its PID"""
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "Unknown"

def get_cpu_usage(pids):
    """Get CPU usage details for the given PIDs"""
    cpu_data = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            cpu_data.append({
                "pid": pid,
                "cmd_line": " ".join(p.cmdline()),
                "cpu_percent": p.cpu_percent(interval=0.1),
                "memory_percent": p.memory_percent(),
                "num_threads": p.num_threads(),
                "io_read_bytes": p.io_counters().read_bytes,
                "io_write_bytes": p.io_counters().write_bytes,
                "cpu_power_w": get_cpu_power()
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return cpu_data

def get_cpu_power():
    """Reads CPU power usage from Intel RAPL or AMD hwmon sensors."""
    power_w = None
    
    # Try Intel RAPL power reading
    rapl_paths = glob.glob("/sys/class/powercap/intel-rapl:*")
    for path in rapl_paths:
        energy_path = os.path.join(path, "energy_uj")
        if os.path.exists(energy_path):
            try:
                with open(energy_path, "r") as f:
                    energy_uj = int(f.read().strip())
                    power_w = energy_uj / 1e6  # Convert from microjoules to watts
                    break
            except:
                pass

    # Try AMD hwmon power reading
    if power_w is None:
        hwmon_paths = glob.glob("/sys/class/hwmon/hwmon*/power1_input")
        for path in hwmon_paths:
            try:
                with open(path, "r") as f:
                    power_uw = int(f.read().strip())
                    power_w = power_uw / 1e6  # Convert from microwatts to watts
                    break
            except:
                pass

    return power_w if power_w is not None else 0.0

def log_usage(dmon_log_path, pmon_log_path, cpu_log_path):
    """Continuously logs GPU and CPU usage to separate files."""
    with open(dmon_log_path, "w") as dmon_log, open(pmon_log_path, "w") as pmon_log, open(cpu_log_path, "w") as cpu_log:
        dmon_log.write("timestamp,gpu_id,power_w,gpu_temp_c,mem_temp_c,sm_util,mem_util,enc_util,dec_util,jpg_util,ofa_util,fb_mem_mb,bar1_mem_mb,ccpm_mem_mb\n")
        pmon_log.write("timestamp,gpu_id,pid,type,sm_util,mem_util,enc_util,dec_util,fb_mem_mb,ccpm_mem_mb,cmd_line\n")
        cpu_log.write("timestamp,pid,cmd_line,cpu_percent,memory_percent,num_threads,io_read_bytes,io_write_bytes,cpu_power_w\n")

    print(f"Logging GPU & CPU usage to {dmon_log_path}, {pmon_log_path}, {cpu_log_path}... Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Get GPU usage
            dmon_lines = run_nvidia_smi("nvidia-smi dmon -s pum -c 1")
            pmon_lines = run_nvidia_smi("nvidia-smi pmon -s um -c 1")

            dmon_data = parse_dmon_output(dmon_lines)
            pmon_data = parse_pmon_output(pmon_lines)

            # Extract PIDs from pmon data for CPU tracking
            gpu_pids = [p["pid"] for p in pmon_data]
            cpu_data = get_cpu_usage(gpu_pids)

            # Write logs
            with open(dmon_log_path, "a") as dmon_log:
                for d in dmon_data:
                    dmon_log.write(f"{timestamp},{d['gpu_id']},{d['power_w']},{d['gpu_temp_c']},{d['mem_temp_c']},{d['sm_util']},{d['mem_util']},{d['enc_util']},{d['dec_util']},{d['jpg_util']},{d['ofa_util']},{d['fb_mem_mb']},{d['bar1_mem_mb']},{d['ccpm_mem_mb']}\n")

            with open(pmon_log_path, "a") as pmon_log:
                for p in pmon_data:
                    pmon_log.write(f"{timestamp},{p['gpu_id']},{p['pid']},{p['type']},{p['sm_util']},{p['mem_util']},{p['enc_util']},{p['dec_util']},{p['fb_mem_mb']},{p['ccpm_mem_mb']},{p['cmd_line']}\n")

            with open(cpu_log_path, "a") as cpu_log:
                for c in cpu_data:
                    cpu_log.write(f"{timestamp},{c['pid']},{c['cmd_line']},{c['cpu_percent']},{c['memory_percent']},{c['num_threads']},{c['io_read_bytes']},{c['io_write_bytes']},{c['cpu_power_w']}\n")

            time.sleep(1)  # Log every second

    except KeyboardInterrupt:
        print("\nStopped logging GPU & CPU usage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log GPU and CPU usage for processes running on GPU.")
    parser.add_argument("--dmon_log", type=str, default=os.path.join(os.getcwd(), "gpu_dmon_log.csv"))
    parser.add_argument("--pmon_log", type=str, default=os.path.join(os.getcwd(), "gpu_pmon_log.csv"))
    parser.add_argument("--cpu_log", type=str, default=os.path.join(os.getcwd(), "cpu_log.csv"))
    args = parser.parse_args()

    log_usage(args.dmon_log, args.pmon_log, args.cpu_log)
