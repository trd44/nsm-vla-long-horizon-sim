import time
import glob

def read_rapl_energy():
    """Reads CPU package energy in joules from RAPL sysfs."""
    rapl_files = glob.glob("/sys/class/powercap/intel-rapl:0:0/energy_uj")
    if not rapl_files:
        raise RuntimeError("No RAPL energy file found. Are you on an Intel CPU?")
    with open(rapl_files[0], "r") as f:
        energy_uj = int(f.read().strip())
    return energy_uj / 1_000_000  # convert microjoules to joules

def average_cpu_power(duration_sec=60, sample_interval=1.0):
    """Computes average CPU package power over a period in seconds."""
    samples = []
    start_energy = read_rapl_energy()
    start_time = time.time()

    while (time.time() - start_time) < duration_sec:
        time.sleep(sample_interval)
        samples.append(read_rapl_energy())

    end_energy = samples[-1]
    avg_power = (end_energy - start_energy) / duration_sec
    print(f"start energy is {start_energy}, end energy is {end_energy}")
    print(f"duration sec is {duration_sec}")
    print(f"Average CPU package power over {duration_sec}s: {avg_power:.2f} W")

if __name__ == "__main__":
    average_cpu_power(120)  # 20-minute test
