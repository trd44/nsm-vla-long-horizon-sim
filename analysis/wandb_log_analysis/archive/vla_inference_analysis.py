#!/usr/bin/env python3
"""
vla_inference_analysis.py
-------------------------
Analyse W&B *wide‑format* CSVs that you produced with `wandb_download.py`.

Modes
-----
1. **Power + energy**
       python vla_inference_analysis.py \
           --power-csv  data/system_gpu.0.powerWatts.csv \
           --runtime-csv data/_runtime.csv \
           --save per_run_power_energy.csv

2. **Any other metric (e.g., GPU util %)**
       python vla_inference_analysis.py \
           --metric-csv data/system_gpu.0.gpu.csv

Outputs
-------
* Per‑run table printed to console (+ optional CSV)
* Aggregate stats (mean / totals across runs)
* Mean episode duration (s) when runtime is supplied
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
def analyse_power(power_csv: Path, runtime_csv: Path) -> tuple[pd.DataFrame, dict]:
    power_df   = pd.read_csv(power_csv)
    runtime_df = pd.read_csv(runtime_csv)

    if power_df.shape != runtime_df.shape:
        raise SystemExit("❌  runtime and power CSVs must have identical shape")

    per_run_stats = {}
    for run in power_df.columns:
        p = power_df[run].dropna().values
        t = runtime_df[run].dropna().values
        if len(p) == 0 or len(t) == 0:
            continue
        if len(p) != len(t):
            N = min(len(p), len(t))
            p, t = p[:N], t[:N]

        duration_s = t[-1] if len(t) else 0.0
        mean_p = p.mean()
        # Convert W·s (joules) to Wh: divide by 3600
        energy_Wh = mean_p * duration_s / 3600.0

        per_run_stats[run] = {
            "mean_power_W": mean_p,
            "energy_Wh":    energy_Wh,
            "duration_s":   duration_s,
        }

    per_run = pd.DataFrame(per_run_stats).T
    overall = {
        "runs": per_run.shape[0],
        "mean_power_W": per_run["mean_power_W"].mean(),
        "total_energy_Wh": per_run["energy_Wh"].sum(),
        "mean_duration_s": per_run["duration_s"].mean(),
    }
    return per_run, overall

# ----------------------------------------------------------------------------- #
def analyse_generic(metric_csv: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(metric_csv)
    per_run = pd.DataFrame({
        "mean":   df.mean(),
        "p95":    df.quantile(0.95),
        "median": df.median(),
        "std":    df.std(),
    })
    overall = {
        "runs": df.shape[1],
        "mean_of_means": per_run["mean"].mean(),
        "median_of_means": per_run["mean"].median(),
    }
    return per_run, overall

# ----------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-csv", type=Path,
                    help="Wide CSV with system/gpu.*powerWatts")
    ap.add_argument("--runtime-csv", type=Path,
                    help="Wide CSV with _runtime (seconds since start)")
    ap.add_argument("--metric-csv", type=Path,
                    help="Wide CSV for any other scalar metric")
    ap.add_argument("--save", type=Path,
                    help="Path to save per‑run table as CSV")
    args = ap.parse_args()

    if args.power_csv and args.runtime_csv:
        per_run, overall = analyse_power(args.power_csv, args.runtime_csv)
    elif args.metric_csv:
        per_run, overall = analyse_generic(args.metric_csv)
    else:
        raise SystemExit("❌  Provide (--power-csv & --runtime-csv) *or* --metric-csv")

    # -------- print -----------------------------------------------------------
    print("\n=== Per‑run statistics ===")
    print(per_run.to_string(float_format=lambda x: f"{x:8.3f}"))

    print("\n=== Aggregate stats ===")
    # Convert duration to a friendly string (hh:mm:ss) if it exists
    for k, v in overall.items():
        print(f"{k:<20}: {v:8.3f}" if isinstance(v, (float, np.floating)) else
              f"{k:<20}: {v}")

    if args.save:
        per_run.to_csv(args.save)
        print(f"\n✔️  Per‑run table saved to {args.save}")

# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()