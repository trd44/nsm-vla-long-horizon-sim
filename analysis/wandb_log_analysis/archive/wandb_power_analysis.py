#!/usr/bin/env python3
"""
Analyze W&B system metrics, skipping periods when training was paused.

Example:
    python analyze_training_metrics.py --data-dir /path/to/csvs \
        --out metrics_report.csv --gap-mult 10
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- helpers ----------------------------------------------------------
def parse_time_col(series: pd.Series) -> pd.Series:
    """Convert 'Relative Time (Process)' to seconds (float)."""
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float)
    # assume HH:MM:SS[.ms]
    return pd.to_timedelta(series).dt.total_seconds()

def effective_deltas(t: np.ndarray, gap_mult: float = 10.0):
    """Clamp large gaps so 'paused' periods are excluded."""
    dt = np.diff(t)
    med = np.median(dt)
    threshold = gap_mult * med
    # The first sample contributes no time; append median to keep lengths equal
    return np.minimum(dt, threshold), threshold

def summarize(col: pd.Series):
    col = col.dropna()
    return {"mean": col.mean(), "max": col.max()}

# ---------- main analysis ----------------------------------------------------
def analyze_one(file: Path, gap_mult: float):
    df = pd.read_csv(file)
    t = parse_time_col(df["Relative Time (Process)"]).values
    dt_eff, thresh = effective_deltas(t, gap_mult)
    # Align dt with row indices (dt[i] is between rows i and i+1)
    dt_eff = np.insert(dt_eff, 0, 0.0)  # first sample has zero duration

    results = {}
    for col in df.columns:
        if col == "Relative Time (Process)" or col.endswith("__MIN") or col.endswith("__MAX"):
            continue
        values = df[col].values
        # For power we need energy; for others just average & max
        if "powerWatts" in col:
            energy_j = np.sum(values * dt_eff)           # watt-seconds (joules)
            active_time = dt_eff.sum()
            results["gpu_power_mean_W"] = energy_j / active_time
            results["gpu_energy_kWh"] = energy_j / (3.6e6)
        else:
            stats = summarize(df[col])
            key = col.split("/")[-1].split(".")[0]       # nice short name
            results[f"{key}_mean"] = stats["mean"]
            results[f"{key}_max"] = stats["max"]
    results["active_wall_clock_s"] = dt_eff.sum()
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory holding the six CSVs")
    p.add_argument("--gap-mult", type=float, default=30.0,
                   help="Treat a gap > gap_mult × median dt as 'paused'")
    p.add_argument("--out", help="Optional CSV path to save summary")
    args = p.parse_args()

    files = [
        "cpu_utilization_pct.csv",
        "gpu_utilization_pct.csv",
        "gpu_memory.csv",
        "gpu_power.csv",
        "process_memory.csv",
        "system_memory_pct.csv",
    ]
    summaries = []
    for f in files:
        path = Path(args.data_dir) / f
        if not path.exists():
            print(f"⚠️  Missing {path}, skipping")
            continue
        summaries.append(analyze_one(path, args.gap_mult))

    # merge all dicts (every file has active_wall_clock_s, take the first)
    merged = {}
    for d in summaries:
        merged.update(d)
    active_time_td = timedelta(seconds=merged.pop("active_wall_clock_s"))
    print("\n=== Training Metrics (active only) ===")
    print(f"Wall-clock training time  : {active_time_td}")
    print(f"Average GPU power (W)     : {merged.get('gpu_power_mean_W', 'N/A'):.1f}")
    print(f"Total GPU energy (kWh)    : {merged.get('gpu_energy_kWh', 'N/A'):.3f}")
    for k, v in merged.items():
        if k.startswith("gpu_power") or k.startswith("gpu_energy"):
            continue
        print(f"{k:<24}: {v:.2f}")

    if args.out:
        pd.DataFrame([merged]).to_csv(args.out, index=False)
        print(f"\n✔️  Summary saved to {args.out}")

if __name__ == "__main__":
    main()