#!/usr/bin/env python3
"""
Analyze W&B experiment CSVs for energy usage across runs.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import tyro
except ImportError:
    print("❌ Missing dependencies. Run: pip install numpy pandas tyro")
    sys.exit(1)


@dataclasses.dataclass
class Args:
    path: Path = Path("data/experiments_exports/FINAL_Journal_Zero_Shot_2_Color_to_3_Color_NO_SWAP_AssemblyLineSorting")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    time_col = "Relative Time (Process)"
    if time_col not in df.columns:
        time_col = df.columns[0]
    df = df.rename(columns={time_col: "time_s"})
    return df


def _compute_energy_stats(time_s: np.ndarray, power_w: np.ndarray) -> dict[str, float]:
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0:
        return {"avg_power_w": float("nan"), "energy_j": float("nan"), "energy_wh": float("nan")}
    energy_j = float(np.trapezoid(power_w, time_s))
    energy_wh = energy_j / 3600.0
    avg_power_w = energy_j / duration_s
    return {"avg_power_w": avg_power_w, "energy_j": energy_j, "energy_wh": energy_wh, "duration_s": duration_s}


def _analyze_metric_csv(csv_path: Path) -> tuple[list[dict[str, object]], dict[str, float]]:
    df = _load_csv(csv_path)
    if df.empty or "time_s" not in df.columns or len(df.columns) < 2:
        return [], {}

    df = df.sort_values("time_s")
    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    results: list[dict[str, object]] = []

    for col in [c for c in df.columns if c != "time_s"]:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = np.isfinite(time_s) & np.isfinite(values)
        if mask.sum() < 2:
            continue
        stats = _compute_energy_stats(time_s[mask], values[mask])
        results.append(
            {
                "file": csv_path.name,
                "run": col,
                "avg_power_w": stats["avg_power_w"],
                "energy_j": stats["energy_j"],
                "energy_wh": stats["energy_wh"],
                "duration_s": stats["duration_s"],
            }
        )

    if not results:
        return [], {}

    avg_power = float(np.mean([r["avg_power_w"] for r in results]))
    avg_energy_j = float(np.mean([r["energy_j"] for r in results]))
    avg_energy_wh = float(np.mean([r["energy_wh"] for r in results]))
    avg_duration = float(np.mean([r["duration_s"] for r in results]))
    summary = {
        "avg_power_w": avg_power,
        "avg_energy_j": avg_energy_j,
        "avg_energy_wh": avg_energy_wh,
        "avg_duration_s": avg_duration,
        "runs": float(len(results)),
    }
    return results, summary


def main(args: Args) -> None:
    path = args.path
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        output_dir = path
    elif path.is_file() and path.suffix.lower() == ".csv":
        csv_paths = [path]
        output_dir = path.parent
    else:
        print(f"❌ Path not found or not a CSV/directory: {path}")
        return

    if not csv_paths:
        print(f"⚠️  No CSV files found in: {path}")
        return

    lines: list[str] = ["Experiment Energy Summary"]
    any_results = False

    for csv_path in csv_paths:
        per_run, summary = _analyze_metric_csv(csv_path)
        if not per_run:
            continue
        any_results = True
        lines.append("")
        lines.append(f"{csv_path.name}:")
        lines.append(f"  runs: {int(summary['runs'])}")
        lines.append(f"  avg_power_w:   {summary['avg_power_w']:.3f} W")
        lines.append(f"  avg_energy_j:  {summary['avg_energy_j']:.3f} J")
        lines.append(f"  avg_energy_wh: {summary['avg_energy_wh']:.6f} Wh")
        lines.append(f"  avg_duration_s:{summary['avg_duration_s']:.3f} s")
        for r in per_run:
            lines.append(f"  {r['run']}")
            lines.append(f"    avg_power_w: {r['avg_power_w']:.3f} W")
            lines.append(f"    energy_j:    {r['energy_j']:.3f} J")
            lines.append(f"    energy_wh:   {r['energy_wh']:.6f} Wh")
            lines.append(f"    duration_s:  {r['duration_s']:.3f} s")

    if not any_results:
        print("⚠️  No usable data found in CSVs.")
        return

    output_text = "\n".join(lines) + "\n"
    print(output_text, end="")

    output_path = output_dir / "experiment_energy_summary.txt"
    output_path.write_text(output_text)
    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
