"""Merge VLA seed chunks into one file per condition, and print the comparison table.

  python scripts/vla_eval/merge.py --results results/vla_5000 --task hanoi_4 --checkpoint hanoi_3
  python scripts/vla_eval/merge.py --results results/vla_5000 --table
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from planning.task_configs import PAPER_TASKS, summarize  # noqa: E402

# Paper Tables 3 and 4, VLA columns: (success %, advancement %).
PAPER_VLA = {
    ("sort_2color", "finetuned"): (100.0, 100.0),
    ("tower_3_set", "finetuned"): (98.0, 98.7),
    ("hanoi_3", "finetuned"): (90.0, 92.4),
    ("sort_3color", "finetuned"): (2.0, 42.0),
    ("tower_3_random", "finetuned"): (8.0, 12.0),
    ("tower_4_set", "finetuned"): (88.0, 92.5),
    ("hanoi_4", "finetuned"): (78.0, 92.4),
    ("sort_2color_swapped", "zeroshot"): (0.0, 0.0),
    ("sort_3color", "zeroshot"): (4.0, 68.0),
    ("tower_3_random", "zeroshot"): (12.0, 20.0),
    ("tower_4_set", "zeroshot"): (2.0, 30.3),
    ("hanoi_4", "zeroshot"): (0.0, 0.0),
}


def merge(results, task, checkpoint, seed, episodes):
    cond = "finetuned" if task == checkpoint else "zeroshot"
    label = f"{task}_vla_{cond}_{checkpoint}"
    records = []
    for path in sorted(glob.glob(os.path.join(results, "chunks", f"{label}_seed*.jsonl"))):
        records += [json.loads(line) for line in open(path)]
    records = sorted({r["seed"]: r for r in records}.values(), key=lambda r: r["seed"])
    expected = set(range(seed, seed + episodes))
    missing = sorted(expected - {r["seed"] for r in records})
    with open(os.path.join(results, f"{label}_seed{seed}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, PAPER_TASKS[task])
    paper = PAPER_VLA.get((task, cond))
    summary.update(condition=cond, checkpoint_key=checkpoint, seeds=[seed, seed + episodes - 1],
                   missing_seeds=missing, paper_vla_success_pct=paper and paper[0],
                   paper_vla_advancement_pct=paper and paper[1])
    if records:
        summary.update(checkpoint=records[0]["checkpoint"], server_config=records[0]["server_config"],
                       prompt=records[0]["prompt"])
    with open(os.path.join(results, f"{label}_seed{seed}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{label}: {summary['successes']}/{summary['episodes']} success, "
          f"{summary['advancement_pct']:.1f}% advancement" + (f"; MISSING seeds {missing}" if missing else ""))


def table(results):
    nsm = {}
    nsm_dir = os.path.join(os.path.dirname(os.path.abspath(results)), "nsm_final_5000", "vision")
    for path in glob.glob(os.path.join(nsm_dir, "*.jsonl")):
        task = os.path.basename(path).split("_vision_")[0]
        for line in open(path):
            nsm.setdefault(task, []).append(json.loads(line)["success"])
    print("| Task | Condition | VLA success (Wilson 95%) | VLA adv. | Paper VLA success / adv. | NSM vision success |")
    print("|---|---|---|---|---|---|")
    for (task, cond), paper in PAPER_VLA.items():
        paths = glob.glob(os.path.join(results, f"{task}_vla_{cond}_*_seed*.summary.json"))
        if not paths:
            print(f"| `{task}` | {cond} | not run | | {paper[0]:g} / {paper[1]:g} | |")
            continue
        s = json.load(open(paths[0]))
        lo, hi = s["success_wilson95_pct"]
        n = nsm.get(task)
        nsm_txt = f"{100 * sum(n) / len(n):.0f}" if n else "-"
        print(f"| `{task}` | {cond} | {s['success_pct']:.0f} ({lo:.0f}-{hi:.0f}), n={s['episodes']} | "
              f"{s['advancement_pct']:.1f} | {paper[0]:g} / {paper[1]:g} | {nsm_txt} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--task")
    p.add_argument("--checkpoint")
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    if a.table:
        table(a.results)
    else:
        merge(a.results, a.task, a.checkpoint, a.seed, a.episodes)
