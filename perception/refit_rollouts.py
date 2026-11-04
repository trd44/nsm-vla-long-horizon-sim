"""Refit the regressor on teleported scenes plus rollout rows from collect_rollouts.py.

Evaluation holds out whole rollout episodes, chosen by a fixed rule on the seed, and a
scene's seed is shared between its vision and gt runs so a held-out start state never
appears in training. Sample weight and hyperparameters are tuned by GroupKFold on the
training episodes only. The saved model is then refit on all data, so the held-out
numbers describe the fit that excluded those episodes. Only the regressor's inputs are
features; GT is a label, a row filter and a reporting key.

    python perception/refit_rollouts.py --teleport DIR --rollouts DIR \
        --out models/regressors/nsm_regressor_rollout.pkl \
        --report models/regressors/nsm_regressor_rollout_report.json
"""
import argparse
import collections
import glob
import json
import os
import time
import warnings

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from perception.model import fit_ray_cascade, predict_xyz
from perception.train_regressor import build_rows, load_scenes

warnings.filterwarnings("ignore")
MIN_TRAIN_IOU = 0.15  # as match_boxes_to_cubes: anything lower may be the wrong object
WEIGHTS = [1.0, 3.0, 10.0]
GRID = [dict(max_iter=600, max_leaf_nodes=63, learning_rate=0.03, min_samples_leaf=30),
        dict(max_iter=400, max_leaf_nodes=31, learning_rate=0.05)]


def held_out(seed):
    return seed % 3 == 1


def load_rollouts(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r["seed"] < 2000:
                    raise SystemExit(f"{f}: seed {r['seed']} is not a training seed")
                rows.append(r)
    return rows


def stats(err):
    err = np.asarray(err) * 1e3
    if not len(err):
        return dict(n=0)
    xy, z = np.linalg.norm(err[:, :2], axis=1), np.abs(err[:, 2])
    pct = lambda v, q: round(float(np.percentile(v, q)), 1)
    return dict(n=int(len(err)), xy_med=pct(xy, 50), xy_p90=pct(xy, 90), xy_p99=pct(xy, 99), z_med=pct(z, 50), z_p90=pct(z, 90),
                z_p99=pct(z, 99), median_signed_mm=np.round(np.median(err, axis=0), 1).tolist())


def score(s):
    return s["xy_med"] + 0.2 * s["xy_p99"] + 0.3 * s["z_med"]


def vis_bin(r):
    b = r.get("box_a")
    area = (b[2] - b[0]) * (b[3] - b[1]) if b else 0
    f = r["vis_a"] / area if area else 0.0
    return "vis<0.25" if f < 0.25 else "vis<0.5" if f < 0.5 else "vis>=0.5"


def by_group(rows, err):
    keys = collections.defaultdict(list)
    for r, e in zip(rows, err):
        lv = f"level{r['level']}"
        for k in ("all", f"task/{r['task']}", f"level/{lv}", f"task_level/{r['task']}/{lv}", f"perception/{r['perception']}",
                  f"task_perception/{r['task']}/{r['perception']}", f"visibility/{vis_bin(r)}", f"name/{r['task']}/{r['name']}/{lv}",
                  f"wrist/{'visible' if r['x'][8] > 0 else 'missing'}"):
            keys[k].append(e)
    return {k: stats(v) for k, v in sorted(keys.items())}


def stabilized(rows, pred):
    """Executor-style estimate: median of readings with the arm clear (> 8cm, as
    stable_estimate_radius, measured from the prediction as the executor does) over each
    span the cube sat still. One value per span, so errors are per placement not per frame."""
    spans = collections.defaultdict(list)
    idx = {}
    for i, r in enumerate(rows):
        key = (r["task"], r["perception"], r["seed"], r["name"])
        g = np.array(r["gt"])
        if key not in idx or np.linalg.norm(idx[key][0] - g) > 0.003:
            idx[key] = (g, idx[key][1] + 1 if key in idx else 0)
        spans[(*key, idx[key][1])].append(i)
    out, meta = [], []
    for k, ids in spans.items():
        use = [i for i in ids if np.linalg.norm(pred[i][:2] - np.array(rows[i]["x"][11:13])) > 0.08]
        if len(use) >= 3:
            med = np.median(pred[use], axis=0)
            out.append(med - np.array(rows[use[0]]["gt"]))
            meta.append(rows[use[0]])
    return meta, np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teleport", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--baseline", default="models/regressors/nsm_regressor.pkl")
    p.add_argument("--out", default="models/regressors/nsm_regressor_rollout.pkl")
    p.add_argument("--report", default="models/regressors/nsm_regressor_rollout_report.json")
    p.add_argument("--tune_folds", type=int, default=3)
    args = p.parse_args()
    t0 = time.time()

    roll = load_rollouts(args.rollouts)
    tele, tele_stats = build_rows(load_scenes(args.teleport))
    tele = [r for r in tele if r["assoc"] in ("oracle", "deploy") and r["train"]]
    tele_groups = np.array([r["scene"] for r in tele])
    Xt = np.array([r["x"] for r in tele], float)
    Yt = np.array([r["gt"] for r in tele], float)
    print("teleport rows", len(tele), "rollout rows", len(roll), flush=True)

    Xr = np.array([r["x"] for r in roll], float)
    Yr = np.array([r["gt"] for r in roll], float)
    Gr = np.array([r["group"] for r in roll])
    trust = np.array([r["iou_a"] >= MIN_TRAIN_IOU for r in roll])
    test = np.array([held_out(r["seed"]) for r in roll])
    print("rollout train", int((~test & trust).sum()), "held out", int(test.sum()), "low iou", int((~trust).sum()), flush=True)
    report = dict(teleport_rows=len(tele), teleport_filter=dict(tele_stats), rollout_rows=len(roll),
                  rollout_rows_low_iou_untrained=int((~trust).sum()), held_out_groups=sorted(set(Gr[test])), train_groups=sorted(set(Gr[~test])),
                  rows_by_task_perception=collections.Counter(f"{r['task']}/{r['perception']}" for r in roll))

    # Teleport held-out scenes: a guard for the families rollouts do not cover (sorting).
    us = np.unique(tele_groups)
    tr_s, te_s = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(us, groups=us))
    te_set = set(us[te_s])
    tele_test = np.array([g in te_set for g in tele_groups])

    def stack(mask_t, mask_r, w):
        X = np.vstack([Xt[mask_t], Xr[mask_r]])
        Y = np.vstack([Yt[mask_t], Yr[mask_r]])
        G = np.r_[tele_groups[mask_t], Gr[mask_r]]
        W = np.r_[np.ones(int(mask_t.sum())), np.full(int(mask_r.sum()), w)]
        return X, Y, G, W

    # Tuning: GroupKFold on rollout training episodes only; teleport test scenes stay out.
    idx = np.where(~test & trust)[0]
    folds = list(GroupKFold(args.tune_folds).split(idx, groups=Gr[idx]))
    tuning, best, best_s = [], None, np.inf
    for params in GRID:
        for w in WEIGHTS:
            if params is not GRID[0] and w != 3.0:  # grid over params only at a middle weight
                continue
            scores = []
            for tr, te in folds:
                mask_r = np.zeros(len(roll), bool)
                mask_r[idx[tr]] = True
                X, Y, G, W = stack(~tele_test, mask_r, w)
                mdl = fit_ray_cascade(X, Y, G, params, agent_only=False, sample_weight=W)
                scores.append(stats(predict_xyz(mdl, Xr[idx[te]]) - Yr[idx[te]]))
            s = np.mean([score(v) for v in scores])
            tuning.append(dict(params=params, weight=w, score=round(float(s), 2), folds=scores))
            print("tune", params, w, round(float(s), 2), f"{time.time() - t0:.0f}s", flush=True)
            if s < best_s:
                best, best_s = (params, w), s
    params, weight = best
    report.update(tuning=tuning, chosen_params=params, chosen_rollout_weight=weight)
    print("chosen", best, flush=True)

    # Held-out comparison. Baseline on the same rows; 'teleport_only' is the same design
    # without rollout rows, showing what those rows contribute.
    X, Y, G, W = stack(~tele_test, ~test & trust, weight)
    candidate = fit_ray_cascade(X, Y, G, params, agent_only=False, sample_weight=W)
    teleport_only = fit_ray_cascade(Xt[~tele_test], Yt[~tele_test], tele_groups[~tele_test], params, agent_only=False)
    baseline = joblib.load(args.baseline)
    Rte = [r for r, k in zip(roll, test) if k]
    Xte, Yte = Xr[test], Yr[test]
    compare = {}
    for name, mdl in (("baseline", baseline), ("rollout_fit", candidate), ("teleport_only_refit", teleport_only)):
        pred = predict_xyz(mdl, Xte)
        meta, serr = stabilized(Rte, pred)
        compare[name] = dict(rollout_held_out=by_group(Rte, pred - Yte), rollout_held_out_stabilized=by_group(meta, serr),
                             teleport_held_out=stats(predict_xyz(mdl, Xt[tele_test]) - Yt[tele_test]))
        print(name, compare[name]["rollout_held_out"]["all"], "\n  stabilized", compare[name]["rollout_held_out_stabilized"]["all"],
              "\n  teleport", compare[name]["teleport_held_out"], flush=True)
    report["compare"] = compare
    report["note_teleport_guard"] = "baseline was trained on these teleport scenes, so its teleport_held_out is optimistic; rollout_fit and teleport_only_refit were not."

    X, Y, G, W = stack(np.ones(len(tele), bool), trust, weight)
    final = fit_ray_cascade(X, Y, G, params, agent_only=False, sample_weight=W)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({**final, "meta": dict(features=14, version="nsm_v1_rollout", model="ray_cascade", teleport_rows=len(tele),
                                       rollout_rows=int(trust.sum()), rollout_weight=weight, params=params)}, args.out, compress=3)
    report["runtime_s"] = round(time.time() - t0)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else float(o))
    print("wrote", args.out, args.report, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
