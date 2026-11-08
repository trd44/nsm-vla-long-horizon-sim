"""Train the NSM cube detector by distillation from OWLv2 pseudo-labels, and evaluate it.

Paper section 3.4.2: "we train a lightweight YOLOv8 bounding-box detector ... via
knowledge distillation from OWLv2 detections". The dataset comes from
perception/owl_label.py (``build``); every training and validation label there is an
OWLv2 detection. Simulator GT is read only by ``evaluate``, on held-out scene groups
and episodes (split "test"), to compare:

  (a) OWLv2 itself (the teacher, decoded exactly as for the labels),
  (b) the distilled YOLO,
  (c) the existing models/yolo/hanoi_yolo.pt,

per class, camera and situation (recall at IoU 0.5, precision, matched IoU), plus the
3D error of the position regressor fed with each detector's boxes. Images are the
executor's preprocessed frames (flipped upright, BGR), predicted with the same
ultralytics defaults the executor uses (conf 0.25, the model's own imgsz).

    python perception/train_yolo_distill.py train --dataset DATASET --out models/yolo/nsm_yolo_owl.pt
    python perception/train_yolo_distill.py evaluate --data DATA --yolo models/yolo/nsm_yolo_owl.pt \
        --report models/yolo/nsm_yolo_owl_report.json
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from perception.owl_label import CAMS, CLASS_NAMES, IMG, LABEL_CFG, det_metrics, load_frames, load_gt, load_owl, \
    match_frame, print_metrics

MAX_AGENT_PX = 40  # Executor_Diffusion.max_agentview_box_px: larger agentview boxes are dropped
OLD_YOLO = "models/yolo/hanoi_yolo.pt"
REGRESSOR = "models/regressors/nsm_regressor_rollout.pkl"


def run_train(args):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite or another --out")
    if os.path.abspath(args.out) == os.path.abspath(OLD_YOLO):
        raise SystemExit("refusing to overwrite the existing hanoi_yolo.pt")
    data_yaml = os.path.join(args.dataset, "data.yaml")
    model = YOLO(args.model)
    t0 = time.time()
    # Colour is the class, so hue augmentation stays at a small default and there is
    # no flip-up-down (the executor always sees upright frames).
    res = model.train(data=data_yaml, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
                      workers=args.workers, project=args.project, name=args.name, exist_ok=True, seed=0,
                      patience=args.patience, hsv_h=0.01, hsv_s=0.5, hsv_v=0.3, fliplr=0.5, flipud=0.0,
                      mosaic=1.0, close_mosaic=2, plots=False, verbose=False, amp=False)
    train_s = time.time() - t0
    best = os.path.join(args.project, args.name, "weights", "best.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy(best, args.out)
    build = json.load(open(os.path.join(args.dataset, "build.json")))
    info = dict(model=args.model, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
                train_seconds=round(train_s), dataset=os.path.abspath(args.dataset), owl_label_build=build,
                val_metrics_vs_owl_labels={k: float(v) for k, v in getattr(res, "results_dict", {}).items()})
    json.dump(info, open(os.path.splitext(args.out)[0] + "_train.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def yolo_dets(model_path, frames, data, keys, batch=16):
    """Run a YOLO model as the executor does (CPU, ultralytics defaults). {key: dets}."""
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(model_path)
    out, times = {}, []
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        ims = [cv2.imread(os.path.join(data, "frames", f"{k}.png")) for k in chunk]
        t0 = time.time()
        res = model.predict(ims, verbose=False, device="cpu")
        times.append((time.time() - t0) / len(chunk))
        torch.set_num_threads(1)
        for k, r in zip(chunk, res):
            d = []
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                d.append([int(b.cls), float(b.conf), x0, y0, x1, y1])
            out[k] = d
    return out, model.names


def executor_filter(dets, tag):
    """Drop agentview boxes the executor ignores (max side over MAX_AGENT_PX)."""
    if tag != "a":
        return dets
    return [d for d in dets if max(d[4] - d[2], d[5] - d[3]) <= MAX_AGENT_PX]


def to_xywhn(d):
    """[cls, conf, x0, y0, x1, y1] -> [cls, conf, cx, cy, w, h] normalised (features.feature_row)."""
    c, conf, x0, y0, x1, y1 = d
    return [c, conf, (x0 + x1) / 2 / IMG, (y0 + y1) / 2 / IMG, (x1 - x0) / IMG, (y1 - y0) / IMG]


def regressor_errors(frames, gt, dets, regressor, sel):
    """3D error per resting cube the detector found in agentview.

    Agent box: the detection matched to the cube (IoU >= 0.5 against GT, used only to
    say WHICH cube a box is). Wrist box: the class's highest-confidence wrist box, as
    the executor takes it. Returns {(fid, cube): (error_m, situation)}."""
    from perception.features import feature_row
    from perception.model import predict_xyz

    rows, keys = [], []
    for fid, f in frames.items():
        if not sel(f):
            continue
        ka, kw = f"{fid}_a", f"{fid}_w"
        if ka not in dets:
            continue
        pairs, _ = match_frame(dets[ka], gt[fid], "a")
        wrist = sorted(dets.get(kw, []), key=lambda d: -d[1])
        for c, d, _ in pairs:
            if d is None or c["sit"] in ("in_hand", "airborne"):
                continue
            w = next((x for x in wrist if x[0] == c["cls"]), None)
            rows.append(feature_row(c["cls"], to_xywhn(d), to_xywhn(w) if w else None, f["ee"]))
            keys.append((fid, c["name"], c["sit"], tuple(c["pos"])))
    if not rows:
        return {}
    pred = predict_xyz(regressor, np.array(rows, dtype=float))
    return {(fid, name): (float(np.linalg.norm(p - np.array(pos))), sit)
            for (fid, name, sit, pos), p in zip(keys, pred)}


def err_stats(errs):
    if not errs:
        return dict(n=0)
    e = np.array(errs) * 1000
    return dict(n=len(e), median_mm=round(float(np.median(e)), 2), mean_mm=round(float(e.mean()), 2),
                p90_mm=round(float(np.percentile(e, 90)), 2), frac_under_15mm=round(float((e < 15).mean()), 4))


def run_evaluate(args):
    import joblib
    import torch

    torch.set_num_threads(args.threads)
    frames = load_frames(args.data)
    gt = load_gt(args.data)
    frames = {k: f for k, f in frames.items() if f["split"] == args.split and k in gt}
    keys = [f"{fid}_{tag}" for fid in sorted(frames) for tag, _ in CAMS]
    print(f"{len(frames)} {args.split} frames, {len(keys)} images")

    detectors = {}
    owl = load_owl(args.data)
    detectors["owlv2"] = {k: owl[k] for k in keys if k in owl}
    timing = {}
    for name, path in (("distilled", args.yolo), ("hanoi_yolo", args.old_yolo)):
        if path:
            d, names = yolo_dets(path, frames, args.data, keys)
            assert names == CLASS_NAMES, (path, names)
            detectors[name] = {k: executor_filter(v, k[-1]) for k, v in d.items()}

    report = dict(split=args.split, n_frames=len(frames), owl_label_cfg=LABEL_CFG, max_agent_px=MAX_AGENT_PX,
                  models=dict(distilled=args.yolo, hanoi_yolo=args.old_yolo, regressor=args.regressor),
                  detection={}, detection_projected_gt={}, regressor={})
    for name, d in detectors.items():
        m = det_metrics(frames, gt, d)
        print_metrics(m, f"{name} vs GT (visible-mask boxes), split={args.split}")
        report["detection"][name] = m
        # Same with the full projected cube box as GT (comparable to earlier recall numbers).
        gt_proj = {fid: [{**c, "box_a": c["proj_a"] if c["box_a"] else None,
                          "box_w": c["proj_w"] if c["box_w"] else None} for c in cubes] for fid, cubes in gt.items()}
        report["detection_projected_gt"][name] = det_metrics(frames, gt_proj, d)

    # Latency of each YOLO on one CPU thread, single image as in the executor.
    import cv2
    from ultralytics import YOLO
    sample = [cv2.imread(os.path.join(args.data, "frames", f"{k}.png")) for k in keys[:40]]
    for name, path in (("distilled", args.yolo), ("hanoi_yolo", args.old_yolo)):
        if not path:
            continue
        m = YOLO(path)
        m.predict(sample[0], verbose=False, device="cpu")
        torch.set_num_threads(1)
        t0 = time.time()
        for im in sample:
            m.predict(im, verbose=False, device="cpu")
        timing[name] = dict(ms_per_image_1thread=round(1000 * (time.time() - t0) / len(sample), 1),
                            imgsz=m.overrides.get("imgsz"), params=sum(p.numel() for p in m.model.parameters()))
    report["timing"] = timing
    print(json.dumps(timing, indent=1))

    reg = joblib.load(args.regressor)
    errs = {name: regressor_errors(frames, gt, d, reg, lambda f: True) for name, d in detectors.items()}
    common = set.intersection(*[set(e) for e in errs.values()])
    for name, e in errs.items():
        r = dict(own=err_stats([v[0] for v in e.values()]),
                 common=err_stats([e[k][0] for k in common]))
        for sit in sorted({v[1] for v in e.values()}):
            r[f"common/{sit}"] = err_stats([e[k][0] for k in common if e[k][1] == sit])
        for src in ("scene", "rollout"):
            r[f"common/{src}"] = err_stats([e[k][0] for k in common if frames[k[0]]["source"] == src])
        report["regressor"][name] = r
        print(f"regressor 3D error with {name}: {json.dumps(r)}")
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
        print("wrote", args.report)


def run_refit(args):
    """Refit the position regressor on one detector's boxes (train + val frames).

    GT positions are regression targets, as for every regressor in this repo; GT boxes
    only say which cube a detection belongs to. Saved as a separate model file."""
    import joblib
    import torch

    from perception.features import feature_row
    from perception.model import fit_ray_cascade

    torch.set_num_threads(args.threads)
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists")
    frames = load_frames(args.data)
    gt = load_gt(args.data)
    frames = {k: f for k, f in frames.items() if f["split"] in ("train", "val") and k in gt}
    keys = [f"{fid}_{tag}" for fid in sorted(frames) for tag, _ in CAMS]
    d, _ = yolo_dets(args.yolo, frames, args.data, keys)
    d = {k: executor_filter(v, k[-1]) for k, v in d.items()}
    X, Y, G = [], [], []
    for fid, f in frames.items():
        pairs, _ = match_frame(d[f"{fid}_a"], gt[fid], "a")
        wrist = sorted(d.get(f"{fid}_w", []), key=lambda x: -x[1])
        for c, det, _ in pairs:
            if det is None or c["sit"] in ("in_hand", "airborne"):
                continue
            w = next((x for x in wrist if x[0] == c["cls"]), None)
            X.append(feature_row(c["cls"], to_xywhn(det), to_xywhn(w) if w else None, f["ee"]))
            Y.append(c["pos"])
            G.append(f"{f['source']}-{f['group']}")
    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    params = dict(max_iter=600, max_leaf_nodes=63, learning_rate=0.03, min_samples_leaf=30)  # nsm_regressor_rollout's
    model = fit_ray_cascade(X, Y, np.array(G), params, agent_only=False)
    model["meta"] = dict(detector=args.yolo, rows=len(X), groups=len(set(G)), params=params,
                         data=os.path.abspath(args.data), note="refit on this detector's boxes; see train_yolo_distill.py")
    joblib.dump(model, args.out, compress=3)
    print(f"fit on {len(X)} rows from {len(set(G))} groups -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True, help="directory written by owl_label.py build")
    t.add_argument("--model", default="yolov8n.pt")
    t.add_argument("--imgsz", type=int, default=416)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--patience", type=int, default=3)
    # MPS training diverges with ultralytics 8.4.202 / torch 2.4 (cls loss 8 -> 20+ within an
    # epoch, while the same run on CPU falls to ~1), so training defaults to the CPU.
    t.add_argument("--device", default="cpu")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--threads", type=int, default=3)
    t.add_argument("--project", required=True, help="ultralytics run directory (scratch)")
    t.add_argument("--name", default="nsm_yolo_owl")
    t.add_argument("--out", default="models/yolo/nsm_yolo_owl.pt")
    t.add_argument("--overwrite", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True, help="owl_label.py data directory (frames, owl_raw, gt)")
    e.add_argument("--split", default="test")
    e.add_argument("--yolo", default="models/yolo/nsm_yolo_owl.pt")
    e.add_argument("--old_yolo", default=OLD_YOLO)
    e.add_argument("--regressor", default=REGRESSOR)
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--report", default=None)
    r = sub.add_parser("refit")
    r.add_argument("--data", required=True)
    r.add_argument("--yolo", default="models/yolo/nsm_yolo_owl.pt")
    r.add_argument("--out", default="models/regressors/nsm_regressor_owl.pkl")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    {"train": run_train, "evaluate": run_evaluate, "refit": run_refit}[args.cmd](args)


if __name__ == "__main__":
    main()
