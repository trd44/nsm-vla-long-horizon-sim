"""OWLv2 pseudo-labels for distilling the NSM cube detector (paper sections 3.4.2, 4.4.2).

The paper trains a lightweight YOLOv8 detector by knowledge distillation from OWLv2
detections queried with the object names as text prompts. This module produces the
teacher side of that: it renders frames from both cameras, runs OWLv2 on them and
writes a YOLO-format dataset whose labels come from OWLv2 alone.

Simulator ground truth is written to a separate gt.jsonl. It is read ONLY by
``evaluate`` (label quality per class/camera/situation) and by the evaluation in
train_yolo_distill.py. ``build`` never opens it, so no training label is created,
filtered or corrected with simulator state.

Images are stored exactly as the executor feeds them to YOLO: resized to 256x256,
flipped vertically to upright and converted RGB->BGR (cv2.imwrite of that array gives
a correctly coloured PNG, which ultralytics reads back as BGR).

Stages (DATA is a scratch directory; frames are about 70 KB each):

    # teleport scenes: cubes placed in sorting bins, towers, stacks; arm anywhere
    python perception/owl_label.py scenes --data DATA --groups 2100-2399 --per_group 5
    # frames hooked from real task execution (gt perception drives the robot; only images kept)
    python perception/owl_label.py rollouts --data DATA --task hanoi_4 --seed 2600 --episodes 4
    # OWLv2 teacher (MPS or CPU)
    python perception/owl_label.py label --data DATA --device mps
    # pseudo-labels -> YOLO dataset (never reads gt.jsonl)
    python perception/owl_label.py build --data DATA --out DATASET
    # label quality against simulator GT (evaluation only)
    python perception/owl_label.py evaluate --data DATA --split test

Seeds: only 2100-2999 are used for data. Held-out (test) frames are whole scene
groups / episodes: teleport groups 2500-2599 and rollout episodes 2900-2999.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np

IMG = 256
CAMS = (("a", "agentview"), ("w", "robot0_eye_in_hand"))
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
CLASS_NAMES = {0: "blue cube", 1: "green cube", 2: "red cube", 3: "yellow cube"}
COLORS = {0: "blue", 1: "green", 2: "red", 3: "yellow"}

SEED_RANGE = (2100, 2999)
TEST_SCENE_GROUPS = (2500, 2599)
TEST_ROLLOUT_SEEDS = (2900, 2999)
# A few training groups are kept aside as YOLO validation (pseudo-labels only).
VAL_EVERY = 10

# Text queries; each class's phrasings are max-pooled. Distractor queries ("a bin",
# "a robot gripper", ...) were tried and made no difference, so none are used.
PROMPTS = {c: [f"a {n} cube", f"a {n} block"] for c, n in COLORS.items()}
DISTRACTORS = []

# Label decoding (``decode``), chosen with ``tune`` on training frames only.
#   color: "off"     - class is OWL's best prompt (pure OWL, as in the paper)
#          "verify"  - additionally require that colour's pixels to cover min_frac of the box
#          "recolor" - class is the box's dominant cube colour; OWL must score that
#                      colour's prompt >= score_thr. Fixes OWL calling the olive-rendered
#                      yellow cube "green". No simulator state is involved.
#          "recolor_any" - as recolor, but the score is OWL's best colour prompt (any
#                      colour), so a cube OWL found under the wrong colour name is kept.
LABEL_CFG = dict(
    score_thr=0.10,      # min max-pooled class score
    margin=-1.0,         # class score must beat the best distractor score by this (-1: off)
    nms_iou=0.3,         # class-agnostic NMS between cube boxes
    max_side_a=40,       # px; agentview cubes are 10-25 px, larger boxes are bins/arm
    min_side=3,          # px
    max_aspect=2.5,
    color="recolor_any",
    min_frac=0.10,
)
# OpenCV HSV hue ranges (0-179) of the four rendered cube colours, saturation > 80.
HUE_RANGES = {0: [(105, 135)], 1: [(45, 85)], 2: [(0, 10), (170, 180)], 3: [(18, 40)]}


def color_fractions(image_bgr, box):
    """Fraction of the box covered by each cube colour (saturated pixels in hue range)."""
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    patch = image_bgr[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    if patch.size == 0:
        return np.zeros(4)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ok = (hsv[:, 1] > 80) & (hsv[:, 2] > 40)
    h = hsv[:, 0]
    return np.array([sum(((h >= a) & (h < b) & ok).sum() for a, b in HUE_RANGES[c]) for c in range(4)]) / len(h)


def frame_split(source, group):
    if source == "scene":
        if TEST_SCENE_GROUPS[0] <= group <= TEST_SCENE_GROUPS[1]:
            return "test"
    elif TEST_ROLLOUT_SEEDS[0] <= group <= TEST_ROLLOUT_SEEDS[1]:
        return "test"
    return "val" if group % VAL_EVERY == 0 else "train"


def check_seed(seed):
    if not SEED_RANGE[0] <= seed <= SEED_RANGE[1]:
        raise SystemExit(f"seed {seed} outside the data-generation range {SEED_RANGE}")


def proc(im, cv2):
    """Executor preprocessing: resize, flip upright, RGB->BGR."""
    return cv2.cvtColor(cv2.flip(cv2.resize(np.asarray(im, dtype=np.uint8), (IMG, IMG)), 0), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Simulator ground truth (evaluation only)
# ---------------------------------------------------------------------------

def cube_meta(sim):
    """{cube: gid, half, cls, body, joint} from geoms and materials."""
    from perception.features import TEXTURE_TO_CLS

    m = sim.model
    out = {}
    for j in m.joint_names:
        g = re.fullmatch(r"(cube\d+)_joint0", j)
        if not g:
            continue
        c = g.group(1)
        gid = m.geom_name2id(f"{c}_g0_vis")
        import mujoco
        mat = mujoco.mj_id2name(m._model, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[gid]))
        num = int(re.search(r"number(\d)", mat).group(1))
        out[c] = dict(gid=gid, half=float(m.geom_size[gid][0]), cls=TEXTURE_TO_CLS[num],
                      body=m.body_name2id(f"{c}_main"), joint=j)
    return out


def bin_regions(sim):
    m = sim.model
    out = []
    for b in m.body_names:
        if re.fullmatch(r"(bin|platform)\d_main", b):
            gid = m.geom_name2id(b.replace("_main", "_g0_vis"))
            out.append((np.array(sim.data.body_xpos[m.body_name2id(b)]), np.array(m.geom_size[gid])))
    return out


def situations(sim, meta, grasped=(), ee=None):
    """Per-cube situation tag from simulator poses: in_hand, in_bin, stacked, base, airborne.

    ``covered`` marks a cube with another cube on top; ``arm_near`` an end effector
    within 6 cm in xy and 15 cm above (agent-view occlusion is likely)."""
    pos = {c: np.array(sim.data.body_xpos[i["body"]]) for c, i in meta.items()}
    bins = bin_regions(sim)
    out = {}
    for c, p in pos.items():
        h = meta[c]["half"]
        below = [o for o, q in pos.items() if o != c and np.linalg.norm(q[:2] - p[:2]) < h + meta[o]["half"]
                 and abs((p[2] - q[2]) - (h + meta[o]["half"])) < 0.012]
        above = [o for o, q in pos.items() if o != c and np.linalg.norm(q[:2] - p[:2]) < h + meta[o]["half"]
                 and abs((q[2] - p[2]) - (h + meta[o]["half"])) < 0.012]
        in_bin = any(np.all(np.abs(p[:2] - bp[:2]) < bs[:2]) and p[2] < bp[2] + bs[2] + 2 * h + 0.06 for bp, bs in bins)
        if c in grasped:
            sit = "in_hand"
        elif in_bin:
            sit = "in_bin"
        elif below:
            sit = "stacked"
        elif p[2] < 0.86 + h:  # table top is 0.8; HeightStacking platform tops at about 0.85
            sit = "base"
        else:
            sit = "airborne"
        near = ee is not None and np.linalg.norm(np.asarray(ee)[:2] - p[:2]) < 0.06 and 0 < ee[2] - p[2] < 0.15
        out[c] = dict(sit=sit, covered=bool(above), arm_near=bool(near))
    return out


def gt_record(sim, meta, grasped=(), ee=None):
    """Mask-bbox and projected boxes (upright pixels) per cube and camera."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    from perception.collect_data import cube_box

    sits = situations(sim, meta, grasped, ee)
    segs = {}
    for tag, cam in CAMS:
        s = sim.render(camera_name=cam, width=IMG, height=IMG, depth=False, segmentation=True)[::-1]
        segs[tag] = (s[..., 0], s[..., 1], get_camera_transform_matrix(sim, cam, IMG, IMG))
    rows = []
    for c, info in meta.items():
        rec = dict(name=c, cls=info["cls"], pos=[round(float(v), 5) for v in sim.data.body_xpos[info["body"]]],
                   **sits[c])
        for tag, (typ, sid, M) in segs.items():
            mask = (typ == 5) & (sid == info["gid"])
            n = int(mask.sum())
            rec[f"vis_{tag}"] = n
            if n:
                ys, xs = np.nonzero(mask)
                rec[f"box_{tag}"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            else:
                rec[f"box_{tag}"] = None
            pb = cube_box(sim, dict(body=info["body"], half=info["half"]), M, np)
            rec[f"proj_{tag}"] = [round(v, 2) for v in pb] if pb else None
        if rec["vis_a"] or rec["vis_w"]:
            rows.append(rec)
    return rows


class FrameWriter:
    """Writes frame PNGs, frames.jsonl (no GT) and gt.jsonl (evaluation only)."""

    def __init__(self, data, tag):
        import cv2
        self.cv2 = cv2
        self.data = data
        os.makedirs(os.path.join(data, "frames"), exist_ok=True)
        self.f_meta = open(os.path.join(data, f"frames_{tag}.jsonl"), "a")
        self.f_gt = open(os.path.join(data, f"gt_{tag}.jsonl"), "a")

    def write(self, fid, obs, meta, gt):
        for tag, cam in CAMS:
            self.cv2.imwrite(os.path.join(self.data, "frames", f"{fid}_{tag}.png"), proc(obs[f"{cam}_image"], self.cv2))
        self.f_meta.write(json.dumps(dict(id=fid, **meta)) + "\n")
        self.f_gt.write(json.dumps(dict(id=fid, cubes=gt)) + "\n")

    def flush(self):
        self.f_meta.flush()
        self.f_gt.flush()


# ---------------------------------------------------------------------------
# Teleport scenes
# ---------------------------------------------------------------------------

SCENE_KEYS = ["Hanoi", "Hanoi4x3", "HeightStacking3", "HeightStacking4", "CubeSorting", "AssemblyLineSorting"]
# Bins appear with cubes in them only in the sorting tasks; weight those up, since
# that is where the current detector fails most.
SCENE_WEIGHTS = np.array([0.15, 0.18, 0.12, 0.15, 0.20, 0.20])


def run_scenes(args):
    os.environ.setdefault("MUJOCO_GL", "cgl")
    warnings.filterwarnings("ignore")
    import cv2
    import robosuite as suite
    import torch

    from perception.collect_data import SceneEnv, SETTLE_STEPS

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    lo, hi = map(int, args.groups.split("-"))
    groups = [g for g in range(lo, hi + 1) if g % args.workers == args.worker]
    for g in groups:
        check_seed(g)
    ctrl = suite.load_controller_config(default_controller="OSC_POSITION")
    writer = FrameWriter(args.data, f"scenes_{lo}_{hi}_w{args.worker}")
    done = set()
    meta_path = writer.f_meta.name
    if os.path.exists(meta_path):
        done = {json.loads(l)["group"] for l in open(meta_path)}
    keys = {g: SCENE_KEYS[np.random.default_rng(g).choice(len(SCENE_KEYS), p=SCENE_WEIGHTS / SCENE_WEIGHTS.sum())]
            for g in groups}
    env = None
    t0 = time.time()
    n = 0
    for g in sorted(groups, key=lambda g: (keys[g], g)):
        if g in done:
            continue
        key = keys[g]
        rng = np.random.default_rng(g)
        np.random.seed(g)
        if env is None or env.key != key:
            if env is not None:
                env.env.close()
            env = SceneEnv(key, suite, np, ctrl)
        else:
            env.reset()
        sim = env.env.sim
        meta = cube_meta(sim)
        for k in range(args.per_group):
            placed, names = env.arrangement(rng)
            for c in env.cubes:
                if c not in names:
                    env.place(c, np.array([3.0, 3.0, 0.1]))
            for c, (xyz, _, _) in placed.items():
                env.place(c, xyz, rng.uniform(-0.25, 0.25) if rng.random() < 0.3 else 0.0)
            sim.forward()
            mode, target = env.sample_ee(placed, rng)
            env.ik(np.clip(target, [-0.3, -0.4, env.table_z + 0.015], [0.3, 0.4, 1.25]))
            sim.forward()
            for _ in range(SETTLE_STEPS):
                obs, *_ = env.env.step(np.array([0, 0, 0, -1.0]))
            ee = obs["robot0_eef_pos"]
            gt = [r for r in gt_record(sim, {c: meta[c] for c in names}, ee=ee)]
            fid = f"s{g}_{k}"
            writer.write(fid, obs, dict(source="scene", env=key, task=key, group=g, split=frame_split("scene", g),
                                        ee=[round(float(v), 5) for v in ee], ee_mode=str(mode)), gt)
            n += 1
        writer.flush()
        print(f"w{args.worker} group {g} {key}: {n} frames, {n / (time.time() - t0):.2f}/s", flush=True)
    if env is not None:
        env.env.close()


# ---------------------------------------------------------------------------
# Frames from real task execution
# ---------------------------------------------------------------------------

def run_rollouts(args):
    """Drive the task with the driver in gt perception mode and keep every stride-th frame.

    Ground-truth perception only steers the robot to realistic states (cubes in hand,
    in bins, towers, arm occluding). The images are what is kept; GT goes to gt.jsonl."""
    for s in range(args.seed, args.seed + args.episodes):
        check_seed(s)
    sys.argv = ["analysis/experiments_neurosymbolic.py", "--task", args.task, "--perception", "gt",
                "--seed", str(args.seed), "--episodes", str(args.episodes), "--overwrite",
                "--results_dir", os.path.join(args.data, "rollout_results")]
    import analysis.experiments_neurosymbolic as driver

    writer = FrameWriter(args.data, f"roll_{args.task}_{args.seed}")
    state = dict(seed=None, step=0, detector=None, meta=None, n=0, offset=0)
    run_episode = driver.run_episode
    step = driver.DictObs.step

    def episode(env, detector, monitor, task, skills, ep, a):
        state.update(seed=a.seed + ep, step=0, detector=detector, meta=None,
                     offset=int(np.random.default_rng(a.seed + ep).integers(args.stride)))
        out = run_episode(env, detector, monitor, task, skills, ep, a)
        writer.flush()
        print(f"[owl_label] {args.task} seed {state['seed']}: {state['n']} frames so far", flush=True)
        return out

    def logged_step(self, action):
        obs, *rest = step(self, action)
        state["step"] += 1
        if state["seed"] is not None and (state["step"] + state["offset"]) % args.stride == 0:
            det = state["detector"]
            sim = det.env.sim
            if state["meta"] is None:
                state["meta"] = {c: v for c, v in cube_meta(sim).items() if c in det.objects}
            grasped = {c for c in det.objects if det.grasped(c)}
            ee = obs["robot0_eef_pos"]
            gt = gt_record(sim, state["meta"], grasped, ee)
            fid = f"r{args.task}_{state['seed']}_{state['step']}"
            writer.write(fid, obs, dict(source="rollout", env=det.env.__class__.__name__, task=args.task,
                                        group=state["seed"], split=frame_split("rollout", state["seed"]),
                                        ee=[round(float(v), 5) for v in ee], skill=self.monitor.skill), gt)
            state["n"] += 1
        return (obs, *rest)

    driver.run_episode = episode
    driver.DictObs.step = logged_step
    driver.main()


# ---------------------------------------------------------------------------
# OWLv2 teacher
# ---------------------------------------------------------------------------

class OwlTeacher:
    """OWLv2 with per-class text prompts. ``raw`` keeps every box whose best query
    sigmoid score passes ``keep``; ``decode`` turns raw boxes into class labels."""

    def __init__(self, device="cpu", model=OWL_MODEL, prompts=None, distractors=None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(model).to(device).eval()
        self.prompts = prompts or PROMPTS
        self.distractors = DISTRACTORS if distractors is None else distractors
        self.queries, self.q_cls = [], []
        for c in sorted(self.prompts):
            for p in self.prompts[c]:
                self.queries.append(p)
                self.q_cls.append(c)
        for p in self.distractors:
            self.queries.append(p)
            self.q_cls.append(-1)
        self.q_cls = np.array(self.q_cls)
        with torch.no_grad():
            t = self.processor(text=[self.queries], return_tensors="pt")
            self.text = {k: v.to(device) for k, v in t.items()}

    def pixels(self, images_bgr):
        """Owlv2Processor preprocessing in torch: [0,1], pad to square with 0.5 grey,
        bilinear resize to the model size (upsampling, so no anti-aliasing), CLIP
        normalisation. The HF processor does the same with scipy on the CPU, which is
        several times slower than the model on MPS."""
        torch = self.torch
        F = torch.nn.functional
        ip = self.processor.image_processor
        size = ip.size["height"]
        mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)
        out = []
        for im in images_bgr:
            t = torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).to(self.device).permute(2, 0, 1).float() / 255.0
            h, w = t.shape[1:]
            s = max(h, w)
            if h != w:
                t = F.pad(t, (0, s - w, 0, s - h), value=0.5)
            out.append(t)
        x = F.interpolate(torch.stack(out), size=(size, size), mode="bilinear", align_corners=False)
        return (x - mean) / std

    def raw(self, images_bgr, keep=0.05):
        """images_bgr: list of HxWx3 uint8 arrays in the executor's BGR layout.

        Returns per image a list of [x0, y0, x1, y1, objectness, *query scores] in pixels."""
        torch = self.torch
        with torch.no_grad():
            px = self.pixels(images_bgr)
            ids = self.text["input_ids"].repeat(len(images_bgr), 1)
            mask = self.text["attention_mask"].repeat(len(images_bgr), 1)
            out = self.model(input_ids=ids, attention_mask=mask, pixel_values=px)
            scores = torch.sigmoid(out.logits).cpu().numpy()          # B x P x Q
            obj = torch.sigmoid(out.objectness_logits).cpu().numpy()  # B x P
            boxes = out.pred_boxes.cpu().numpy()                       # B x P x 4, cxcywh in [0,1]
        res = []
        for b, im in enumerate(images_bgr):
            h, w = im.shape[:2]
            s = max(h, w)  # the processor pads to a square at the top-left
            cls_scores = scores[b][:, self.q_cls >= 0]
            sel = np.nonzero(cls_scores.max(1) >= keep)[0]
            cx, cy, bw, bh = (boxes[b][sel] * s).T
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).clip(0, s)
            res.append([[*map(lambda v: round(float(v), 2), xyxy[i]), round(float(obj[b][j]), 4),
                         *[round(float(v), 4) for v in scores[b][j]]] for i, j in enumerate(sel)])
        return res

    def detect(self, image_bgr, cam="a", cfg=None, keep=0.05):
        """One image -> decoded [cls, conf, x0, y0, x1, y1] detections."""
        return decode(self.raw([image_bgr], keep=keep)[0], self.q_cls, cam, cfg, image_bgr)


def nms(boxes, scores, thr):
    order = np.argsort(-np.asarray(scores))
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def decode(raw, q_cls, cam, cfg=None, image=None):
    """Raw OWL boxes -> [cls, conf, x0, y0, x1, y1]. No simulator state is used.

    Class = colour whose best prompt scores highest (or, with color="recolor", the
    box's dominant cube colour); dropped when the score is low, when a distractor
    scores higher (by ``margin``), when the box shape cannot be a cube in that camera,
    or when the colour check fails. Then class-agnostic NMS."""
    cfg = {**LABEL_CFG, **(cfg or {})}
    q_cls = np.asarray(q_cls)
    n_cls = int(q_cls.max()) + 1
    dets = []
    for r in raw:
        x0, y0, x1, y1 = r[:4]
        w, h = x1 - x0, y1 - y0
        if min(w, h) < cfg["min_side"] or max(w, h) / max(min(w, h), 1e-6) > cfg["max_aspect"]:
            continue
        if cam == "a" and max(w, h) > cfg["max_side_a"]:
            continue
        sc = np.asarray(r[5:])
        per = np.array([sc[q_cls == c].max() for c in range(n_cls)])
        dist = sc[q_cls < 0].max() if (q_cls < 0).any() else 0.0
        c = int(per.argmax())
        if cfg["color"] != "off":
            frac = color_fractions(image, r[:4])
            if cfg["color"] in ("recolor", "recolor_any"):
                c = int(frac.argmax())
            if frac[c] < cfg["min_frac"]:
                continue
        # recolor_any: OWL's best colour prompt acts as a class-agnostic cube score.
        conf = float(per.max() if cfg["color"] == "recolor_any" else per[c])
        if conf < cfg["score_thr"] or conf < dist + cfg["margin"]:
            continue
        dets.append([c, conf, x0, y0, x1, y1])
    if not dets:
        return []
    keep = nms([d[2:] for d in dets], [d[1] for d in dets], cfg["nms_iou"])
    return [dets[i] for i in keep]


def read_jsonl(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def load_frames(data):
    import glob
    frames = read_jsonl(sorted(glob.glob(os.path.join(data, "frames_*.jsonl"))))
    return {f["id"]: f for f in frames}


def load_gt(data):
    """Simulator GT. Evaluation only."""
    import glob
    return {g["id"]: g["cubes"] for g in read_jsonl(sorted(glob.glob(os.path.join(data, "gt_*.jsonl"))))}


def run_label(args):
    """Raw OWL boxes for every frame image; keeps all boxes with a class score >= 0.03,
    so decoding can be re-tuned without re-running the model."""
    import cv2
    import torch

    torch.set_num_threads(args.threads)
    frames = load_frames(args.data)
    path = os.path.join(args.data, "owl_raw.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path)}
    todo = [(fid, tag) for fid in sorted(frames) for tag, _ in CAMS if f"{fid}_{tag}" not in done]
    print(f"{len(todo)} images to label ({len(done)} done)", flush=True)
    teacher = OwlTeacher(args.device)
    meta_path = os.path.join(args.data, "owl_queries.json")
    json.dump(dict(model=OWL_MODEL, queries=teacher.queries, q_cls=teacher.q_cls.tolist()), open(meta_path, "w"), indent=1)
    t0 = time.time()
    with open(path, "a") as fh:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            ims = [cv2.imread(os.path.join(args.data, "frames", f"{fid}_{tag}.png")) for fid, tag in chunk]
            for (fid, tag), r in zip(chunk, teacher.raw(ims, keep=0.03)):
                fh.write(json.dumps(dict(key=f"{fid}_{tag}", raw=r)) + "\n")
            fh.flush()
            if args.device == "mps" and (i // args.batch) % 10 == 9:
                torch.mps.empty_cache()  # keep the unified-memory footprint small
            if (i // args.batch) % 20 == 0:
                rate = (i + len(chunk)) / (time.time() - t0)
                print(f"{i + len(chunk)}/{len(todo)} images, {rate:.2f}/s", flush=True)
    print(f"labelled in {time.time() - t0:.0f}s", flush=True)


def load_owl(data, cfg=None, keys=None):
    """{image key: decoded detections} using LABEL_CFG (or overrides)."""
    import cv2
    cfg = {**LABEL_CFG, **(cfg or {})}
    q = json.load(open(os.path.join(data, "owl_queries.json")))
    out = {}
    with open(os.path.join(data, "owl_raw.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if keys is not None and r["key"] not in keys:
                continue
            im = cv2.imread(os.path.join(data, "frames", r["key"] + ".png")) if cfg["color"] != "off" else None
            out[r["key"]] = decode(r["raw"], q["q_cls"], r["key"][-1], cfg, im)
    return out


def run_build(args):
    """YOLO dataset from OWL labels only. gt.jsonl is not read here."""
    frames = load_frames(args.data)
    cfg = json.loads(args.cfg) if args.cfg else None
    owl = load_owl(args.data, cfg)
    out = os.path.abspath(args.out)
    counts = {}
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    for fid, f in frames.items():
        for tag, _ in CAMS:
            key = f"{fid}_{tag}"
            if key not in owl:
                continue
            split = f["split"]
            src = os.path.abspath(os.path.join(args.data, "frames", f"{key}.png"))
            dst = os.path.join(out, "images", split, f"{key}.png")
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            with open(os.path.join(out, "labels", split, f"{key}.txt"), "w") as fh:
                for c, conf, x0, y0, x1, y1 in owl[key]:
                    x0, y0, x1, y1 = [min(max(v, 0.0), IMG) for v in (x0, y0, x1, y1)]
                    fh.write(f"{c} {(x0 + x1) / 2 / IMG:.6f} {(y0 + y1) / 2 / IMG:.6f} "
                             f"{(x1 - x0) / IMG:.6f} {(y1 - y0) / IMG:.6f}\n")
            k = (split, tag)
            counts[k] = counts.get(k, [0, 0])
            counts[k][0] += 1
            counts[k][1] += len(owl[key])
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for c, n in CLASS_NAMES.items():
            fh.write(f"  {c}: {n}\n")
    for (split, tag), (n, b) in sorted(counts.items()):
        print(f"{split:5s} cam {tag}: {n} images, {b} OWL boxes")
    json.dump(dict(label_cfg={**LABEL_CFG, **(cfg or {})}, counts={f"{s}/{t}": v for (s, t), v in counts.items()}),
              open(os.path.join(out, "build.json"), "w"), indent=1)


# ---------------------------------------------------------------------------
# Detection evaluation against simulator GT (shared with train_yolo_distill.py)
# ---------------------------------------------------------------------------

MIN_VIS = {"a": 30, "w": 60}  # px of visible cube; fewer is too occluded to be required


def match_frame(dets, cubes, tag, iou_thr=0.5):
    """Greedy class-aware matching of [cls, conf, x0, y0, x1, y1] to GT mask boxes.

    Returns (per-cube list of (cube, matched det or None, iou), classes of false positives).
    Detections matching an ignored (barely visible) cube are not false positives."""
    req = [c for c in cubes if c[f"box_{tag}"] is not None and c[f"vis_{tag}"] >= MIN_VIS[tag]]
    ign = [c for c in cubes if c[f"box_{tag}"] is not None and c[f"vis_{tag}"] < MIN_VIS[tag]]
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    used = set()
    result = {}
    fp = []
    for i in order:
        d = dets[i]
        best, bi = None, iou_thr
        for k, c in enumerate(req):
            if k in used or c["cls"] != d[0]:
                continue
            v = _iou(d[2:6], c[f"box_{tag}"])
            if v >= bi:
                best, bi = k, v
        if best is not None:
            used.add(best)
            result[best] = (d, bi)
        elif not any(c["cls"] == d[0] and _iou(d[2:6], c[f"box_{tag}"]) >= 0.3 for c in ign):
            fp.append(int(d[0]))
    return [(c, *result.get(k, (None, 0.0))) for k, c in enumerate(req)], fp


def det_metrics(frames, gt, dets_by_key, keys_filter=None, iou_thr=0.5):
    """Recall / precision / matched IoU per camera x class and camera x situation.

    dets_by_key: {f"{fid}_{tag}": [[cls, conf, x0, y0, x1, y1], ...]}"""
    acc = {}

    def add(k, tp=0, n=0, fp=0, iou=None):
        a = acc.setdefault(k, dict(tp=0, n=0, fp=0, iou=[]))
        a["tp"] += tp
        a["n"] += n
        a["fp"] += fp
        if iou is not None:
            a["iou"].append(iou)

    for fid, f in frames.items():
        if keys_filter and not keys_filter(f):
            continue
        for tag, _ in CAMS:
            key = f"{fid}_{tag}"
            if key not in dets_by_key:
                continue
            pairs, fp = match_frame(dets_by_key[key], gt[fid], tag, iou_thr)
            add((tag, "all", "all"), fp=len(fp))
            add((tag, "task", f["task"]), fp=len(fp))
            add((tag, "source", f["source"]), fp=len(fp))
            for c in fp:
                add((tag, "class", CLASS_NAMES[c].split()[0]), fp=1)
            for c, d, iou in pairs:
                hit = d is not None
                cname = CLASS_NAMES[c["cls"]].split()[0]
                ks = [(tag, "all", "all"), (tag, "class", cname), (tag, "sit", c["sit"]), (tag, "task", f["task"]),
                      (tag, "source", f["source"])]
                if c["covered"]:
                    ks.append((tag, "sit", "covered"))
                if c["arm_near"]:
                    ks.append((tag, "sit", "arm_near"))
                for k in ks:
                    add(k, tp=int(hit), n=1, iou=iou if hit else None)
    out = {}
    for (tag, kind, name), a in sorted(acc.items()):
        ndet = a["tp"] + a["fp"]
        out[f"{tag}/{kind}/{name}"] = dict(
            n=a["n"], recall=round(a["tp"] / a["n"], 4) if a["n"] else None,
            precision=round(a["tp"] / ndet, 4) if (kind != "sit" and ndet) else None,
            fp=a["fp"] if kind != "sit" else None,
            mean_iou=round(float(np.mean(a["iou"])), 4) if a["iou"] else None)
    return out


def print_metrics(m, title):
    print(f"--- {title}")
    print(f"{'key':34s} {'n':>6s} {'recall':>7s} {'prec':>6s} {'IoU':>6s}")
    for k, v in m.items():
        f = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"{k:34s} {v['n']:6d} {f(v['recall']):>7s} {f(v['precision']):>6s} {f(v['mean_iou']):>6s}")


def run_evaluate(args):
    frames = load_frames(args.data)
    gt = load_gt(args.data)
    cfg = json.loads(args.cfg) if args.cfg else None
    owl = load_owl(args.data, cfg)
    dets = {k: [[c, conf, x0, y0, x1, y1] for c, conf, x0, y0, x1, y1 in v] for k, v in owl.items()}
    m = det_metrics(frames, gt, dets, lambda f: args.split == "all" or f["split"] == args.split)
    print_metrics(m, f"OWLv2 labels vs GT, split={args.split}, cfg={ {**LABEL_CFG, **(cfg or {})} }")
    if args.out:
        json.dump(dict(split=args.split, label_cfg={**LABEL_CFG, **(cfg or {})}, metrics=m), open(args.out, "w"), indent=1)


def run_tune(args):
    """Grid-search the decode thresholds against GT on TRAINING frames (never the test
    split). Only these few scalars are chosen with GT; the labels themselves are OWL's."""
    import itertools

    frames = {k: f for k, f in load_frames(args.data).items() if f["split"] == "train"}
    gt = load_gt(args.data)
    q = json.load(open(os.path.join(args.data, "owl_queries.json")))
    raw = {}
    with open(os.path.join(args.data, "owl_raw.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            if r["key"].rsplit("_", 1)[0] in frames:
                raw[r["key"]] = r["raw"]
    rng = np.random.default_rng(0)
    ids = sorted(frames)
    ids = [ids[i] for i in rng.permutation(len(ids))[:args.max_frames]]
    frames = {k: frames[k] for k in ids}
    import cv2
    raw = {k: v for k, v in raw.items() if k.rsplit("_", 1)[0] in frames}
    ims = {k: cv2.imread(os.path.join(args.data, "frames", k + ".png")) for k in raw}
    grid = dict(score_thr=[0.05, 0.08, 0.1, 0.15, 0.2], color=["off", "verify", "recolor"], min_frac=[0.1, 0.2],
                nms_iou=[0.3, 0.5])
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid, vals))
        dets = {k: decode(v, q["q_cls"], k[-1], cfg, ims[k]) for k, v in raw.items()}
        m = det_metrics(frames, gt, dets)
        f1s = []
        for tag, _ in CAMS:
            a = m[f"{tag}/all/all"]
            r, pr = a["recall"] or 0, a["precision"] or 0
            f1s.append(2 * r * pr / (r + pr) if r + pr else 0)
        rows.append((float(np.mean(f1s)), cfg, [m[f"{t}/all/all"] for t, _ in CAMS]))
    rows.sort(key=lambda r: -r[0])
    for f1, cfg, m in rows[:10]:
        print(f"meanF1={f1:.3f} {cfg} " + " | ".join(f"R={x['recall']} P={x['precision']}" for x in m))
    if args.out:
        json.dump([dict(f1=f, cfg=c, metrics=m) for f, c, m in rows], open(args.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes")
    s.add_argument("--data", required=True)
    s.add_argument("--groups", required=True, help="seed range, e.g. 2100-2399; one env reset per group")
    s.add_argument("--per_group", type=int, default=5)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--worker", type=int, default=0)
    r = sub.add_parser("rollouts")
    r.add_argument("--data", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--episodes", type=int, default=4)
    r.add_argument("--stride", type=int, default=40, help="keep every stride-th env step")
    lab = sub.add_parser("label")
    lab.add_argument("--data", required=True)
    lab.add_argument("--device", default="cpu")
    lab.add_argument("--batch", type=int, default=2)
    lab.add_argument("--threads", type=int, default=2)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cfg", default=None, help="JSON overrides of LABEL_CFG")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    e.add_argument("--cfg", default=None)
    e.add_argument("--out", default=None)
    tu = sub.add_parser("tune")
    tu.add_argument("--data", required=True)
    tu.add_argument("--max_frames", type=int, default=800)
    tu.add_argument("--out", default=None)
    args = p.parse_args()
    {"tune": run_tune, "scenes": run_scenes, "rollouts": run_rollouts, "label": run_label, "build": run_build,
     "evaluate": run_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
