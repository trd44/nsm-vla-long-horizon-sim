"""Collect a pixel-to-world regression dataset for all paper task scene families.

Each scene: place cubes directly in the sim, teleport the arm by IK to a sampled
EEF pose, step briefly to settle, render both cameras at 256x256 and run YOLO. All
detections are stored raw, so box selection can be studied offline; ground truth
is read only for labels and for pairing boxes to cubes. Output is one JSONL per
worker, appended to and flushed periodically; re-running resumes.

    python perception/collect_data.py --out DIR --scenes 6000 --workers 3 --max_minutes 60
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings

# Weights set how many scenes each env config gets.
ENV_CONFIGS = {
    "Hanoi": ("Hanoi", {}, 0.20),
    "Hanoi4x3": ("Hanoi4x3", {}, 0.20),
    "HeightStacking3": ("HeightStacking", dict(min_cubes=3, max_cubes=3, cube_placement_noise=0.025, deterministic_reset=True), 0.15),
    "HeightStacking4": ("HeightStacking", dict(min_cubes=4, max_cubes=4, cube_placement_noise=0.025, deterministic_reset=True), 0.15),
    "CubeSorting": ("CubeSorting", {}, 0.15),
    "AssemblyLineSorting": ("AssemblyLineSorting", dict(num_bins=3, cube_placement_noise=0.025), 0.15),
}
PEGS = [[0.0, -0.18], [0.0, 0.02], [0.0, 0.22]]
PEG_NOISE = 0.03
ROW_NOISE = 0.03
SETTLE_STEPS = 10
SCENES_PER_RESET = 25
DROP = 0.0015


def proc(im, cv2):
    return cv2.cvtColor(cv2.flip(cv2.resize(im, (256, 256)), 0), cv2.COLOR_RGB2BGR)


class SceneEnv:
    """One robosuite env plus cube metadata and teleport helpers."""

    def __init__(self, key, suite, np, ctrl):
        self.key, self.np = key, np
        name, kw, _ = ENV_CONFIGS[key]
        self.env = suite.make(
            name, robots="Kinova3", controller_configs=ctrl, has_renderer=False,
            has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=False,
            camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256,
            camera_widths=256, control_freq=20, horizon=10 ** 7, ignore_done=True,
            hard_reset=True, **kw)
        self.reset()

    def reset(self):
        """Reset the env (fresh init noise, and fresh cube sizes in CubeSorting)."""
        np, env = self.np, self.env
        self.obs = env.reset()
        m = env.sim.model
        self.cubes = {}
        for j in m.joint_names:
            g = re.fullmatch(r"(cube\d+)_joint0", j)
            if not g:
                continue
            c = g.group(1)
            gid = m.geom_name2id(f"{c}_g0_vis")
            mat = m.mat_id2name(m.geom_matid[gid]) if hasattr(m, "mat_id2name") else None
            if mat is None:
                import mujoco
                mat = mujoco.mj_id2name(m._model, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[gid]))
            num = int(re.search(r"number(\d)", mat).group(1))
            self.cubes[c] = dict(joint=j, gid=gid, half=float(m.geom_size[gid][0]), texture=num,
                                 body=m.body_name2id(f"{c}_main"))
        self.table_z = float(env.table_offset[2])
        r = env.robots[0]
        self.site = r.eef_site_id if isinstance(r.eef_site_id, str) else m.site_id2name(r.eef_site_id)
        self.q_idx = list(r._ref_joint_pos_indexes)
        self.v_idx = list(r._ref_joint_vel_indexes)
        self.jnt_range = np.array([m.jnt_range[m.joint_name2id(n)] for n in r.robot_joints])
        self.home_q = env.sim.data.qpos[self.q_idx].copy()
        self.home_ee = np.array(env.sim.data.site_xpos[m.site_name2id(self.site)])
        self.ori = np.array(env.sim.data.site_xmat[m.site_name2id(self.site)]).reshape(3, 3)

    def ik(self, target, iters=60):
        """Damped least-squares IK to target position, holding the reset orientation."""
        np, sim = self.np, self.env.sim
        import robosuite.utils.transform_utils as T
        sid = sim.model.site_name2id(self.site)
        q = sim.data.qpos[self.q_idx].copy()
        for _ in range(iters):
            sim.data.qpos[self.q_idx] = q
            sim.forward()
            ep = target - sim.data.site_xpos[sid]
            R = sim.data.site_xmat[sid].reshape(3, 3)
            er = T.quat2axisangle(T.mat2quat(self.ori @ R.T))
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            J = np.vstack([sim.data.get_site_jacp(self.site).reshape(3, -1)[:, self.v_idx],
                           sim.data.get_site_jacr(self.site).reshape(3, -1)[:, self.v_idx]])
            e = np.r_[np.clip(ep, -0.05, 0.05), np.clip(er, -0.2, 0.2)]
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q = np.clip(q, self.jnt_range[:, 0], self.jnt_range[:, 1])
        sim.data.qpos[self.q_idx] = q
        sim.data.qvel[self.v_idx] = 0.0

    def place(self, cube, xyz, yaw=0.0):
        np, sim = self.np, self.env.sim
        sim.data.set_joint_qpos(self.cubes[cube]["joint"],
                                np.r_[xyz, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        sim.data.set_joint_qvel(self.cubes[cube]["joint"], np.zeros(6))

    # ---- arrangement samplers: each returns {cube: (xyz, kind, level)} ----
    def _stacks(self, names, rng):
        """Legal stacks (largest at the bottom) on three off-centre pegs."""
        np = self.np
        order = sorted(names, key=lambda c: -self.cubes[c]["half"])
        centers = [np.array(p) + rng.uniform(-PEG_NOISE, PEG_NOISE, 2) for p in PEGS]
        # Some scenes are one full tower, which the tasks visit most.
        if rng.random() < 0.15:
            k0 = int(rng.integers(3))
            pegs = {c: k0 for c in order}
        else:
            pegs = {c: int(rng.integers(3)) for c in order}
        top = {0: self.table_z, 1: self.table_z, 2: self.table_z}
        level = {0: 0, 1: 0, 2: 0}
        out = {}
        for c in order:
            k, h = pegs[c], self.cubes[c]["half"]
            xy = centers[k] + rng.uniform(-0.003, 0.003, 2)
            out[c] = (np.r_[xy, top[k] + h + DROP], "stack", level[k])
            top[k] += 2 * h + DROP
            level[k] += 1
        return out

    def _row(self, names, rng):
        np = self.np
        ys = {"HeightStacking3": (-0.24, 0.13), "HeightStacking4": (-0.24, 0.11)}.get(self.key, (-0.25, 0.25))
        pos = np.linspace(ys[0], ys[1], len(names))
        rng.shuffle(pos)
        out = {}
        for c, y in zip(names, pos):
            out[c] = (np.r_[rng.uniform(-ROW_NOISE, ROW_NOISE), y + rng.uniform(-ROW_NOISE, ROW_NOISE),
                            self.table_z + self.cubes[c]["half"] + DROP], "row", 0)
        return out

    def _tower(self, names, rng):
        """Partial tower on the HeightStacking platform; the rest stay in the row."""
        np, sim = self.np, self.env.sim
        out = self._row(names, rng)
        n = int(rng.integers(1, len(names) + 1))
        chosen = list(rng.permutation(names))[:n]
        pid = sim.model.body_name2id("platform_main")
        gid = sim.model.geom_name2id("platform_g0_vis")
        top = float(sim.data.body_xpos[pid][2] + sim.model.geom_size[gid][2])
        center = sim.data.body_xpos[pid][:2] + rng.uniform(-0.03, 0.03, 2)
        for lvl, c in enumerate(chosen):
            h = self.cubes[c]["half"]
            out[c] = (np.r_[center + rng.uniform(-0.004, 0.004, 2), top + h + DROP], "tower", lvl)
            top += 2 * h + DROP
        return out

    def _bins(self, names, rng):
        """Row cubes, with a random subset resting inside the sorting bins."""
        np, sim = self.np, self.env.sim
        out = self._row(names, rng)
        prefix = "platform" if self.key == "CubeSorting" else "bin"
        bodies = [b for b in sim.model.body_names if re.fullmatch(prefix + r"\d_main", b)]
        n = int(rng.integers(0, len(names) + 1))
        for c in list(rng.permutation(names))[:n]:
            b = bodies[int(rng.integers(len(bodies)))]
            pid = sim.model.body_name2id(b)
            gid = sim.model.geom_name2id(b.replace("_main", "_g0_vis"))
            half = self.cubes[c]["half"]
            slack = max(0.0, float(sim.model.geom_size[gid][0]) - 0.006 - half - 0.002)
            top = float(sim.data.body_xpos[pid][2] + sim.model.geom_size[gid][2])
            xy = sim.data.body_xpos[pid][:2] + rng.uniform(-slack, slack, 2)
            out[c] = (np.r_[xy, top + half + DROP], "bin", 0)
        return out

    def arrangement(self, rng):
        np = self.np
        names = sorted(self.cubes)
        if self.key == "Hanoi":
            names = ["cube1", "cube2", "cube3"]
        if self.key in ("Hanoi", "Hanoi4x3"):
            out = self._stacks(names, rng)
        elif self.key.startswith("HeightStacking"):
            out = self._tower(names, rng) if rng.random() < 0.7 else self._row(names, rng)
        else:
            out = self._bins(names, rng)
        if rng.random() < 0.10:
            # A cube off any structure, such as after a missed placement. Only a
            # cube with nothing on top moves, so no stack is left floating.
            tops = [c for c in names if not any(
                o != c and np.linalg.norm(out[o][0][:2] - out[c][0][:2]) < 0.02 and out[o][2] > out[c][2]
                for o in names)]
            c = tops[int(rng.integers(len(tops)))]
            out[c] = (np.r_[rng.uniform(-0.12, 0.12), rng.uniform(-0.3, 0.3),
                            self.table_z + self.cubes[c]["half"] + DROP], "free", 0)
        return out, names

    def sample_ee(self, placed, rng):
        np = self.np
        mode = rng.choice(["home", "hover", "grasp", "workspace"], p=[0.15, 0.35, 0.15, 0.35])
        tops = [c for c, v in placed.items() if not any(
            o != c and np.linalg.norm(placed[o][0][:2] - v[0][:2]) < 0.02 and placed[o][2] > v[2]
            for o in placed)]
        if mode == "home":
            t = self.home_ee + rng.normal(0, 0.01, 3)
        elif mode == "hover":
            c = rng.choice(sorted(placed))
            t = placed[c][0] + np.r_[rng.normal(0, 0.012, 2), rng.uniform(0.06, 0.22)]
        elif mode == "grasp":
            c = rng.choice(sorted(tops))
            t = placed[c][0] + np.r_[rng.normal(0, 0.006, 2), rng.uniform(0.0, 0.035)]
        else:
            t = np.r_[rng.uniform(-0.2, 0.15), rng.uniform(-0.32, 0.32), rng.uniform(0.86, 1.15)]
        return str(mode), t


def cube_box(sim, cube, cam_mat, np):
    """Project the cube's eight corners. Returns an xyxy box in upright image pixels."""
    pos = sim.data.body_xpos[cube["body"]]
    R = sim.data.body_xmat[cube["body"]].reshape(3, 3)
    h = cube["half"]
    corners = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
    pts = pos + corners @ R.T
    hom = np.c_[pts, np.ones(8)] @ cam_mat[:3].T
    if (hom[:, 2] <= 1e-4).any():
        return None
    # camera_utils' matrix maps world points to (col, row, depth) on the upright image.
    uv = hom[:, :2] / hom[:, 2:3]
    x0, x1 = uv[:, 0].min(), uv[:, 0].max()
    y0, y1 = uv[:, 1].min(), uv[:, 1].max()
    if x1 < 0 or y1 < 0 or x0 >= 256 or y0 >= 256:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def run_worker(args):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                      MUJOCO_GL=os.environ.get("MUJOCO_GL", "cgl"))
    warnings.filterwarnings("ignore")
    import cv2
    import numpy as np
    import robosuite as suite
    import torch
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    from ultralytics import YOLO

    from perception.features import TEXTURE_TO_CLS

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    yolo = YOLO(args.yolo)
    ctrl = suite.load_controller_config(default_controller="OSC_POSITION")
    path = os.path.join(args.out, f"scenes_w{args.worker}.jsonl")
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    seed = args.seed * 1000 + args.worker * 100 + done
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2 ** 31))
    envs, keys = {}, args.envs or list(ENV_CONFIGS)
    weights = np.array([ENV_CONFIGS[k][2] for k in keys])
    weights = weights / weights.sum()
    t0 = time.time()
    n = done
    fh = open(path, "a")
    target = args.scenes // args.workers + (1 if args.worker < args.scenes % args.workers else 0)
    while n < target and (time.time() - t0) / 60 < args.max_minutes:
        key = keys[rng.choice(len(keys), p=weights)]
        # Keep one env alive per process. With several, a hard reset in one rebuilds GL
        # textures the others still reference, and cubes then render in the wrong colours.
        if key in envs:
            envs[key].reset()
        else:
            for old in envs.values():
                old.env.close()
            envs = {key: SceneEnv(key, suite, np, ctrl)}
        se = envs[key]
        env, sim = se.env, se.env.sim
        for _ in range(SCENES_PER_RESET):
            if n >= target:
                break
            placed, names = se.arrangement(rng)
            # Hanoi has an inactive fourth cube: keep it out of view.
            for c in se.cubes:
                if c not in names:
                    se.place(c, np.array([3.0, 3.0, 0.1]))
            for c, (xyz, _, _) in placed.items():
                yaw = rng.uniform(-0.25, 0.25) if rng.random() < 0.3 else 0.0
                se.place(c, xyz, yaw)
            sim.forward()
            mode, target_ee = se.sample_ee(placed, rng)
            se.ik(np.clip(target_ee, [-0.3, -0.4, se.table_z + 0.015], [0.3, 0.4, 1.25]))
            sim.forward()
            for _s in range(SETTLE_STEPS):
                obs, *_ = env.step(np.array([0, 0, 0, -1.0]))
            speed = max(np.linalg.norm(sim.data.get_joint_qvel(se.cubes[c]["joint"])[:3]) for c in names)
            if speed > 0.02:
                for _s in range(5):
                    obs, *_ = env.step(np.array([0, 0, 0, -1.0]))
                speed = max(np.linalg.norm(sim.data.get_joint_qvel(se.cubes[c]["joint"])[:3]) for c in names)
            a, w = proc(obs["agentview_image"], cv2), proc(obs["robot0_eye_in_hand_image"], cv2)
            res = yolo.predict([a, w], verbose=False, device="cpu")
            torch.set_num_threads(1)  # ultralytics may raise this during predictor setup
            boxes = [[[int(b.cls), round(float(b.conf), 4), *[round(v, 5) for v in b.xywhn[0].tolist()]]
                      for b in r.boxes] for r in res]
            cams = {}
            for cam, tag in (("agentview", "a"), ("robot0_eye_in_hand", "w")):
                M = get_camera_transform_matrix(sim, cam, 256, 256)
                seg = sim.render(camera_name=cam, width=256, height=256, depth=False, segmentation=True)
                cams[tag] = (M, seg[..., 1], seg[..., 0])
            rows = []
            for c in names:
                info = se.cubes[c]
                rec = dict(name=c, cls=TEXTURE_TO_CLS[info["texture"]], half=info["half"],
                           gt=[round(float(v), 5) for v in sim.data.body_xpos[info["body"]]],
                           placed=[round(float(v), 5) for v in placed[c][0]],
                           kind=placed[c][1], level=placed[c][2])
                for tag, (M, sid, typ) in cams.items():
                    box = cube_box(sim, info, M, np)
                    rec[f"box_{tag}"] = box
                    rec[f"vis_{tag}"] = int(((typ == 5) & (sid == info["gid"])).sum())
                rows.append(rec)
            ee = [round(float(v), 5) for v in obs["robot0_eef_pos"]]
            scene = dict(id=f"{args.seed}-{args.worker}-{n}", env=key, ee_mode=mode, ee=ee,
                         max_speed=round(float(speed), 4), cubes=rows, boxes_a=boxes[0], boxes_w=boxes[1])
            fh.write(json.dumps(scene) + "\n")
            n += 1
            if n % 20 == 0:
                fh.flush()
                print(f"w{args.worker} n={n} {(n - done) / (time.time() - t0):.2f} scenes/s", flush=True)
    fh.close()
    print(f"w{args.worker} finished n={n}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=6000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max_minutes", type=float, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--yolo", default="models/yolo/hanoi_yolo.pt")
    p.add_argument("--envs", nargs="*", default=None, help="restrict to these ENV_CONFIGS keys")
    p.add_argument("--worker", type=int, default=-1)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.worker >= 0:
        return run_worker(args)
    base = [sys.executable, __file__, "--out", args.out, "--scenes", str(args.scenes), "--workers", str(args.workers),
            "--max_minutes", str(args.max_minutes), "--seed", str(args.seed), "--yolo", args.yolo]
    if args.envs:
        base += ["--envs", *args.envs]
    procs = [subprocess.Popen(base + ["--worker", str(i)]) for i in range(args.workers)]
    codes = [q.wait() for q in procs]
    print("worker exit codes", codes)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
