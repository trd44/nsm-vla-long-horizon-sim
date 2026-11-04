"""Log pixel-to-world rows from real task execution (DAgger-style perception data).

Wraps analysis/experiments_neurosymbolic.py without editing it, and takes the same
arguments plus --rollout_out. Every cube detection the executor would see is logged
with the exact 14-feature row of perception/features.py. Ground truth is read from the
simulator for LABELS, FILTERS and DIAGNOSTICS only: nothing logged here feeds back into
the executor, and in vision mode the logging hook returns the regressor's prediction
unchanged.

With --perception gt the policy is driven by simulator poses and the executor never
runs YOLO, so this wrapper runs YOLO itself on the frames the policy sees. Those
detections are logged and discarded. That reaches states a failing vision run does not.

Use training seeds only (2000 and up). Development used 0-9; 1000 and up are held out.

    python perception/collect_rollouts.py --task hanoi_4 --perception vision --seed 2000 \
        --episodes 8 --results_dir OUT/res --rollout_out OUT/data
"""
import argparse
import json
import os
import sys

import numpy as np

MIN_SEED = 2000  # below this are the development and held-out evaluation seeds
SPEED_MAX = 0.02  # m/s, as in collect_data.py's moving-scene filter
TILT_COS = float(np.cos(np.deg2rad(15.0)))  # cube's z axis against world z
REST_TOL = 0.006  # height away from any resting surface that counts as lifted
# A row is a near-duplicate of the cube's last kept row unless one of these changed.
MIN_AGENT_PX = 2
MIN_WRIST_PX = 6
MIN_EE_M = 0.012


class RolloutLogger:
    def __init__(self, path, task, perception):
        self.fh = open(path, "a")
        self.task, self.perception = task, perception
        self.seed = None
        self.frame = 0
        self.pending = []
        self.last = {}
        self.stats = {}
        self.geom = {}
        self.agent_M = None

    def count(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    def start_episode(self, seed):
        self.seed, self.frame, self.last = int(seed), 0, {}

    def begin_frame(self):
        self.frame += 1
        self.pending = []

    def add(self, executor, cls_id, cls_name, row):
        self.pending.append((executor, cls_id, cls_name, row))

    # ---- geometry read from the simulator, for labels and filters only ----
    def _half(self, sim, cube):
        if cube not in self.geom:
            gid = sim.model.geom_name2id(f"{cube}_g0_vis")
            self.geom[cube] = (gid, float(sim.model.geom_size[gid][0]), sim.model.body_name2id(f"{cube}_main"))
        return self.geom[cube]

    def _support(self, executor, cube, pos, positions):
        """Returns (cubes below, lifted). Compares against table, platform and lower cubes."""
        env = executor.detector.env
        sim = env.sim
        _, half, _ = self._half(sim, cube)
        below, rests = 0, [float(env.table_offset[2]) + half]
        if "platform_main" in sim.model.body_names:
            body = sim.data.body_xpos[sim.model.body_name2id("platform_main")]
            size = sim.model.geom_size[sim.model.geom_name2id("platform_g0_vis")]
            if np.abs(pos[:2] - body[:2]).max() < size[0] + half:
                rests.append(float(body[2] + size[2]) + half)
        for other, p in positions.items():
            if other == cube or not str(other).startswith("cube"):
                continue
            try:
                _, h2, _ = self._half(sim, other)
            except Exception:  # an inactive cube has no such geom
                continue
            if np.linalg.norm(p[:2] - pos[:2]) < half + h2 and p[2] < pos[2] - 0.5 * half:
                below += 1
                rests.append(float(p[2]) + h2 + half)
        return below, min(abs(float(pos[2]) - r) for r in rests) > REST_TOL

    def _is_new(self, cube, row):
        last = self.last.get(cube)
        if last is None:
            return True
        ch = np.abs(np.array(row[1:5]) - np.array(last[1:5])).max() >= MIN_AGENT_PX
        had, has = last[8] > 0, row[8] > 0
        cw = had != has or (has and np.abs(np.array(row[6:10]) - np.array(last[6:10])).max() >= MIN_WRIST_PX)
        ce = np.linalg.norm(np.array(row[11:14]) - np.array(last[11:14])) >= MIN_EE_M
        return bool(ch or cw or ce)

    def end_frame(self):
        """Filter the frame's detections and write the survivors. Reads GT for labels."""
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        from perception.collect_data import cube_box
        from perception.features import iou_xyxy

        if not self.pending:
            return
        executor = self.pending[0][0]
        sim = executor.detector.env.sim
        positions = {k: np.array(v, copy=True) for k, v in executor.detector.get_all_objects_pos().items()}
        grasped = executor.get_grasped_objects()
        best = {}
        for _, cls_id, _, row in self.pending:
            if cls_id not in best or row[5] > best[cls_id]:
                best[cls_id] = row[5]
        n_of_class = {}
        for _, cls_id, _, _ in self.pending:
            n_of_class[cls_id] = n_of_class.get(cls_id, 0) + 1
        seg = {}
        for _, cls_id, cls_name, row in self.pending:
            self.count("detections")
            cube = executor.color_to_pddl.get(cls_name)
            if cube is None or cube not in positions:
                self.count("skip_unmapped_class")
                continue
            if cube in grasped:
                self.count("skip_grasped")
                continue
            pos = positions[cube]
            gid, half, body = self._half(sim, cube)
            if float(sim.data.body_xmat[body].reshape(3, 3)[2, 2]) < TILT_COS:
                self.count("skip_tumbled")
                continue
            if np.linalg.norm(sim.data.get_joint_qvel(f"{cube}_joint0")[:3]) > SPEED_MAX:
                self.count("skip_moving")
                continue
            level, lifted = self._support(executor, cube, pos, positions)
            if lifted:
                self.count("skip_lifted")
                continue
            # The executor keeps the best box per class; others are false positives.
            if row[5] < best[cls_id]:
                self.count("non_top_box")
                continue
            if not self._is_new(cube, row):
                self.count("skip_duplicate")
                continue
            self.last[cube] = row
            if self.agent_M is None:
                self.agent_M = get_camera_transform_matrix(sim, "agentview", 256, 256)
            info = dict(name=cube, half=half, body=body)
            box_a = cube_box(sim, info, self.agent_M, np)
            box_w = cube_box(sim, info, get_camera_transform_matrix(sim, "robot0_eye_in_hand", 256, 256), np)
            if not seg:
                for tag, cam in (("a", "agentview"), ("w", "robot0_eye_in_hand")):
                    s = sim.render(camera_name=cam, width=256, height=256, depth=False, segmentation=True)
                    seg[tag] = (s[..., 0], s[..., 1])
            vis = {t: int(((typ == 5) & (sid == gid)).sum()) for t, (typ, sid) in seg.items()}

            def corners(cx, cy, w, h):  # integer xywh as the executor keeps them
                return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

            iou_a = iou_xyxy(corners(*row[1:5]), box_a) if box_a else 0.0
            iou_w = iou_xyxy(corners(*row[6:10]), box_w) if box_w and row[8] > 0 else 0.0
            self.fh.write(json.dumps(dict(
                env=self.task, task=self.task, perception=self.perception, seed=self.seed, group=f"{self.task}-{self.seed}",
                scene=f"{self.task}-{self.seed}", frame=self.frame, skill=executor.id, name=cube, cls=int(cls_id),
                kind="rollout", level=int(level), ee_mode=executor.id, x=[round(float(v), 5) for v in row],
                gt=[round(float(v), 5) for v in pos], vis_a=vis["a"], vis_w=vis["w"], iou_a=round(iou_a, 3),
                iou_w=round(iou_w, 3), n_boxes_class=n_of_class[cls_id], box_a=box_a, assoc="deploy")) + "\n")
            self.count("rows")
        self.pending = []
        self.fh.flush()

    def write_stats(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=1)


def run_gt_detections(executor, observation, logger):
    """Same detection loop as Executor_Diffusion.yolo_estimate step 1, results discarded."""
    import cv2

    logger.begin_frame()
    image1 = cv2.cvtColor(cv2.flip(cv2.resize(np.array(observation["agentview_image"], dtype=np.uint8), (256, 256)), 0), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(cv2.flip(cv2.resize(np.array(observation["robot0_eye_in_hand_image"], dtype=np.uint8), (256, 256)), 0), cv2.COLOR_RGB2BGR)
    model = executor.yolo_model
    p1 = model.predict(image1, verbose=False, device=executor.device)[0]
    p2 = model.predict(image2, verbose=False, device=executor.device)[0]
    ee = observation["robot0_eef_pos"]
    for pred in p1.boxes:
        cls_id = int(pred.cls)
        x, y, w, h = [int(v * 256) for v in pred.xywhn.tolist()[0]]
        if max(w, h) > executor.max_agentview_box_px:
            continue
        wrist, conf2 = [0, 0, 0, 0], 0.0
        for pred2 in p2.boxes:
            if int(pred2.cls) == cls_id:
                wrist, conf2 = [int(v * 256) for v in pred2.xywhn.tolist()[0]], float(pred2.conf)
                break
        logger.add(executor, cls_id, model.names[cls_id],
                   [float(cls_id), x, y, w, h, float(pred.conf), *wrist, conf2, float(ee[0]), float(ee[1]), float(ee[2])])
    logger.end_frame()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout_out", required=True)
    own, rest = parser.parse_known_args()
    os.makedirs(own.rollout_out, exist_ok=True)
    sys.argv = ["analysis/experiments_neurosymbolic.py"] + rest

    import analysis.experiments_neurosymbolic as driver
    from planning.executor import Executor_Diffusion as Ex

    args = driver.parse_args()
    if args.seed < MIN_SEED:
        raise SystemExit(f"--seed {args.seed}: rollout data must come from training seeds, {MIN_SEED} and up")
    tag = f"{args.task}_{args.perception}_seed{args.seed}"
    logger = RolloutLogger(os.path.join(own.rollout_out, f"{tag}.jsonl"), args.task, args.perception)

    run_episode = driver.run_episode

    def logged_episode(env, detector, monitor, task, skills, episode, a):
        logger.start_episode(a.seed + episode)
        try:
            return run_episode(env, detector, monitor, task, skills, episode, a)
        finally:
            logger.write_stats(os.path.join(own.rollout_out, f"{tag}.stats.json"))

    driver.run_episode = logged_episode

    if args.perception == "vision":
        pixel_to_world = Ex.pixel_to_world_dual
        yolo_estimate = Ex.yolo_estimate

        def logged_pixel_to_world(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z,
                                  sim=None, image_h=256, image_w=256, cls_name=None):
            name = cls_name or self.yolo_model.names[int(cls_id)]
            logger.add(self, int(cls_id), name, [float(cls_id), px1, py1, w1, h1, float(conf1), px2, py2, w2, h2, float(conf2),
                                                 float(ee_x), float(ee_y), float(ee_z)])
            return pixel_to_world(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z,
                                  sim=sim, image_h=image_h, image_w=image_w, cls_name=cls_name)

        def logged_yolo_estimate(self, *a, **k):
            logger.begin_frame()
            out = yolo_estimate(self, *a, **k)
            logger.end_frame()
            return out

        Ex.pixel_to_world_dual = logged_pixel_to_world
        Ex.yolo_estimate = logged_yolo_estimate
    else:
        from ultralytics import YOLO

        model = YOLO(args.yolo)
        get_object_obs = Ex.get_object_obs
        calls = {"n": 0}

        def logged_object_obs(self, env, objects_pos, predicted_pos, obj_to_pick, place_to_drop, relative_obs=False, first_obs=True):
            if first_obs:
                calls["n"] += 1
                if calls["n"] % 2 == 0:  # the executor's yolo_frequency
                    self.yolo_model = model
                    run_gt_detections(self, env._get_observations(), logger)
            return get_object_obs(self, env, objects_pos, predicted_pos, obj_to_pick, place_to_drop,
                                  relative_obs=relative_obs, first_obs=first_obs)

        Ex.get_object_obs = logged_object_obs

    driver.main()


if __name__ == "__main__":
    main()
