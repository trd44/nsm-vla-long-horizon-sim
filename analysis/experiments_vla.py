"""Evaluate the fine-tuned pi0 VLA under the same conditions as the NSM evaluation.

The policy runs in an openpi policy server (see scripts/vla_eval/). This script is the
client: it builds each paper task exactly as analysis/experiments_neurosymbolic.py does,
and scores it with the same functions, so the two models' numbers are directly comparable:

- Environments: planning.task_configs.make_env with the task's paper kwargs (Kinova3,
  20 Hz, uniform +/-2.5 cm spawn noise, swapped bins), so the same seed gives the same
  cube layout for both models. Only the controller differs: the VLA was trained on
  OSC_POSE (rotation zeroed), the NSM's diffusion skills on OSC_POSITION.
- Seeding: episode i of a run uses seed + i, set before the reset, as in the NSM eval.
- Success: planning.task_configs.failed_goals after the same settle, i.e. every goal
  predicate (including the inside-the-bin-walls check for sorting) holds at the end.
- Timeout: 600 env steps without sub-task progress (paper section 4.4.1). Progress is
  a cube being grasped, or the moves left dropping to a new low. Moves left are the
  goals not yet holding for sorting and towers (each move completes one goal), and the
  length of a fresh Metric-FF plan from the current state for Hanoi. Both count
  progress in any order, so a VLA that sorts cubes in a different order than the
  planner is not cut off.
- Advancement: the largest fraction of the initial plan's moves completed, measured as
  (initial moves - moves left) / initial moves.

VLA protocol kept from openpi/examples/robosuite: 50 settle steps, images flipped and
resized to 224 (openpi_client.image_tools), state = 7 joint angles + gripper width,
one constant command per episode, 50-step action chunks with rotation zeroed.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "openpi", "packages", "openpi-client", "src"))

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from planning.task_configs import (  # noqa: E402
    NO_PROGRESS_LIMIT,
    PAPER_TASKS,
    failed_goals,
    make_detector,
    make_env,
    plan_for_state,
    summarize,
)

# Fine-tuned checkpoints (Hugging Face) with the openpi train config that serves them and
# the single command each was fine-tuned with (instruction_override in that config).
CHECKPOINTS = {
    "sort_2color": ("tduggan93/pi0-2-color-block-sorting", "pi0_cube_sorting_lora",
                    "Sort the cubes by size and color."),
    "tower_3_set": ("tduggan93/pi0-3-block-tower-stacking-set-order", "pi0_3_block_tower_stacking_det_lora",
                    "Stack the 3 blocks from largest to smallest on the gray area."),
    # The config's command is a sorting sentence although this is the Hanoi checkpoint;
    # it is what the model was fine-tuned with, so it is what the model is given.
    "hanoi_3": ("tduggan93/pi0-3-block-hanoi", "pi0_hanoi_50_lora", "Sort the cubes by color."),
    "sort_3color": ("tduggan93/pi0-3-color-block-sorting", "pi0_assembly_line_sorting_lora",
                    "Sort the cubes on the assembly line by color."),
    "tower_3_random": ("tduggan93/pi0-3-block-tower-stacking-random-order", "pi0_height_stacking_lora",
                       "Assemble the block tower."),
    "tower_4_set": ("tduggan93/pi0-4-block-tower-stacking", "pi0_4_block_tower_stacking_lora",
                    "Stack the 4 blocks from largest to smallest on the gray area."),
    "hanoi_4": ("tduggan93/pi0-4-block-hanoi", "pi0_hanoi_4x3_50_lora", "Play 4 block Towers of Hanoi."),
}

# The paper's 12 VLA conditions: (task, checkpoint key, condition).
CONDITIONS = [
    ("sort_2color", "sort_2color", "finetuned"),
    ("tower_3_set", "tower_3_set", "finetuned"),
    ("hanoi_3", "hanoi_3", "finetuned"),
    ("sort_3color", "sort_3color", "finetuned"),
    ("tower_3_random", "tower_3_random", "finetuned"),
    ("tower_4_set", "tower_4_set", "finetuned"),
    ("hanoi_4", "hanoi_4", "finetuned"),
    ("sort_2color_swapped", "sort_2color", "zeroshot"),
    ("sort_3color", "sort_2color", "zeroshot"),
    ("tower_3_random", "tower_3_set", "zeroshot"),
    ("tower_4_set", "tower_3_set", "zeroshot"),
    ("hanoi_4", "hanoi_3", "zeroshot"),
]

SETTLE_VLA = 50          # zero-action steps before the policy starts (VLA runner protocol)
SETTLE_END = 10          # open-gripper steps before the final goal check (NSM SETTLE_STEPS)
REPLAN_STEPS = 50        # actions executed from each predicted chunk
RESIZE = 224
CAMERA = 256
MAX_UNSOLVABLE_RESETS = 50


def get_state(detector):
    return detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)


def preprocess(env, obs):
    """Same as openpi/examples/robosuite/obs_processor.py."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8)
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    joints = np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"]).astype(np.float32)
    left = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_left_inner_finger")]
    right = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_right_inner_finger")]
    width = np.array([np.linalg.norm(left - right)], dtype=np.float32)
    return img, wrist, np.concatenate([joints, width]).astype(np.float32)


class Progress:
    """600-step no-progress timeout over grasps and remaining-plan reductions."""

    CHECK_EVERY = 10

    def __init__(self, detector, task, goals, initial_moves):
        self.detector, self.task, self.goals = detector, task, goals
        self.initial_moves = initial_moves
        self.best_remaining = initial_moves
        self.grasp_counted = False
        self.was_grasped = set()
        self.last_progress = 0
        self.steps = 0
        self.unsolvable = False

    def remaining_moves(self):
        # Sorting and towers: every move completes exactly one goal, so the moves left are
        # the goals not yet holding (place_effect_holds, as in the NSM eval). Re-planning is
        # avoided there because the tower platform has no groundings, which makes plans from
        # mid-episode states wrong. Hanoi goals only hold at the end, so there the moves left
        # are the length of a fresh plan from the current state.
        if len(self.goals) == self.initial_moves:
            return len(failed_goals(self.detector, self.goals))
        plan, _, _, _ = plan_for_state(get_state(self.detector), self.detector, self.task, goals=self.goals)
        if plan is False:
            return None
        return len(plan) // 2

    def step(self):
        """Call once per env step. Returns True when the goals hold."""
        self.steps += 1
        grasped = {c for c in self.detector.objects if self.detector.grasped(c)}
        if grasped - self.was_grasped and not self.grasp_counted:
            self.grasp_counted = True
            self.last_progress = self.steps
        released = self.was_grasped - grasped
        self.was_grasped = grasped
        done = False
        # Re-plan on a release and then every CHECK_EVERY steps while nothing is held: a
        # cube is often still settling when the fingers open, so its effect shows up later.
        if released or (not grasped and self.steps % self.CHECK_EVERY == 0):
            remaining = self.remaining_moves()
            self.unsolvable = remaining is None
            if remaining is not None and remaining < self.best_remaining:
                self.best_remaining = remaining
                self.grasp_counted = False
                self.last_progress = self.steps
            done = remaining == 0
        return done

    @property
    def advancement(self):
        return (self.initial_moves - self.best_remaining) / self.initial_moves

    def timed_out(self):
        return self.steps - self.last_progress > NO_PROGRESS_LIMIT


def run_episode(env, detector, task, client, prompt, episode, args, video_dir=None):
    seed = args.seed + episode
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    unsolvable_resets = 0
    zero = np.zeros(env.action_dim)
    while True:
        obs = env.reset()
        obs, _, _, _ = env.step(zero)
        plan, goals, _, _ = plan_for_state(get_state(detector), detector, task)
        if plan:
            break
        unsolvable_resets += 1
        if unsolvable_resets >= MAX_UNSOLVABLE_RESETS:
            raise RuntimeError(f"{unsolvable_resets} resets without a plan on {task.name}")
    initial_cubes = {c: np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(f"{c}_main")]).round(4).tolist()
                     for c in detector.objects if f"{c}_main" in env.sim.model.body_names}
    print(f"Plan ({len(plan)} ops): {plan}\nGoals: {goals}", flush=True)
    progress = Progress(detector, task, goals, len(plan) // 2)

    settle = zero.copy()
    settle[-1] = -1.0
    for _ in range(SETTLE_VLA):
        obs, _, _, _ = env.step(settle)
        progress.steps += 1
    progress.last_progress = progress.steps

    frames = []
    chunk = []
    inference_s = []
    goals_held = False
    terminal = None
    while progress.steps < args.max_steps:
        img, wrist, state = preprocess(env, obs)
        if video_dir is not None:
            frames.append(img)
        if not chunk:
            t0 = time.time()
            result = client.infer({"observation/image": img, "observation/wrist_image": wrist,
                                   "observation/state": state, "prompt": prompt})
            inference_s.append(time.time() - t0)
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != env.action_dim:
                raise RuntimeError(f"Policy returned actions of shape {actions.shape}; env needs (N, {env.action_dim})")
            chunk = list(actions[:REPLAN_STEPS])
        action = np.array(chunk.pop(0), dtype=float)
        action[3:6] = 0.0
        obs, _, _, _ = env.step(action)
        if progress.step():
            goals_held = True
            terminal = "goals_reached"
            break
        if progress.timed_out():
            terminal = f"timeout_{NO_PROGRESS_LIMIT}"
            break
    else:
        terminal = f"max_steps_{args.max_steps}"

    open_action = zero.copy()
    open_action[-1] = -1.0
    if terminal == "goals_reached":
        for _ in range(SETTLE_END):
            env.step(open_action)
            progress.steps += 1
    unmet = failed_goals(detector, goals)
    success = not unmet
    if video_dir is not None and frames:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        imageio.mimwrite(os.path.join(video_dir, f"{task.name}_seed{seed}.mp4"), frames, fps=20)
    record = {
        "episode": episode,
        "seed": seed,
        "base_seed": args.seed,
        "task": task.name,
        "perception": "vla",
        "success": success,
        "advancement": 1.0 if success else progress.advancement,
        "moves_ok": progress.initial_moves - progress.best_remaining,
        "num_moves": progress.initial_moves,
        "plan_len": len(plan),
        "env_steps": progress.steps,
        "wall_s": time.time() - start,
        "timed_out": terminal.startswith("timeout"),
        "failure_reason": None if success else ("goal_not_held_at_end" if goals_held else terminal),
        "unmet_goals": unmet,
        "unsolvable_state": progress.unsolvable,
        "plan": plan,
        "goals": goals,
        "unsolvable_resets": unsolvable_resets,
        "initial_cubes": initial_cubes,
        "inference_calls": len(inference_s),
        "mean_inference_s": float(np.mean(inference_s)) if inference_s else None,
    }
    print(("Episode SUCCEEDED" if success else f"Episode FAILED ({record['failure_reason']}); unmet goals: {unmet}")
          + f" steps={progress.steps} advancement={record['advancement']:.3f}", flush=True)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(PAPER_TASKS))
    parser.add_argument("--checkpoint", required=True, choices=sorted(CHECKPOINTS),
                        help="Which fine-tuned checkpoint the server is running (key in CHECKPOINTS).")
    parser.add_argument("--prompt", default=None, help="Override the checkpoint's training command.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Hard cap on env steps per episode (the 600-step no-progress timeout ends almost all).")
    parser.add_argument("--results_dir", default="results/vla")
    parser.add_argument("--video", action="store_true", help="Save the policy's agentview frames per episode.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task = PAPER_TASKS[args.task]
    repo, config, trained_prompt = CHECKPOINTS[args.checkpoint]
    prompt = args.prompt or trained_prompt
    condition = "finetuned" if args.checkpoint == args.task else "zeroshot"
    label = f"{task.name}_vla_{condition}_{args.checkpoint}"
    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.join(args.results_dir, f"{label}_seed{args.seed}")
    if os.path.exists(f"{stem}.jsonl") and not args.overwrite:
        raise SystemExit(f"{stem}.jsonl already exists; pass --overwrite or another --results_dir")
    print(f"# task={task.name} condition={condition} checkpoint={repo} config={config}\n"
          f"# prompt={prompt!r} episodes={args.episodes} seeds={args.seed}..{args.seed + args.episodes - 1}", flush=True)

    import robosuite as suite
    env = make_env(
        task,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=CAMERA,
        camera_widths=CAMERA,
    )
    detector = make_detector(task, env)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"# server metadata: {client.get_server_metadata()}", flush=True)

    records = []
    video_dir = os.path.join(args.results_dir, "videos", label) if args.video else None
    with open(f"{stem}.jsonl", "w") as out:
        for episode in range(args.episodes):
            print(f"\n[{label}] Episode {episode + 1}/{args.episodes} (seed {args.seed + episode})", flush=True)
            record = run_episode(env, detector, task, client, prompt, episode, args, video_dir)
            record.update(condition=condition, checkpoint=repo, server_config=config, prompt=prompt)
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            summary = summarize(records, task)
            summary.update(condition=condition, checkpoint=repo, server_config=config, prompt=prompt, seed=args.seed,
                           env_kwargs=task.env_kwargs, controller="OSC_POSE")
            with open(f"{stem}.summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Running: success {summary['successes']}/{summary['episodes']}, "
                  f"advancement {summary['advancement_pct']:.1f}%", flush=True)
    client._ws.close()
    env.close()


if __name__ == "__main__":
    main()
