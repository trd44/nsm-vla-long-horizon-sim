"""Evaluate the neuro-symbolic model on the paper's tasks (Tables 3-4).

Example:
    python analysis/experiments_neurosymbolic.py --task hanoi_3 --perception vision --seed 0

Metrics follow the paper's protocol (sections 3.2, 4.4):
  * An episode ends once no plan operator has advanced within 600 env steps.
  * Success: every goal predicate holds at the end of the episode according to the
    simulator detector, after the arm has retreated and the scene has settled. Neither
    "all skills returned ok" nor a goal seen earlier counts if a cube later fell.
  * Advancement: the fraction of plan moves (PICK + PLACE) whose effect the detector
    verified after the arm retreated. It never decreases within an episode. The paper
    does not define this metric formally, so this definition is ours.

A failed move is not retried. By default the next move runs only if its preconditions
hold in the detector state, otherwise the episode ends. --replan instead replans from
the detector state; this is not described in the paper, so it is off by default.
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import random
import time

import gym
import joblib
import numpy as np
import robosuite as suite
import torch
from robosuite.wrappers import GymWrapper
from ultralytics import YOLO

from planning.executor import Executor_Diffusion
from planning.task_configs import (
    LEGACY_ENVS,
    NO_PROGRESS_LIMIT,
    TALL_PICK_HOME_Z,
    TALL_PICK_LEVEL,
    cubes_below,
    PAPER_TASKS,
    REPO_ROOT,
    EpisodeTimeout,
    ProgressMonitor,
    failed_goals,
    legacy_task,
    make_detector,
    make_env,
    missing_preconditions,
    parse_moves,
    place_effect_holds,
    plan_for_state,
    summarize,
)

POLICY_SETS = {
    "original": dict(directory="policies/hanoi_original", gripper_close_sign=-1),
    "retrained": dict(directory="policies/hanoi", gripper_close_sign=1),
}
# Home pose between moves. Measured on all five environments: reachable within 1 cm in
# 58-60 steps after the 50-step lift, and within about 5 cm of the pose after env.reset().
RESET_GRIPPER_POS = np.array([-0.080193391, -0.03391656, 0.95828137])
MAX_UNSOLVABLE_RESETS = 50
MAX_REPLANS = 5
SETTLE_STEPS = 10
GT_LABEL = "GROUND-TRUTH PERCEPTION (simulator poses, NOT vision)"
VISION_LABEL = "VISION PERCEPTION (YOLO + regressor)"


def termination_indicator(operator):
    if operator == 'pick':
        def Beta(state, symgoal):
            return state[f"grasped({symgoal[0]})"]
    elif operator == 'drop':
        def Beta(state, symgoal):
            key = f"in({symgoal[0]},{symgoal[1]})"
            if key not in state:
                # Hanoi-style detectors only report on(x,y), not in(x,y)
                key = f"on({symgoal[0]},{symgoal[1]})"
            return state[key] and not state[f"grasped({symgoal[0]})"]
    elif operator == 'reach_pick':
        def Beta(state, symgoal):
            return state[f"over(gripper,{symgoal[0]})"]
    elif operator == 'reach_drop':
        def Beta(state, symgoal):
            return state[f"over(gripper,{symgoal[1]})"]
    else:
        raise ValueError(f"Unknown operator: {operator}")
    return Beta


class DictObs(gym.Env):
    """Gym adapter returning dict observations; every env step goes through the monitor."""

    def __init__(self, env, monitor):
        super().__init__()
        self.env = env
        self.monitor = monitor
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        self.env.reset()
        return self.env._get_observations()

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.monitor.on_step()
        return self.env._get_observations(), reward, terminated or truncated, info

    def render(self, mode='human', *args, **kwargs):
        self.env.render()

    def _get_observations(self):
        return self.env._get_observations()

    def __getattr__(self, name):
        return getattr(self.env, name)


def get_state(detector):
    return detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)


def collect_observations(env, detector, count, action):
    observations = []
    for _ in range(count):
        env.step(action)
        obs = env._get_observations()
        obs['objects_pos'] = detector.get_all_objects_pos()
        observations.append(obs)
    return observations


def reset_gripper(env, detector, render=False, max_open_steps=80, max_home_steps=200):
    """Open the gripper, lift, then return to the home pose."""
    for _ in range(max_open_steps):
        if get_state(detector)["open_gripper(gripper)"]:
            break
        env.step(np.array([0, 0, 0, -1]))
        if render:
            env.render()
    else:
        print(f"[reset_gripper] open timeout after {max_open_steps} steps; continuing")
    for _ in range(50):
        env.step(np.array([0, 0, 0.5, 0]))
        if render:
            env.render()
    delta = RESET_GRIPPER_POS - env._get_observations()["robot0_eef_pos"]
    for _ in range(max_home_steps):
        if np.linalg.norm(delta) <= 0.01:
            break
        env.step(4.5 * np.array([delta[0], delta[1], delta[2], 0]))
        if render:
            env.render()
        delta = RESET_GRIPPER_POS - env._get_observations()["robot0_eef_pos"]
    else:
        print(f"[reset_gripper] home timeout after {max_home_steps} steps "
              f"(dist={np.linalg.norm(delta):.4f}); continuing")


def lift_to_home_z(env, z, render=False, max_steps=80):
    """Raise the arm straight up to height z."""
    for _ in range(max_steps):
        error = z - env._get_observations()["robot0_eef_pos"][2]
        if abs(error) <= 0.01:
            return
        env.step(np.array([0, 0, 4.5 * error, 0]))
        if render:
            env.render()


def reset_until_planned(env, detector, task, n_act, render=False):
    """Reset until the planner finds a plan; unsolvable resets are logged and counted."""
    unsolvable = 0
    while True:
        env.reset()
        if render:
            env.render()
        observations = collect_observations(env, detector, n_act, np.zeros(env.action_space.shape))
        plan, goals, _, _ = plan_for_state(get_state(detector), detector, task)
        if plan:
            return plan, goals, observations, unsolvable
        unsolvable += 1
        print(f"[UNSOLVABLE RESET] #{unsolvable} on {task.name}: plan={plan}; resetting again", flush=True)
        if unsolvable >= MAX_UNSOLVABLE_RESETS:
            raise RuntimeError(f"{unsolvable} consecutive resets gave no plan on {task.name}; "
                               "the detector state and spawn noise disagree")


def execute_move(env, detector, skills, move, goals, observations, tracking_data, args):
    """Run the move's skills. Returns (observations, tracking data, first failed skill)."""
    cube, _, destination = move
    failed_skill = None
    for skill in skills:
        env.monitor.skill = skill.id
        skill.model.reset()
        if tracking_data:
            skill.set_tracking_data(tracking_data)
        observations, success, _ = skill.execute(
            env, observations, args.n_act, (cube, destination), list(goals), args.render)
        tracking_data = skill.get_tracking_data()
        if args.debug:
            print(f"\t  {skill.id}: {'ok' if success else 'FAILED'}")
        if not success and failed_skill is None:
            failed_skill = skill.id
            env.monitor.failed_skill = env.monitor.failed_skill or skill.id
        # Without the cube in hand the remaining skills cannot place it.
        if skill.id == 'Pick' and not detector.grasped(cube):
            break
    return observations, tracking_data, failed_skill


def run_episode(env, detector, monitor, task, skills, episode, args):
    seed = args.seed + episode
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for skill in skills:
        if hasattr(skill, "reset_tracking"):
            skill.reset_tracking()

    start = time.time()
    monitor.start_episode()
    plan, goals, observations, unsolvable_resets = reset_until_planned(env, detector, task, args.n_act, args.render)
    print(f"Plan ({len(plan)} ops): {plan}\nGoal predicates: {goals}", flush=True)
    original_plan = list(plan)
    moves = parse_moves(plan)
    total_moves = len(moves)

    tracking_data = {}
    moves_ok = 0
    advancement = 0.0
    replans = 0
    failure_skill = None
    failure_move = None
    failure_reason = None
    terminal_reason = None
    open_action = np.array([0, 0, 0, -1])

    monitor.arm()
    try:
        reset_gripper(env, detector, args.render)
        observations = collect_observations(env, detector, args.n_act, open_action)
        index = 0
        while index < len(moves):
            move = moves[index]
            cube, _, destination = move
            state = get_state(detector)
            missing = missing_preconditions(state, detector, move)
            if missing:
                terminal_reason = f"preconditions_unmet:{move}:{missing}"
                break
            print(f"Move {index + 1}/{len(moves)}: {cube} -> {destination}", flush=True)
            if cubes_below(state, cube) >= TALL_PICK_LEVEL:
                lift_to_home_z(env, TALL_PICK_HOME_Z, args.render)
                observations = collect_observations(env, detector, args.n_act, open_action)
            monitor.set_move(cube, destination)
            observations, tracking_data, failed_skill = execute_move(
                env, detector, skills, move, goals, observations, tracking_data, args)
            # The effect only counts if it survives the gripper leaving.
            monitor.skill = "retreat"
            reset_gripper(env, detector, args.render)
            # Without this the next skill's first frames predate the retreat.
            observations = collect_observations(env, detector, args.n_act, open_action)

            if place_effect_holds(detector, cube, destination):
                moves_ok += 1
                index += 1
                remaining = len(moves) - index
                progress = (total_moves - remaining) / total_moves if args.replan else moves_ok / total_moves
                advancement = max(advancement, min(1.0, max(0.0, progress)))
                print(f"+++ Move verified. moves_ok={moves_ok}/{total_moves}, advancement={advancement:.3f}", flush=True)
                continue

            print(f"--- Move failed (first failed skill: {failed_skill}). moves_ok={moves_ok}/{total_moves}", flush=True)
            if failure_move is None:
                failure_skill = failed_skill or "effect_not_verified"
                failure_move = list(move)
            if not args.replan:
                index += 1
                continue
            if replans >= MAX_REPLANS:
                terminal_reason = f"replan_limit:{MAX_REPLANS}"
                break
            replans += 1
            state = get_state(detector)
            if any(detector.grasped(other) for other in detector.objects):
                terminal_reason = "replan_cube_still_grasped"
                break
            new_plan, _, _, _ = plan_for_state(state, detector, task, goals=goals)
            if new_plan is False:
                terminal_reason = "replan_unsolvable"
                break
            print(f"Replan #{replans} ({len(new_plan)} ops): {new_plan}", flush=True)
            moves = parse_moves(new_plan)
            index = 0
    except EpisodeTimeout as timeout:
        terminal_reason = f"timeout_{NO_PROGRESS_LIMIT}: {timeout}"
        print(f"[TIMEOUT] {timeout}", flush=True)
        failure_skill = failure_skill or monitor.failed_skill or monitor.skill
        if failure_move is None and monitor.move is not None:
            failure_move = list(moves[index])

    timed_out = terminal_reason is not None and terminal_reason.startswith("timeout_")
    monitor.disarm()
    if not timed_out:
        collect_observations(env, detector, SETTLE_STEPS, open_action)
    unmet = failed_goals(detector, goals)
    success = not unmet
    if success:
        advancement = 1.0
    else:
        failure_reason = terminal_reason or ("skill_failed" if failure_move else "goal_not_held_at_end")
    print(("Episode SUCCEEDED" if success else f"Episode FAILED ({failure_reason}); unmet goals: {unmet}"), flush=True)

    return {
        "episode": episode,
        "seed": seed,
        "base_seed": args.seed,
        "task": task.name,
        "perception": args.perception,
        "policy_set": args.policy_set,
        "replan": args.replan,
        "success": success,
        "advancement": advancement,
        "moves_ok": moves_ok,
        "plan_len": len(original_plan),
        "num_moves": total_moves,
        "env_steps": monitor.steps,
        "wall_s": time.time() - start,
        "timed_out": timed_out,
        "failure_skill": None if success else failure_skill,
        "failure_move": None if success else failure_move,
        "failure_reason": failure_reason,
        "unmet_goals": unmet,
        "replans": replans,
        "plan": original_plan,
        "goals": goals,
        "unsolvable_resets": unsolvable_resets,
    }


def print_summary(summary, label):
    print("=" * 78)
    print(f"{summary['task']} | {label}")
    print(f"episodes={summary['episodes']} unsolvable_resets={summary['unsolvable_resets']}")
    print(f"success     {summary['success_pct']:6.1f}%  Wilson 95% "
          f"[{summary['success_wilson95_pct'][0]:.1f}, {summary['success_wilson95_pct'][1]:.1f}]  "
          f"paper {summary['paper_success_pct']}")
    print(f"advancement {summary['advancement_pct']:6.1f}%  95% CI "
          f"[{summary['advancement_ci95_pct'][0]:.1f}, {summary['advancement_ci95_pct'][1]:.1f}]  "
          f"paper {summary['paper_advancement_pct']}")
    print("=" * 78, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', type=str, default=None, choices=list(PAPER_TASKS), help='Paper task to evaluate')
    parser.add_argument('--env', type=str, default=None, choices=list(LEGACY_ENVS),
                        help='Legacy: run a bare environment with no paper configuration. Prefer --task.')
    parser.add_argument('--perception', type=str, default='vision', choices=['vision', 'gt'],
                        help='vision: YOLO + regressor. gt: simulator poses, for isolating policy behaviour.')
    parser.add_argument('--gt_perception', action='store_true', help='Deprecated alias for --perception gt')
    parser.add_argument('--policy_set', type=str, default='original', choices=list(POLICY_SETS),
                        help='original: policies/hanoi_original with gripper_close_sign=-1. '
                             'retrained: policies/hanoi with gripper_close_sign=+1.')
    parser.add_argument('--regressor', type=str, default='models/regressors/nsm_regressor.pkl', help='Regressor path')
    parser.add_argument('--yolo', type=str, default='models/yolo/hanoi_yolo.pt', help='YOLO weights path')
    parser.add_argument('--n_inference_steps', type=int, default=8, help='Diffusion denoising steps (paper section 6.1)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to run')
    parser.add_argument('--replan', action='store_true',
                        help='Replan from the detector state after a failed move (not in the paper)')
    parser.add_argument('--seed', type=int, default=0, help='Base seed; episode i uses seed+i')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for result files')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing results for this run')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--size', type=int, default=256, help='Size of the rendered images')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--n_act', type=int, default=4, help='Number of actions to execute per policy call')
    parser.add_argument('--env_kwargs', type=str, default='{}',
                        help='JSON dict overriding the task env kwargs; the run no longer matches the paper setup')
    args = parser.parse_args()
    if bool(args.task) == bool(args.env):
        parser.error("Pass exactly one of --task and --env")
    if args.gt_perception:
        args.perception = 'gt'
    return args


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    task = PAPER_TASKS[args.task] if args.task else legacy_task(args.env)
    env_kwargs_override = json.loads(args.env_kwargs)
    use_vision = args.perception == 'vision'
    label = VISION_LABEL if use_vision else GT_LABEL
    print("#" * 78)
    print(f"# {label}")
    print(f"# task={task.name} env={task.env_name} policy_set={args.policy_set} "
          f"replan={args.replan} episodes={args.episodes} seed={args.seed}")
    if env_kwargs_override:
        print(f"# WARNING: env kwargs overridden, not the paper setup: {env_kwargs_override}")
    if task.notes:
        print(f"# NOTE: {task.notes}")
    print("#" * 78, flush=True)

    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.join(args.results_dir, f"{task.name}_{args.perception}_seed{args.seed}")
    results_path = f"{stem}.jsonl"
    if os.path.exists(results_path) and not args.overwrite:
        raise SystemExit(f"{results_path} already exists; pass --overwrite or another --results_dir")

    required = [args.yolo, args.regressor] if use_vision else []
    for path in required:
        if not os.path.exists(path):
            raise SystemExit(f"Missing perception model {path}; pass --yolo / --regressor")

    policy_set = POLICY_SETS[args.policy_set]
    policy_path = lambda name: os.path.join(policy_set["directory"], f"{name}.ckpt")
    executor_kwargs = dict(
        oracle=True,
        use_yolo=use_vision,
        debug=args.debug,
        num_inference_steps=args.n_inference_steps,
        gripper_close_sign=policy_set["gripper_close_sign"],
        color_to_pddl=task.color_to_pddl,
    )
    reach_pick = Executor_Diffusion(id='ReachPick', policy=policy_path('reach_pick'),
                                    Beta=termination_indicator('reach_pick'), nulified_action_indexes=[3],
                                    horizon=8 / args.n_act * 25, **executor_kwargs)
    pick = Executor_Diffusion(id='Pick', policy=policy_path('grasp'),
                              Beta=termination_indicator('pick'), nulified_action_indexes=[0, 1],
                              horizon=8 / args.n_act * 25, **executor_kwargs)
    reach_place = Executor_Diffusion(id='ReachDrop', policy=policy_path('reach_drop'),
                                     Beta=termination_indicator('reach_drop'), nulified_action_indexes=[3],
                                     horizon=8 / args.n_act * 35, **executor_kwargs)
    drop = Executor_Diffusion(id='Drop', policy=policy_path('drop'),
                              Beta=termination_indicator('drop'), nulified_action_indexes=[0, 1],
                              horizon=8 / args.n_act * 25, **executor_kwargs)
    skills = [reach_pick, pick, reach_place, drop]

    raw_env = make_env(
        task,
        env_kwargs_override=env_kwargs_override,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=args.size,
        camera_widths=args.size,
    )
    detector = make_detector(task, raw_env)
    monitor = ProgressMonitor(detector)
    env = DictObs(GymWrapper(raw_env), monitor)

    yolo_model = YOLO(args.yolo) if use_vision else None
    regressor_model = joblib.load(args.regressor) if use_vision else None
    for skill in skills:
        skill.load_policy(detector=detector, yolo_model=yolo_model, regressor_model=regressor_model, image_size=args.size)

    records = []
    summary_path = f"{stem}.summary.json"
    with open(results_path, 'w') as results_file:
        for episode in range(args.episodes):
            print(f"\n[{label}] Episode {episode + 1}/{args.episodes} (seed {args.seed + episode})", flush=True)
            record = run_episode(env, detector, monitor, task, skills, episode, args)
            record["perception_label"] = label
            records.append(record)
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

            summary = summarize(records, task)
            summary.update(perception=args.perception, perception_label=label, policy_set=args.policy_set,
                           replan=args.replan, n_inference_steps=args.n_inference_steps, seed=args.seed,
                           regressor=args.regressor if use_vision else None, yolo=args.yolo if use_vision else None,
                           env_kwargs={**task.env_kwargs, **env_kwargs_override})
            with open(summary_path, 'w') as summary_file:
                json.dump(summary, summary_file, indent=2)
            print(f"Running: success {summary['successes']}/{summary['episodes']}, "
                  f"advancement {summary['advancement_pct']:.1f}%", flush=True)
            if hasattr(Executor_Diffusion, "gt_fallback_summary"):
                Executor_Diffusion.gt_fallback_summary()
    print_summary(summary, label)


if __name__ == "__main__":
    main()
