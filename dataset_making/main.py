"""Main script for recording demonstrations of the Robosuite environments."""
import os
import time
import numpy as np
import imageio
import tyro

from dataset_making.args import Args
from dataset_making.utils import to_datestring
from dataset_making.record_demos import RecordDemos
# from dataset_making.panda_hanoi_detector import (
#     PandaHanoiDetector as HanoiDetector
# )

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import (
    HanoiDetector,
    KitchenDetector, 
    NutAssemblyDetector, 
    CubeSortingDetector,
    HeightStackingDetector,
    AssemblyLineSortingDetector,
    PatternReplicationDetector
)


def make_env(args):
    """Create and return a wrapped robosuite environment."""
    ctrl_cfg = suite.load_controller_config(default_controller='OSC_POSE')
    if args.env[:5] == 'Hanoi':  # Hanoi environments
        env = suite.make(
            args.env,
            robots=args.robot,
            controller_configs=ctrl_cfg,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=1e8,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            random_block_placement=args.random_block_placement,
            random_block_selection=args.random_block_selection,
            cube_init_pos_noise_std=args.cube_placement_noise,
            peg_xy_jitter=args.peg_xy_jitter
        )
        print(f"Peg XY jitter: {args.peg_xy_jitter}")
        print(f"Environment settings: \
            random_block_placement={args.random_block_placement}, \
            random_block_selection={args.random_block_selection}")
        
    else:  # Other environments
        env = suite.make(
            args.env,
            robots=args.robot,
            controller_configs=ctrl_cfg,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=1e8,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            cube_placement_noise=args.cube_placement_noise,
        )

    # Gym-compatible wrapper, proprio_obs toggles vision-based vs proprio inputs
    env = GymWrapper(env, proprio_obs=not args.vision)
    return env


def get_detector(env, env_name):
    """Instantiate the correct detector for the chosen environment."""
    if env_name == 'AssemblyLineSorting':
        return AssemblyLineSortingDetector(env)
    if env_name == 'CubeSorting':
        return CubeSortingDetector(env)
    if env_name == 'Hanoi' or env_name == 'Hanoi4x3':
        return HanoiDetector(env)
    if env_name == 'HeightStacking':
        return HeightStackingDetector(env)
    if env_name == 'KitchenEnv':
        return KitchenDetector(env)
    if env_name == 'NutAssembly':
        return NutAssemblyDetector(env)
    if env_name == 'PatternReplication':
        return PatternReplicationDetector(env)
    raise ValueError(f"Unknown env {env_name}")


def perform_demonstration(
    args,
    env,
    recorder,
    attempt,
    successes,
    fps: int = 30,
) -> bool:
    """Record a single episode video and return success status."""
    agent_camera="agentview"
    wrist_camera="robot0_eye_in_hand"
    save_hd_agent_video: bool = args.save_hd_agent_video
    save_hd_wrist_video: bool = args.save_hd_wrist_video
    if save_hd_agent_video:
        hd_agent_frames = []
    if save_hd_wrist_video:
        hd_wrist_frames = []

    # Hook into the recorder so that every time it records a step,
    # we grab a frame from both cameras
    original_env_record = getattr(recorder.env, "record_step", None)
    def hooked_env_record(obs, action):
        if save_hd_agent_video:
            f = env.sim.render(width=640, height=480, camera_name=agent_camera)
            hd_agent_frames.append(f)
        if save_hd_wrist_video:
            f = env.sim.render(width=640, height=480, camera_name=wrist_camera)
            hd_wrist_frames.append(f)
        return recorder.record_step(obs, action)
    
    # Override the environment's record_step method with our hooked version to 
    # capture video frames during each step
    recorder.env.record_step = hooked_env_record

    # Reset and grab the initial frame from both cameras
    obs = recorder.reset(successes=successes)
    if args.verbose:
        print(f"Obs: {obs}")
        print(f"Plan: {recorder.plan}")
    
    # Used to be an issue with the plan being None. Haven't seen it in a while.
    # TODO: Remove?
    while recorder.plan is None or obs is None:
        obs = recorder.reset(successes=successes)
        print("Plan is None, resetting")
    
    # Save initial frames. Not captured normally?
    if save_hd_agent_video:
        f = env.sim.render(width=640, height=480, camera_name=agent_camera)
        hd_agent_frames.append(f)
    if save_hd_wrist_video:
        f = env.sim.render(width=640, height=480, camera_name=wrist_camera)
        hd_wrist_frames.append(f)
    
    # Run the whole trajectory (will call hooked_record_step under the hood)
    success = recorder.run_trajectory(obs)

    # Finally grab one more frame from both cameras
    if save_hd_agent_video:
        f = env.sim.render(width=640, height=480, camera_name=agent_camera)
        hd_agent_frames.append(f)
        filename = f"episode_{attempt:03d}_seed{args.seed}_agentview.mp4"
        path = os.path.join('dataset_making', 'hd_videos', args.env, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Flip frames vertically to fix upside-down issue
        flipped_frames = [np.flipud(frame) for frame in hd_agent_frames]
        imageio.mimwrite(path, flipped_frames, fps=fps, macro_block_size=None)
        print(f"Saved hd agentview video to {path}")
        print(f"(success={success}, frames={len(hd_agent_frames)})")

    if save_hd_wrist_video:
        f = env.sim.render(width=640, height=480, camera_name=wrist_camera)
        hd_wrist_frames.append(f)
        filename = f"episode_{attempt:03d}_seed{args.seed}_wrist.mp4"
        path = os.path.join('dataset_making', 'hd_videos', args.env, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Flip frames vertically to be consistent with agentview video
        flipped_frames = [np.flipud(frame) for frame in hd_wrist_frames]
        imageio.mimwrite(path, flipped_frames, fps=fps, macro_block_size=None)
        print(f"Saved hd wrist video to {path}")
        print(f"(success={success}, frames={len(hd_wrist_frames)})")

    # Restore original hook in case you want to reuse recorder
    if original_env_record:
        recorder.env.record_step = original_env_record
    else:
        del recorder.env.record_step

    return success


def main(args: Args) -> None:
    """Main function for running the dataset making script"""
    # Seed everything
    np.random.seed(args.seed)

    # Build experiment directory structure
    exp_id = args.name if args.name else to_datestring(time.time())    
    args.env_dir = os.path.join(args.dir, args.env, exp_id)    
    os.makedirs(args.env_dir, exist_ok=True)
    print(f"Saving dataset to {args.env_dir}")

    pddl_path = os.path.join('planning', 'PDDL', args.env.lower()) + os.sep

    # 1) Make the env
    env = make_env(args)

    # 2) Create the detector
    detector = get_detector(env, args.env)
    
    # 3) Create the recorder
    recorder = RecordDemos(
        args=args,
        env=env,
        detector=detector,
        pddl_path=pddl_path,
        randomize=True,
    )

    attempt = 0
    successes = 0   
    while successes < args.episodes:
        print(f"\n=== Attempt {attempt} ===")
        success = perform_demonstration(args, env, recorder, attempt, successes)        
        if success:
            recorder.save_trajectory(attempt)
            successes += 1
        else:
            print(f"Attempt {attempt} failed.")            
        attempt += 1
        print(f"\n=== Successes {successes}/{args.episodes} ===")


if __name__ == '__main__':
    tyro.cli(main)