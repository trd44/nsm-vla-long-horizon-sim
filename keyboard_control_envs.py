# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script that offers an easy way to test random actions in a MimicGen environment.
Similar to the demo_random_action.py script from robosuite.
"""
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from detection.coffee_detector import CoffeeDetector
from detection.cleanup_detector import CleanupDetector
from detection.nut_assembly_detector import NutAssemblyDetector
from robosuite.wrappers import GymWrapper
from utils import *
from learning.learner import *
import numpy as np
import mimicgen

config = load_config("config.yaml")

def choose_mimicgen_environment():
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """

    # try to import robosuite task zoo to include those envs in the robosuite registry
    try:
        import robosuite_task_zoo
    except ImportError:
        print("Failed to import robosuite_task_zoo")
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # keep only envs that correspond to the different reset distributions from the paper
    # only keep envs that end with "Novelty"
    envs = sorted([x for x in all_envs if x[-7:] == "Novelty"])
    domain = config['planning']['domain']
    lower_case_domain = domain.lower().replace('_', '')
    envs = [env for env in envs if (lower_case_domain in env.lower() or domain in env) and 'pre_novelty' not in env.lower()]

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    print("Chosen environment: {}\n".format(envs[k]))
    # Return the chosen environment name
    return envs[k]


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment
    options["env_name"] =  choose_mimicgen_environment()

    # Choose robot
    options["robots"] = choose_robots(exclude_bimanual=True)

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        reward_shaping=True,
        hard_reset=False,
    )
    # Wrap this environment in a visualization wrapper
    config = load_config("config.yaml") # make sure planning is set to the nut_assembly domain
    plan = load_plan(config)
    grounded_op = plan[0] # a hack to get the pick-up-nut-from-peg operator
    dummy_executed_operators = OrderedDict()
    visual_env = VisualizationWrapper(env, indicator_configs=None)
    wrapped_env = GymWrapper(visual_env)
    wrapped_env = OperatorWrapper(wrapped_env, grounded_op, dummy_executed_operators, config)
    wrapped_env.reset()
    wrapped_env.viewer.set_camera(camera_id=0)

    # keyboard control
    from robosuite.devices import Keyboard

    device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
    wrapped_env.viewer.add_keypress_callback(device.on_press)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Get the camera ID for the camera you want to modify (e.g., "frontview")
    camera_id = env.sim.model.camera_name2id("agentview")

    # Set the new camera position (x, y, z)
    wrapped_env.sim.model.cam_pos[camera_id] = np.array([0, 0, 2])  # Modify these values to set the desired position
    env.sim.model.cam_quat[camera_id] = np.array([0.0, 0, 0, 1])  # Example quaternion
    
    # If options["env_name"] starts with "Co", use the Coffee_Detector
    if options["env_name"][:2] == "Co":
        detector = CoffeeDetector(env)
    elif options["env_name"][:2] == "Cu":
        detector = CleanupDetector(env)
    elif options["env_name"][:2] == "Nu":
        detector = NutAssemblyDetector(env)
    else:
        raise ValueError("Unrecognized environment name: {}".format(options["env_name"]))

    while True:
        # Reset the environment
        obs = wrapped_env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        wrapped_env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = wrapped_env.robots[0]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm='left', env_configuration="single-arm-opposed"
            )

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                # Update last grasp
                last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = wrapped_env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                
                action = np.concatenate([rem_action, action])
                
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: wrapped_env.action_dim]

            # Step through the simulation and render
            obs, reward, done, truncated, info = wrapped_env.step(action)
            groundings = detector.detect_binary_states(env)
            wrapped_env.render()

    # Get action limits
    # low, high = env.action_spec

    # # do visualization
    # for i in range(10000):
    #     action = np.random.uniform(low, high)
    #     obs, reward, done, _ = env.step(action)
    #     #detector.exclusively_occupying_gripper('coffee_pod')
    #     env.render()
