import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import NutAssemblyDetector
from robosuite.wrappers.nutassembly.assembly import AssemblyWrapper
from robosuite.wrappers.nutassembly.grasp_roundnut import GraspRoundNutWrapper
from robosuite.wrappers.nutassembly.assemble_squarenut import AssembleSquareNutWrapper
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

# diffusion policy import
from typing import Dict
import numpy as np
from tqdm.auto import tqdm

# env import
import gymnasium as gym


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)


    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

    env = suite.make(
        "NutAssembly",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        horizon=1000000,
        use_camera_obs=True,
        use_object_obs=False,
        render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    )

    # Wrap the environment
    env = GymWrapper(env)
    env = AssembleSquareNutWrapper(env)

    device = Keyboard()
    env.viewer.add_keypress_callback(device.on_press)
    device.start_control()

    #Reset the environment
    try:
        obs, info = env.reset()
    except Exception as e:
        obs = env.reset()
        info = None
    
    state = info['state']
    obj_to_pick = 'RoundNut'
    gripper_body = env.sim.model.body_name2id('gripper0_eef')


    while True:
        # Set active robot
        active_robot = env.robots[0]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm="right", env_configuration="single-arm-opposed"
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested

        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            action = np.concatenate([action, rem_action])

        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        # Step through the simulation and render
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except:
            obs, reward, done, info = env.step(action)
        
        new_state = info['state']
        
        if new_state != state:
            # CHeck if the change is linked a grounded predicate 'on(o1,o2)' or 'clear(o1)'
            diff = {k: new_state[k] for k in new_state if k not in state or new_state[k] != state[k]}
            # If any key in diff has 'on' or 'clear' in it, print the change and the new state
            #if any(['on' in k or 'clear' in k for k in diff]):
            if any(['on' in k for k in diff]):
                print("Change detected: {}".format(diff))
                print("State: {}".format(new_state))
                print("\n\n")
                state = new_state


        #print("Obs: {}\n\n".format(obs))
        env.render()
