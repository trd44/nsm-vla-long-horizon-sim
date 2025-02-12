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

    #Reset the environment
    try:
        obs, info = env.reset()
    except Exception as e:
        obs = env.reset()
        info = None
    
    state = info['state']
    obj_to_pick = 'SquareNut'
    gripper_body = env.sim.model.body_name2id('gripper0_eef')


    for i in range(1000):
        while not state['over(gripper,{})'.format(obj_to_pick)]:
            gripper_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id('gripper0_eef')])
            object_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(f'{obj_to_pick}_main')])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            next_obs, reward, _, _, info = env.step(action)
            env.render()
            state = info['state']
            # filter and keep only the predicates that are "over"
            # state = {predicate: state[predicate] for predicate in state.keys() if 'over' in predicate}
            # print("State: ", state)

        # Shift slightely to the right
        print("Shifting slightly to the left...")
        for _ in range(10):
            action = np.asarray([0,0.5,0,0])
            action = action
            next_obs, reward, _, _, info = env.step(action)
            env.render()
            state = info['state']

        print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            action = action
            next_obs, reward, _, _, info = env.step(action)
            env.render()
            state = info['state']

        print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(obj_to_pick)]:
            gripper_pos = np.asarray(env.sim.data.body_xpos[gripper_body])
            object_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(f'{obj_to_pick}_main')])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            next_obs, _, _, _, info  = env.step(action)
            env.render()
            state = info['state']

        print("Closing gripper...")
        while not state['grasped({})'.format(obj_to_pick)]:
            action = np.asarray([0,0,0,1])
            next_obs, _, _, _, info  = env.step(action)
            env.render()
            state = info['state']

        # Reset the environment
        obs, info = env.reset()


    # state = info[0]['state'][-1]
    # # Detect the state of the environment
    # # detector = Robosuite_Hanoi_Detector(env)
    # # state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
    # # print("Initial state: ", state)

    # # Create a lambda function that maps "on(cube1,peg1)" to "p1(o1,o3)"
    # def map_predicate(predicate):
    #     # Extract the objects from the predicate
    #     objects = predicate.split('(')[1].split(')')[0].split(',')
    #     # Map the objects to their corresponding ids
    #     obj_mapping = {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o6', 'peg1': 'o3', 'peg2': 'o4', 'peg3': 'o5'}
    #     # Map the predicate to the PDDL format
    #     return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[1]]})"
    # def change_predicate(predicate):
    #     # Extract the objects from the predicate
    #     objects = predicate.split('(')[1].split(')')[0].split(',')
    #     # Change clear(cube1) to p1(o1,o1)
    #     obj_mapping = {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o6', 'peg1': 'o3', 'peg2': 'o4', 'peg3': 'o5'}
    #     return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[0]]})"
    # # Filter and keep only the predicates that are "on" and are True and map them to the PDDL format
    # init_predicates = {map_predicate(predicate): True for predicate in state.keys() if 'on' in predicate and state[predicate]}
    # # Filter and keep only the predicates that are "clear" and are True and map them to the PDDL format
    # init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})
    # print("Initial predicates: ", init_predicates)

    # # Usage
    # add_predicates_to_pddl('problem_static.pddl', init_predicates)

    # # Generate a plan
    # plan, _ = call_planner("domain_asp", "problem_dummy")
    # print("Plan: ", plan)

