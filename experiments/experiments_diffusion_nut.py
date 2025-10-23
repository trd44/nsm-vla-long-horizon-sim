import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from statistics import mean 
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.nutassembly.assembly import AssemblyWrapper
from robosuite.wrappers.nutassembly.object_state import AssembleStateWrapper
from PDDL.planner import *
from PDDL.executor import *

# diffusion policy import
from typing import Dict
import numpy as np
from tqdm.auto import tqdm

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from ultralytics import YOLO
import joblib

# env import
import gym
#import pymunk.pygame_util

if __name__ == "__main__":

    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--demos', type=int, default=5, help='number of demos')
    parser.add_argument('--use_yolo', action='store_true', help='Use YOLO for object detection')
    args = parser.parse_args()
    #np.random.seed(args.seed)

    yolo_model = YOLO("PDDL/yolo_nutassembly.pt")
    regressor_model = joblib.load("data/nutassembly_dual_cam_calibration_models.pkl")

    def termination_indicator(operator):
        if operator == 'pick':
            def Beta(state, symgoal):
                condition = state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'drop':
            def Beta(state, symgoal):
                condition = state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'reach_pick':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[0]})"]
                return condition
        elif operator == 'reach_drop':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[1]})"]
                return condition
        else:
            def Beta(state, symgoal):
                condition = False
                return condition
        return Beta
    

    # Create an env wrapper which transforms the outputs of reset() and step() into gym formats (and not gymnasium formats)
    class GymnasiumToGymWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 10

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        def reset(self):
            obs, info = self.env.reset()
            #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
            #obs = np.concatenate([
            #    keypoint, 
            #    obs], axis=-1)
            #obs = np.concatenate([obs, info["keypoint"]])
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
            #obs = np.concatenate([
            #    keypoint, 
            #    obs], axis=-1)
            return obs, reward, terminated or truncated, info

        def render(self, mode='human', *args, **kwargs):
            self.env.render()

        def close(self):
            self.env.close()

        def seed(self, seed=None):
            self.env.seed(seed)

        def set_task(self, task):
            self.env.set_task(task)

    # Create an env wrapper which transforms the outputs of reset() and step() into gym formats (and not gymnasium formats)
    class fourWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 4

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    class threeWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 3

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    class twoWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 2

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    class tenWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 10

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    # # Load executors
    place = Executor_Diffusion(id='PlaceNut', 
                       #policy=f"./nut_policies/{args.demos}_demos/nut_place.ckpt", 
                       policy="/home/lorangpi/CyclicLxM/yolo_policies/nut_yolo/reachplace.ckpt",
                       I={}, 
                       Beta=termination_indicator('reach_drop'),
                       nulified_action_indexes=[3],
                       wrapper = tenWrapper,
                       horizon=20,
                       oracle=True,
                       use_yolo=args.use_yolo,
                       yolo_model=yolo_model,
                       regressor_model=regressor_model)
    
    reach_pick = Executor_Diffusion(id='ReachPick', 
                            #policy="/home/lorangpi/Enigma/saved_policies/reach_pick/epoch=7900-train_loss=0.008.ckpt",
                            # WORKING POLICY BELOW
                            #policy="/home/lorangpi/Enigma/saved_policies_27u/reach_pick/epoch=2550-train_loss=0.062.ckpt",
                            #policy=f"./nut_policies/{args.demos}_demos/reach_pick.ckpt",
                            #policy=f"./policies/neurosym_{args.demos}/reach_pick.ckpt",
                            policy="/home/lorangpi/CyclicLxM/yolo_policies/nut_yolo/reachpick.ckpt",
                            I={}, 
                            Beta=termination_indicator('reach_pick'),
                            nulified_action_indexes=[3],
                            oracle=True,
                            wrapper = tenWrapper,
                            horizon=10,
                            use_yolo=args.use_yolo,
                            yolo_model=yolo_model,
                            regressor_model=regressor_model)
    grasp = Executor_Diffusion(id='Grasp', 
                    #policy="/home/lorangpi/Enigma/saved_policies/grasp/epoch=7700-train_loss=0.021.ckpt", 
                    # WORKING POLICY BELOW
                            #policy="/home/lorangpi/Enigma/saved_policies_27u/grasp/epoch=3250-train_loss=0.027.ckpt",
                    #policy=f"./nut_policies/{args.demos}_demos/grasp.ckpt",
                    #policy=f"./policies/neurosym_{args.demos}/grasp.ckpt",
                    policy="/home/lorangpi/CyclicLxM/yolo_policies/nut_yolo/pick.ckpt",
                    I={}, 
                    Beta=termination_indicator('pick'),
                    nulified_action_indexes=[0, 1],
                    oracle=True,
                    wrapper = tenWrapper,
                    horizon=10,
                    use_yolo=args.use_yolo,
                    yolo_model=yolo_model,
                    regressor_model=regressor_model)
    drop = Executor_Diffusion(id='Drop', 
                    # WORKING POLICY BELOW
                            #policy="/home/lorangpi/Enigma/saved_policies_27u/drop/epoch=3350-train_loss=0.051.ckpt",
                    #policy=f"./nut_policies/{args.demos}_demos/drop.ckpt",
                    policy="/home/lorangpi/CyclicLxM/yolo_policies/nut_yolo/place.ckpt",
                    I={}, 
                    Beta=termination_indicator('drop1'),
                    nulified_action_indexes=[0, 1],
                    oracle=True,
                    wrapper = tenWrapper,
                    horizon=4,
                    use_yolo=args.use_yolo,
                    yolo_model=yolo_model,
                    regressor_model=regressor_model)

    Move_action = [reach_pick, grasp, place, drop]


    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

    device = "cpu"
    def env_fn():
        env = suite.make(
            "NutAssembly",
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            horizon=20000,
            use_camera_obs=False,
            render_camera="robot0_eye_in_hand",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        )

        # Wrap the environment
        env = GymWrapper(env)
        env = AssemblyWrapper(env, horizon=2000, render=args.render)
        env = AssembleStateWrapper(env)
        env = GymnasiumToGymWrapper(env)
        env = MultiStepWrapper(
            env=env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
        return env

    n_obs_steps = 8
    n_action_steps = 8
    max_steps = 20000
    env_fns = [env_fn]
    dummy_env = env_fn()

    print(dummy_env.observation_space)

    obs_dim = 10
    high = np.inf * np.ones(obs_dim)
    low = -high
    observation_space = gym.spaces.Box(low, high, dtype=np.float64)
    action_space = gym.spaces.Box(low=dummy_env.action_space.low, high=dummy_env.action_space.high, dtype=np.float64)

    print(observation_space)


    def gen_dummy_env():
        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            # env.observation_space = gym.spaces.Box(
            #     -8, 8, shape=(15,), dtype=np.float32)
            # env.action_space = gym.spaces.Box(
            #     -8, 8, shape=(4,), dtype=np.float32)
            env.observation_space = observation_space
            env.action_space = action_space
            env.metadata = {
                'render.modes': ['human', 'rgb_array', 'depth_array'],
                'video.frames_per_second': 12
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env
        return dummy_env_fn

    env = AsyncVectorEnv(env_fns, dummy_env_fn=gen_dummy_env(), shared_memory=False)

    #Reset the environment
    env.set_task(("squarenut", "roundpeg"))
    try:
        obs, info = env.reset()
    except Exception as e:
        obs = env.reset()

    obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
    print("Info: ", info)
    state = info[0]['state'][-1]
    # Detect the state of the environment
    # detector = Robosuite_Hanoi_Detector(env)
    # state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
    # print("Initial state: ", state)

    # Create a lambda function that maps "on(cube1,peg1)" to "p1(o1,o3)"
    def map_predicate(predicate):
        # Extract the objects from the predicate
        objects = predicate.split('(')[1].split(')')[0].split(',')
        # Map the objects to their corresponding ids
        obj_mapping = {'roundnut': 'o1', 'squarenut': 'o2', 'roundpeg': 'o3', 'squarepeg': 'o4', 'table': 'o5'}
        # Map the predicate to the PDDL format
        return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[1]]})"
    def change_predicate(predicate):
        # Extract the objects from the predicate
        objects = predicate.split('(')[1].split(')')[0].split(',')
        # Change clear(cube1) to p1(o1,o1)
        obj_mapping = {'roundnut': 'o1', 'squarenut': 'o2', 'roundpeg': 'o3', 'squarepeg': 'o4', 'table': 'o5'}
        return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[0]]})"
    # Filter and keep only the predicates that are "on" and are True and map them to the PDDL format
    init_predicates = {map_predicate(predicate): True for predicate in state.keys() if 'on' in predicate and state[predicate]}
    # Filter and keep only the predicates that are "clear" and are True and map them to the PDDL format
    init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})
    print("Initial predicates: ", init_predicates)

    # Usage
    add_predicates_to_pddl('problem_staticassembly.pddl', init_predicates)

    # Generate a plan
    plan, _ = call_planner("domain_assembly", "problem_dummyassembly")
    print("Plan: ", plan)

    obj_mapping = {'o1': 'roundnut', 'o2': 'squarenut', 'o3': 'roundpeg', 'o4': 'squarepeg', 'o5': 'table'}

    reset_gripper_pos = np.array([-0.14193391, -0.03391656,  0.95828137])
    hanoi_successes = 0
    num_valid_pick_place_queries = 0
    pick_place_success = 0
    pick_place_successes = []
    percentage_advancement = []
    valid_pick_place_success = 0

    def reset_gripper(env):
        print("Resetting gripper")
        # First move up
        for _ in range(5):
            action = np.array([0, 0, 0.5, 0])
            obs, reward, done, info = env.step([[action, action, action, action]])
        # Second move to the initial position
        obs = obs[-1][-1]
        current_pos = obs[:3]
        delta = reset_gripper_pos - current_pos
        action = 5*np.array([delta[0], delta[1], delta[2], 0])
        while np.linalg.norm(delta) > 0.01:
            #print("Curent pos: ", current_pos)
            action = 5*np.array([delta[0], delta[1], delta[2], 0])
            action = action * 0.9
            obs, reward, done, info = env.step([[action, action, action, action]])
            obs = obs[-1][-1]
            current_pos = obs[:3]
            delta = reset_gripper_pos - current_pos
            #print(f"Delta: {delta}, Current pos: {current_pos}, Reset pos: {reset_gripper_pos}")

    for i in range(100):
        print("Episode: ", i)
        success = False
        valid_state = False
        plan = False
        np.random.seed(args.seed + i)
        # Reset the environment until a valid state is reached
        while plan == False:
            # Reset the environment
            try:
                obs, info = env.reset()
            except Exception as e:
                obs = env.reset()
            obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
            state = info[0]['state'][-1]
            # Generate the plan
            init_predicates = {map_predicate(predicate): True for predicate in state.keys() if 'on' in predicate and state[predicate]}
            init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})

            add_predicates_to_pddl('problem_staticassembly.pddl', init_predicates)
            plan, _ = call_planner("domain_assembly", "problem_dummyassembly")
        print("Plan: ", plan)

        pick_place_success = 0
        # Execute the first operator in the plan
        reset_gripper(env)
        for operator in plan:
            #operator = "MOVE O5 O2 O4"
            print("\nExecuting operator: ", operator)
            # Concatenate the observations with the operator effects
            obj_to_pick = obj_mapping[operator.split(' ')[2].lower()]
            obj_to_drop = obj_mapping[operator.split(' ')[3].lower()]
            env.set_task((obj_to_pick, obj_to_drop))
            print("Set attributes: ", obj_to_pick, obj_to_drop)
            obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
            print("Picking object: {}, Dropping object: {}".format(obj_to_pick, obj_to_drop))
            #pick_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_pick]][:3]
            #drop_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_drop]][:3]
            num_valid_pick_place_queries += 1
            for action_step in Move_action:
                #if action_step.model == None:
                action_step.load_policy()
                print("\tExecuting action: ", action_step.id)
                symgoal = (obj_to_pick, obj_to_drop)
                goal = []
                obs, success = action_step.execute(env, obs, goal, symgoal, render=args.render)
                if not success:
                    print("Execution failed.\n")
                    #break
            if success:
                pick_place_success += 1
                valid_pick_place_success += 1
                print("+++ Object successfully picked and placed.")
                print(f"Successfull operations: {pick_place_success}, Out of: {len(plan)}, Percentage advancement: {pick_place_success/len(plan)}")
                if operator == plan[-1]:
                    continue
            else:
                continue
                # Print the number of operators that were successfully executed out of the total number of operators in the plan
                print("--- Object not picked and placed.")
                print(f"Successfull operations: {pick_place_success}, Out of: {len(plan)}, Percentage advancement: {pick_place_success/len(plan)}")
                break
            reset_gripper(env)#, obs[-1][-1])
            # Move up the gripper again
            #for _ in range(5):
            #    action = np.array([0, 0, 500, 0])
            #    obs, reward, done, info = env.step([[action, action, action, action]])
        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success/len(plan))
        if success:
            hanoi_successes += 1
            print("Hanoi Execution succeeded.\n")
        print("Success rate: ", hanoi_successes/(i+1))
        print("\n\n")

        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(pick_place_success/len(plan))

        print("Successfull pick_place: ", pick_place_successes)
        print("Percentage advancement: ", percentage_advancement)
        print("Mean Successful pick_place: ", mean(pick_place_successes))
        print("Mean Percentage advancement: ", mean(percentage_advancement))
        print("Pick placce success rate: ", valid_pick_place_success/num_valid_pick_place_queries)

        print("Success rate: ", hanoi_successes/(i+1))
        # Write the results to a file results_seed_{args.seed}.txt
        os.makedirs("results", exist_ok=True)
        with open(f"results/results_neurosym_{args.demos}_seed_{args.seed}.txt", 'w') as file:
            file.write("Success rate: {}\n".format(hanoi_successes/(100)))
            file.write("Mean Successful pick_place: {}\n".format(mean(pick_place_successes)))
            file.write("Mean Percentage advancement: {}\n".format(mean(percentage_advancement)))
            file.write("Pick placce success rate: {}\n".format(valid_pick_place_success/num_valid_pick_place_queries))

