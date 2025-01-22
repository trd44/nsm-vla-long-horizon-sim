import copy
import time
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import csv
import stable_baselines3
import logging
import argparse
import json
from tarski import fstrips as fs
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from learning.learning_utils import *
from learning.custom_gym_wrapper import *
from learning.custom_eval_callback import CustomEvalCallback
from utils import *

# set up a logger here to log the terminal printouts for the training of each subgoal
logger = logging.getLogger('learning')
logger.setLevel(logging.INFO)

class MinimalWrapper(gym.Wrapper):
    def __init__(self, env, config, domain):
        super().__init__(env)
        self.env = env
        self.config = config
        self.domain = domain
        self.detector = load_detector(config=config, domain=domain, env=env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = self.action(action)
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        
        obs_from_detector = self.detector.get_obs()
        obs_from_unwrapped = self.env.unwrapped._get_observations()
        binary_obs = self.detector.detect_binary_states(self.env)
        dist = np.linalg.norm(obs_from_unwrapped['gripper1_pos'] - obs_from_detector['mug1_pos'])
        dist2 = np.linalg.norm(obs_from_detector['gripper1_pos'] - obs_from_detector['mug1_pos'])
        normalized = dist / np.linalg.norm(obs_from_unwrapped['gripper1_to_obj_max_possible_dist'])
        grasp = binary_obs['exclusively-occupying-gripper mug1 gripper1']
        if grasp:
            reward = 1
        else:
            reward = 1 - normalized
        done = done or grasp
        step_cost = - 1
        return obs, reward + step_cost, done, truncated, info
    
    
    def action(self, action):
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        gripper_opening_min = self.env.action_space.low[-1]
        gripper_opening_max = self.env.action_space.high[-1]
        gripper_opening_range = gripper_opening_max - gripper_opening_min
        gripper_close_threshold = gripper_opening_min + gripper_opening_range/3
        gripper_open_threshold = gripper_opening_max - gripper_opening_range/3
        if action[-1] < gripper_close_threshold: # close the gripper
            action[-1] = gripper_opening_min
        elif action[-1] > gripper_open_threshold: # open the gripper
            action[-1] = gripper_opening_max
        else: # do nothing
            action[-1] = 0
        return action 

if __name__ == '__main__':
    config:dict = load_config("config.yaml")
    algo = 'SAC'

    # parse model commandline args
    parser = argparse.ArgumentParser()
    # add default args according to config
    for arg, val in config['learning'][algo].items():
        parser.add_argument(f'--{arg}', type=type(val), default=val)
        if type(val) == list:
            parser.add_argument(f'--{arg}', type=json.loads, default=val)
    for arg, val in config['learning']['eval'].items():
        if arg in config['learning'][algo]:
            continue
        parser.add_argument(f'--{arg}', type=type(val), default=val)
        if type(val) == list:
            parser.add_argument(f'--{arg}', type=json.loads, default=val)
    parser.add_argument('--total_timesteps', type=int, default=config['learning']['learn_subgoal']['total_timesteps'])
    parser.add_argument('--domain', type=str, default='cleanup')
    parser.add_argument('--net_arch', type=json.loads, default=config['learning'][algo]['policy_kwargs']['net_arch'])
    parser.add_argument('--lr_schedule', type=bool, default=False)
    parser.add_argument('--op_wrap', type=bool, default=False, help='whether to wrap the environment with the operator wrapper')
    parser.add_argument('--rw_shaping', type=int, default=0, help='the reward shaping function candidate to use. Usually from 0 - 2.')
    # parser.add_argument('--domain', type=str, default='cleanup')
    # parser.add_argument('--rl_algorithm', type=str, default='PPO')
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--total_timesteps', type=int, default=2_500_000)
    # parser.add_argument('--n_steps', type=int, default=2048)
    # parser.add_argument('--eval_freq', type=int, default=10_000)
    # parser.add_argument('--n_eval_episodes', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--net_arch', type=str, default='[64, 64]')
    # parser.add_argument('--learning_rate', type=float, default=3e-4)
    # parser.add_argument('--lr_schedule', type=bool, default=False)
    args = parser.parse_args()

    domain = args.domain
    # print the non-default commandline args and construct kwargs for model and eval
    model_kwargs = {}
    eval_kwargs = {}
    save_path = f"./{algo}_approach_{domain}"
    # print the algorithm name and save path
    print(f"#######Training Algorithm: {algo} at {save_path}#######")
    for arg, val in vars(args).items():
        if val != parser.get_default(arg):
            print(f"{arg}: {val}")
            save_path += f"_{arg}_{val}"
        if arg == 'net_arch': # special case for net_arch
            if algo == 'PPO':
                model_kwargs['policy_kwargs'] = {'net_arch': {'pi': val, 'vi': val}}
            else:
                model_kwargs['policy_kwargs'] = {'net_arch': val}
        if arg in config['learning'][algo]:
            model_kwargs[arg] = val
        elif arg in config['learning']['eval']:
            eval_kwargs[arg] = val
    if args.lr_schedule:
        model_kwargs['learning_rate'] = linear_schedule(args.learning_rate)
    
    plan = load_plan(config['planning'][domain]) # load the plan in case the operator wrapper is used
    grounded_op = plan[0] # assuming the first operator is the one we need to learn. It's not necessarily the case
    curr_subgoal = grounded_op.effects[0] if 'not (free gripper1)' not in grounded_op.effects[0].pddl_repr() else grounded_op.effects[1] # assuming the first effect is the subgoal. It's not necessarily the case
    # load LLM generated reward function in case the operator wrapper is used
    op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
    reward_fn_candidates = []
    for i in range(config['learning']['reward_shaping_fn']['num_candidates']):
        try:
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{domain}.{op_name}_{i}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
            reward_fn_candidates.append(llm_reward_shaping_func)
        except:
            raise Exception(f"Reward function {op_name}_{i} not found")
        
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    gym_env = GymWrapper(robosuite_env)
    if args.op_wrap:
        wrapped_env = CollisionAblatedOperatorWrapper(gym_env, grounded_operator=grounded_op, executed_operators={}, config=config, domain=domain, rl_algo=algo, curr_subgoal=curr_subgoal)
        wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
    else:
        wrapped_env = MinimalWrapper(gym_env, config, domain)
    env = Monitor(wrapped_env, filename=f'{save_path}{os.sep}approach_monitor', allow_early_resets=True)

    eval_robosuite_env = load_env(domain, config['simulation'])
    eval_gym_env = GymWrapper(eval_robosuite_env)
    if args.op_wrap:
        eval_wrapped_env = CollisionAblatedOperatorWrapper(gym_env, grounded_operator=grounded_op, executed_operators={}, config=config, domain=domain, rl_algo=algo, curr_subgoal=curr_subgoal)
        eval_wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
    else:
        eval_wrapped_env = MinimalWrapper(eval_gym_env, config, domain)
    eval_env = Monitor(eval_wrapped_env, filename=f'{save_path}{os.sep}approach_eval_monitor', allow_early_resets=True)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    # load the algo from stable_baselines3
    rl_algo = importlib.import_module(f"stable_baselines3.{algo.lower()}").__dict__[algo.upper()]
    # load model if it exists
    if os.path.exists(f"./{save_path}{os.sep}best_model{os.sep}best_model.zip"):
        model = rl_algo.load(f"./{save_path}{os.sep}best_model{os.sep}best_model.zip", env)
    else:
        # create model based on commandline args
        model = rl_algo("MlpPolicy", env, seed=0, **model_kwargs, tensorboard_log=f"./{save_path}{os.sep}tensorboard/")

    # create the logger
    # set up a logger here to log the terminal printouts for the training of each subgoal
    logger = logging.getLogger('learning')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): # Remove duplicate handlers
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{save_path}{os.sep}learner_train_logs.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model.learn(total_timesteps=args.total_timesteps, callback=CustomEvalCallback(eval_env=eval_env, logger=logger, best_model_save_path=f"./{save_path}{os.sep}best_model/", log_path=f"./{save_path}{os.sep}logs/", **eval_kwargs))

    model.save(f"./{save_path}{os.sep}model")
