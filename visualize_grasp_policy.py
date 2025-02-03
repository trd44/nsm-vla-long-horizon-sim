# A minimal script to visualize the grasp policy trained with PPO in the `debug_training.py` script

import copy
import time
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import argparse
from learning.custom_gym_wrapper import *
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
from utils import *
from debug_training import MinimalWrapper


if __name__ == '__main__':
    # parse model commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_wrap', type=bool, default=True, help='whether to wrap the environment with the operator wrapper')
    parser.add_argument('--domain', type=str, default='cleanup', help='the domain of the environment')
    parser.add_argument('--seed', type=int, default=0, help='the seed for the environment')
    parser.add_argument('--rw_shaping', type=int, default=0, help='the reward shaping function to use')
    parser.add_argument('--algo', type=str, default='SAC', help='the algorithm to use')
    args = parser.parse_args()

    np.random.seed(args.seed)
    config:dict = load_config("config.yaml")
    domain:str = args.domain
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    gym_env = GymWrapper(robosuite_env)
    if args.op_wrap:
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
        wrapped_env = CollisionAblatedOperatorWrapper(gym_env, grounded_operator=grounded_op, executed_operators={}, config=config, domain=domain, rl_algo=args.algo, curr_subgoal=curr_subgoal)
        wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
    else:
        wrapped_env = MinimalWrapper(gym_env, config, domain)
    env = Monitor(wrapped_env, filename=f'ppo_{args.domain}_approach_visualize_monitor', allow_early_resets=True)


    model = PPO.load(f'./PPO_approach_cleanup_eval_freq_2000_lr_schedule_True/best_model/best_model.zip', env=env)

    obs, info = env.reset()
    for _ in range(config['learning']['eval']['n_eval_episodes']):
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done:
                obs, info = env.reset()
                break
        obs, info = env.reset()
    env.close()