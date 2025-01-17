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
    domain:str = 'cleanup'
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    gym_env = GymWrapper(robosuite_env)
    wrapped_env = MinimalWrapper(gym_env, config, domain)
    env = Monitor(wrapped_env, filename=f'ppo_{domain}_approach_monitor', allow_early_resets=True)

    eval_robosuite_env = load_env(domain, config['simulation'])
    eval_gym_env = GymWrapper(eval_robosuite_env)
    eval_wrapped_env = MinimalWrapper(eval_gym_env, config, domain)
    eval_env = Monitor(eval_wrapped_env, filename=f'ppo_{domain}_approach_eval_monitor', allow_early_resets=True)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # load the model if it exists
    if os.path.exists(f"ppo_approach_{domain}.zip"):
        model = PPO.load(f"ppo_approach_{domain}", env)
    else:
        model = PPO("MlpPolicy", env, seed=0, tensorboard_log=f"./ppo_approach_{domain}_tensorboard/")

    model.learn(total_timesteps=1_000_000, callback=EvalCallback(eval_env=eval_env, best_model_save_path=f"./ppo_rw_approach_{domain}_best_model/", log_path=f"./ppo_approach_{domain}_logs/", eval_freq=1000, deterministic=True, render=False, n_eval_episodes=2, verbose=1))

    model.save(f"ppo_approach_{domain}")
