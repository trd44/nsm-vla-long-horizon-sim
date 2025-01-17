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
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from utils import *

# set up a logger here to log the terminal printouts for the training of each subgoal
logger = logging.getLogger('learning')
logger.setLevel(logging.INFO)


class DiscreteGripperWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    
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

class GraspRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, config, domain):
        super().__init__(env)
        self.env = env
        self.config = config
        self.domain = domain
        self.detector = load_detector(config=config, domain=domain, env=env.unwrapped)


    
    def reward(self, reward):
        obs_from_detector = self.detector.get_obs()
        obs_from_unwrapped = self.env.unwrapped._get_observations()
        binary_obs = self.detector.detect_binary_states(self.env.unwrapped)
        dist = np.linalg.norm(obs_from_unwrapped['gripper1_pos'] - obs_from_detector['mug1_pos'])
        dist2 = np.linalg.norm(obs_from_detector['gripper1_pos'] - obs_from_detector['mug1_pos'])
        normalized = dist / np.linalg.norm(obs_from_unwrapped['gripper1_to_obj_max_possible_dist'])
        grasp = binary_obs['exclusively-occupying-gripper mug1 gripper1']
        if grasp:
            reward = 1
        else:
            reward = 1 - normalized

        step_cost = - 1
        return reward + step_cost
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(reward)
        binary_obs = self.detector.detect_binary_states(self.env.unwrapped)
        grasp = binary_obs['exclusively-occupying-gripper mug1 gripper1']
        done = done or grasp # end early if the gripper has grasped the mug
        return obs, reward, done, info
     


if __name__ == '__main__':
    config:dict = load_config("config.yaml")
    domain:str = 'cleanup'
    robosuite_env = load_env(domain, config['simulation'])
    gym_env = GymWrapper(robosuite_env)
    action_wrapped_env = DiscreteGripperWrapper(gym_env)
    rw_wrapped_env = GraspRewardWrapper(action_wrapped_env, config, domain)
    env = Monitor(rw_wrapped_env, filename=f'ppo_rw_{domain}_approach_monitor', allow_early_resets=True)

    eval_robosuite_env = load_env(domain, config['simulation'])
    eval_gym_env = GymWrapper(eval_robosuite_env)
    eval_action_wrapped_env = DiscreteGripperWrapper(eval_gym_env)
    eval_rw_wrapped_env = GraspRewardWrapper(eval_action_wrapped_env, config, domain)
    eval_env = Monitor(eval_rw_wrapped_env, filename=f'ppo_rw_{domain}_approach_eval_monitor', allow_early_resets=True)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)


    model = PPO("MlpPolicy", env, seed=0, tensorboard_log=f"./ppo_rw_approach_{domain}_tensorboard/")

    model.learn(total_timesteps=100000, callback=EvalCallback(eval_env=eval_env, best_model_save_path=f"./ppo_rw_approach_{domain}_best_model/", log_path=f"./ppo_rw_approach_{domain}_logs/", eval_freq=1000, deterministic=True, render=False, n_eval_episodes=2, verbose=1))

    model.save(f"ppo_rw_approach_{domain}")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()
    env.close()