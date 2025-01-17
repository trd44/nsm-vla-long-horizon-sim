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
from debug_training import MinimalWrapper


if __name__ == '__main__':
    config:dict = load_config("config.yaml")
    domain:str = 'cleanup'
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    gym_env = GymWrapper(robosuite_env)
    wrapped_env = MinimalWrapper(gym_env, config, domain)
    env = Monitor(wrapped_env, filename=f'ppo_{domain}_approach_visualize_monitor', allow_early_resets=True)


    model = PPO.load(f"ppo_approach_{domain}", env=env)

    obs, info = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()