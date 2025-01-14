import robosuite as suite
from robosuite.wrappers import GymWrapper
import gymnasium as gym
from utils import *

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

config:dict = load_config("config.yaml")
domain:str = config['planning']['domain']
robosuite_env = load_env(domain, config['eval_simulation'])
detector = load_detector(config, robosuite_env)

gym_env = GymWrapper(robosuite_env)
env = Monitor(gym_env)

model = SAC.load("sac_test", env=env)

obs, _ = env.reset()
while True:
    # print(obs)
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        obs, info = env.reset()