import robosuite as suite
from robosuite.wrappers import GymWrapper
import gymnasium as gym
from utils import *

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

class ReachRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, target_pos=(0, 0, 1.0)):
        super().__init__(env)
        self.target_pos = np.array(target_pos)

    def reward(self, reward):
        # "reward" is the environment's original reward
        # Now we can add or replace with a distance-based reward
        current_obs = self.env.unwrapped._get_observations()  # or however you get the arm pose
        ee_pos = current_obs["gripper1_pos"]  # for instance, if that’s how your obs is structured

        dist = np.linalg.norm(ee_pos - self.target_pos)
        reach_shaping = -dist  # negative distance, bigger reward when closer

        # Optionally, give a success bonus if under a certain threshold
        if dist < 0.02:
            reach_shaping += 2.0  # big bonus for success

        # Option 1: override the entire reward
        return reach_shaping

        # Option 2: add shaping to the original
        # return reward + reach_shaping

config:dict = load_config("config.yaml")
domain:str = config['planning']['domain']
robosuite_env = load_env(domain, config['simulation'])
detector = load_detector(config, robosuite_env)

gym_env = GymWrapper(robosuite_env)
wrapped_env = ReachRewardWrapper(gym_env)
env = Monitor(wrapped_env)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create a callback to save the model every N steps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,                   # Save every 100k steps
    save_path='./checkpoints/',         # Folder to save checkpoints
    name_prefix='ddpg_model_checkpoint' # File prefix
)

model = DDPG(
    policy="MlpPolicy",
    env=env,                # your wrapped robosuite env
    action_noise=action_noise,  # optional
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./ddpg_tensorboard/"
)

model.learn(
    total_timesteps=10000000,
    callback=checkpoint_callback
    )

model.save("ddpg_model")

