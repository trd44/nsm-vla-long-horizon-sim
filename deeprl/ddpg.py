import os
import csv
import gymnasium as gym
import numpy as np

from robosuite.wrappers import GymWrapper
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# -----------------
# Example OperatorWrapper
# -----------------
class OperatorWrapper(gym.Wrapper):
    """
    Minimal example wrapper that just demonstrates:
        - Step method
        - Reset method
        - Possibly a custom reward
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Example: track your own episode reward, etc.
        self.episode_reward = 0
        # If you want to record steps to CSV
        self.rollout_save_path = "rollout.csv"
        self.csv_file = open(self.rollout_save_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'reward'])

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.episode_reward += reward
        
        # [Optional] Custom shaping of reward. For example:
        # reward += <some extra shaping> 
        # done = <some custom done condition>

        # Write to CSV if desired
        self.csv_writer.writerow([self.env.time, reward])

        if done or truncated:
            # Reset episode reward
            self.episode_reward = 0

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.csv_file.close()
        return super().close()


# -----------------
# Example usage: training a DDPG policy
# -----------------

def make_env():
    """
    Minimal function that returns a wrapped environment.
    Replace with your actual robot or Mujoco environment creation if needed.
    """
    # For demonstration, we'll just use a standard Gym environment
    env = gym.make("Pendulum-v1")
    # If you have a custom Mujoco environment, do something like:
    # env = SomeMujocoEnv()
    # env = GymWrapper(env)  # if using robosuite
    env = OperatorWrapper(env)
    env = Monitor(env)  # stable-baselines Monitor
    return env


def main():
    # Create training and evaluation environments
    train_env = make_env()
    eval_env = make_env()

    # Callback for evaluation every 10k steps, for instance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ddpg_best_model",
        log_path="./ddpg_logs",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Create the DDPG model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./ddpg_tensorboard/",
        # You can tune any hyperparameters you like here
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=100000,
        learning_starts=5000,
        gamma=0.99,
        tau=0.005
    )

    # Train the model
    model.learn(
        total_timesteps=100000,  # set this to however long you want to train
        callback=eval_callback
    )

    # Save the model
    model.save("ddpg_final_model")

    # (Later) load the model
    loaded_model = DDPG.load("ddpg_final_model", env=train_env)

    # Use the loaded model for further training or evaluation
    obs, info = train_env.reset()
    for _ in range(1000):
        action, _states = loaded_model.predict(obs)
        obs, reward, done, truncated, info = train_env.step(action)
        if done or truncated:
            obs, info = train_env.reset()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
