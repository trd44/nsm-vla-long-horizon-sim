import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
from tarski import fstrips as fs
from robosuite.robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from typing import *

class OperatorWrapper(gym.Wrapper):
    def __init__(self, env, detector:detection.detector.Detector, grounded_operator:fs.Action, executed_operators:Dict[fs.Action:execution.executor.Executor], config:dict):
        super().__init__(env)
        self.detector = detector
        self.grounded_operator = grounded_operator
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = executed_operators
        self.config = config

    def step(self, action):
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        self.detector.update_obs()
        obs:dict = self.detector.get_obs()
        binary_states_with_semantics:dict = self.detector.get_groundings()
        # combine obs with binary_states_with_semantics
        obs.update(binary_states_with_semantics)
        reward = self.compute_reward(obs)
        #TODO: may need to turn obs into a numpy array
        return obs, reward, done, truncated, info

    def reset(self):
        reset_success = False
        while not reset_success:
            # first, reset the environment to the very beginning
            try:
                obs, info = self.env.reset(seed=self.config['learning']['seed'])
            except:
                obs = self.env.reset(seed=self.config['learning']['seed'])
                info = {}
            # second, execute the executors that should be executed before the operator to learn
            reset_success = True
            for op, ex in self.executed_operators.items():
                ex_success = ex.execute(self.detector, op)
                if not ex_success:
                    reset_success = False
                    break

        self.detector.update_obs()
        obs = self.detector.get_obs()
        return obs, info


    def compute_reward(self, obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            obs (dict): the observation with semantics

        Returns:
            float: the reward
        """
        llm_reward_func_module = importlib.import_module(f'learning.reward_functions.{self.config['planning']['domain']}.{self.grounded_operator.name}')

        llm_reward_func = getattr(llm_reward_func_module, 'reward')
        reward = llm_reward_func(obs)
        return reward


class Learner:
    def __init__(self, env, domain:str, detector:detection.detector.Detector, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action:execution.executor.Executor], config:dict):
        self.config = config
        self.detector = detector
        self.domain = domain
        self.env = self._wrap_env(env)
        self.executed_operators = executed_operators
        self.grounded_operator = grounded_operator_to_learn

    def learn(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the operator.

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator and executes the policy for the operator when called
        """
        # TODO: add a customeval callback and pass it into `model.save`
        model = SAC(
            "MlpPolicy",
            env = self.env,
            tensorboard_log=f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/tensorboard_logs',
            **self.config['learning']
        )
        model.learn(
            total_timesteps=self.config['timesteps'])
        model.save(f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/model')
        # TODO: create an Executor_RL object associated with the newly learned policy. Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
    

    def _wrap_env(self, env) -> gym.Wrapper:
        """Wrap the environment in a GymWrapper.

        Args:
            env (gym environment): the environment to wrap

        Returns:
            gym.Wrapper: the wrapped environment
        """
        env = GymWrapper(env)
        env = OperatorWrapper(env, self.detector, self.grounded_operator, self.executed_operators, self.config)
        env = Monitor(env, f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/monitor_logs', allow_early_resets=True)
        return env
    
