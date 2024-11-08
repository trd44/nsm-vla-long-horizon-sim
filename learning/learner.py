import os
import execution
import execution.executor
from tarski import fstrips as fs
from robosuite.robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

class Learner:
    def __init__(self, env, domain:str, grounded_operator_to_learn:fs.Action, config:dict):
        self.env = env
        self.domain = domain
        self.grounded_operator = grounded_operator_to_learn
        self.config = config

    def learn(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the operator.

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator and executes the policy for the operator when called
        """
        model = SAC(
            "MlpPolicy",
            env = self.env,
            tensorboard_log=f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/logs',
            **self.config
        )
        model.learn(
            total_timesteps=self.config['timesteps'])
        model.save(f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/model')
    
