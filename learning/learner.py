import os
import stable_baselines3
import execution
import execution.executor
from tarski import fstrips as fs

class Learner:
    def __init__(self, env, domain:str, grounded_operator_to_learn:fs.Action):
        self.env = env
        self.domain = domain
        self.grounded_operator = grounded_operator_to_learn
        
    def learn(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the operator.

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator and executes the policy for the operator when called
        """
        #TODO: implement the training loop
        pass
    
