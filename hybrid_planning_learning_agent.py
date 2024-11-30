import os
import dill
import importlib
import learning.learner
import planning.hybrid_symbolic_llm_planner
import planning.planning_utils
from utils import *
from tarski import fstrips as fs
from execution.executor import Executor

class HybridPlanningLearningAgent:
    def __init__(self, config_file='config.yaml'):
        self.config:dict = load_config(config_file)
        self.domain:str = self.config['planning']['domain']
        self.planner:planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner = planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner(self.config)
        self.env = load_env(self.domain, self.config['simulation'])
        self.detector = load_detector(self.config, self.env)
    
    def plan_learn_execute(self):
        """generate a plan to achieve the goal based on the domain and problem files whose paths are specified in the config file, learn a policy for each of the newly defined operators and execute each operator in the plan
        """
        iteration = 0
        goal_achieved = False
        while iteration < self.config['plan_learn_execute']['max_iter'] and not goal_achieved:
            self.env.reset()
            plan:List[fs.Action] = self.plan()
            executed_operators:Dict[fs.Action:Executor] = OrderedDict()
            while len(plan) > 0: # execute each operator in the plan
                grounded_operator = plan.pop(0)
                executor_exists, execution_successful, executor = self.execute_operator(grounded_operator)
                if not executor_exists: # learn the operator if it does not have an executor
                    self.learn_operator(grounded_operator, executed_operators)
                    _, execution_successful, executor = self.execute_operator(grounded_operator)
                if not execution_successful: # restart the `plan_learn_execute` loop
                    break
                executed_operators[grounded_operator] = executor
            iteration += 1
            goal_achieved = len(plan) == 0 and execution_successful

    
    def plan(self) -> List[List[fs.Action]]:
        """generate a plan to achieve the goal based on the domain and problem files whose paths are specified in the config file

        Returns:
            List[fs.Action]: a list a sequence of actions to achieve the goal a.k.a the plans
        """
        # try loading the plan from the planning directory in case it has already been generated
        plan = load_plan(config=self.config)
        if plan is not None:
            return plan
        
        # the plan has not been generated yet, so generate it
        _, _, plans = self.planner.search()
        # for this project, we just care about the first plan
        if len(plans) > 0:
            return plans[0]
        raise Exception("No plan found")
    
    def execute_operator(self, grounded_operator:fs.Action) -> Tuple[bool, bool, Executor]:
        """execute the operator in the simulation environment

        Args:
            grounded_operator (fs.Action): the operator to execute
        Returns:
            Tuple[bool, bool, execution.executor.Executor]: a tuple containing a boolean indicating whether the operator has an executor, a boolean indicating whether the operator was executed successfully, and the executor object of the operator.
        """
        executor = load_executor(self.config, grounded_operator=grounded_operator)
        if executor is None:
            return False, False, None # no executor found, not executed successfully, no executor object
        execution_successful = executor.execute(self.detector, grounded_operator)
        return True, execution_successful, executor
        
    def learn_operator(self, grounded_operator:fs.Action, executed_operators:List[fs.Action]=[]):
        """Train an RL agent to learn the operator.

        Args:
            grounded_operator (fs.Action): the grounded operator to learn such as open-drawer(drawer1)
            executed_operators (List[fs.Action], optional): a list of operators that have been executed before the grounded operator. Defaults to [].
        """
        # deep copy env and detector to avoid modifying the original env and detector
        env_copy = deepcopy_env(self.env, self.config['simulation'])
        learner = learning.learner.Learner(env_copy, self.domain, grounded_operator, executed_operators, self.config)
        learner.learn()


if __name__ == '__main__':
    agent = HybridPlanningLearningAgent()
    agent.plan_learn_execute()
    

    