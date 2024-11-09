import copy
import os
import dill
import importlib
import execution.executor
import learning.learner
import planning.hybrid_symbolic_llm_planner
import planning.planning_utils
from utils import *
from robosuite.controllers import load_controller_config
from tarski import fstrips as fs

class HybridPlanningLearningAgent:
    def __init__(self, config_file='config.yaml'):
        self.config:dict = load_config(config_file)
        self.domain:str = self.config['domain'].split('_domain.')[0]
        self.planner:planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner = planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner(self.config['planning'])
        self.env = self._load_env()
        self.detector = self._load_detector()
    
    def plan_learn_execute(self):
        """generate a plan to achieve the goal based on the domain and problem files whose paths are specified in the config file, learn a policy for each of the newly defined operators and execute each operator in the plan
        """
        iteration = 0
        goal_achieved = False
        while iteration < self.config['plan_learn_execute']['max_iter'] and not goal_achieved:
            self.env.reset()
            plan:List[fs.Action] = self.plan()
            executed_operators:Dict[fs.Action:execution.executor.Executor] = {}
            while len(plan) > 0:
                grounded_operator = plan.pop(0)
                executor_exists, execution_successful, executor = self.execute_operator(grounded_operator)
                if not executor_exists:
                    self.learn_operator(grounded_operator, executed_operators)
                    # if not learning_successful:
                    #     raise Exception(f"Learning of operator {grounded_operator} failed")
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
        plan = self._load_plan()
        if plan is not None:
            return plan
        
        # the plan has not been generated yet, so generate it
        _, _, plans = self.planner.search()
        # for this project, we just care about the first plan
        if len(plans) > 0:
            return plans[0]
        raise Exception("No plan found")
    
    def execute_operator(self, grounded_operator:fs.Action) -> Tuple[bool, bool, execution.executor.Executor]:
        """execute the operator in the simulation environment

        Args:
            grounded_operator (fs.Action): the operator to execute
        Returns:
            Tuple[bool, bool, execution.executor.Executor]: a tuple containing a boolean indicating whether the operator has an executor, a boolean indicating whether the operator was executed successfully, and the executor object of the operator.
        """
        executor_module = importlib.import_module(self.config['execution_dir']+'.'+self.domain+'.'+self.domain+'_detector')

        EXECUTORS = getattr(executor_module, self.domain.upper()+'_EXECUTORS')
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        # unpickle the .pkl files in the domain executor directory which is where the learned executors are stored
        learned_executors = {}
        for file in os.listdir(self.config['execution_dir']+'/'+self.domain):
            if file.endswith(".pkl"):
                with open(file, 'rb') as f:
                    learned_executor:execution.executor.Executor = dill.load(f)
                    learned_executors[learned_executor.name] = learned_executor
        
        # check if the operator has an executor
        if grounded_operator_name in EXECUTORS: # operator has an executor
            executor:execution.executor.Executor = EXECUTORS[grounded_operator_name]
        elif grounded_operator_name in learned_executors: # operator has a learned executor
            executor:execution.executor.Executor = learned_executors[grounded_operator_name]
        else: # operator does not have an executor
            return False, False # no executor, not executed successfully
        execution_successful = executor.execute(self.detector, grounded_operator)
        return True, execution_successful
        
    def learn_operator(self, grounded_operator:fs.Action, executed_operators:List[fs.Action]=[]):
        """Train an RL agent to learn the operator.

        Args:
            grounded_operator (fs.Action): the grounded operator to learn such as open-drawer(drawer1)
            executed_operators (List[fs.Action], optional): a list of operators that have been executed before the grounded operator. Defaults to [].
        """
        learner = learning.learner.Learner(copy.deepcopy(self.env), self.domain, self.detector, grounded_operator, executed_operators, self.config)
        learner.learn()

    def _load_plan(self):
        """If the plan has been generated and saved, load the plan from the 
        """
        # search the `planning_dir` for the latest goal node pkl file i.e. the one with the largest number
        def find_file_with_largest_number(directory):
            largest_file = None
            largest_number = None

            for filename in os.listdir(directory):
                if self.config['planning_goal_node'] not in filename:
                    continue
                # Extract number at the end of the file name (e.g., file123)
                match = re.search(r'(\d+)(?=\.\w+$)', filename)
                if match:
                    number = int(match.group(1))
                    # Update largest file and number if this one is larger
                    if largest_number is None or number > largest_number:
                        largest_number = number
                        largest_file = filename

            return largest_file, largest_number
        
        goal_node_pkl, _ = find_file_with_largest_number(self.config['planning_dir'])
        if goal_node_pkl is None:
            return None
        goal_node:planning.planning_utils.SearchNode = planning.planning_utils.unpickle_goal_node(goal_node_pkl)
        plan:List[fs.Action] = planning.planning_utils.reverse_engineer_plan(goal_node)
        return plan
        
    
    def _load_env(self):
        """load the simulation environment based on the problem domain specified in the config file
        """
        envs = set(suite.ALL_ENVIRONMENTS)
        # keep only envs that correspond to the different reset distributions from the paper
        # only keep envs that end with "Novelty"
        envs = [x for x in envs if x[-7:] == "Novelty"]
        # find the novelty env i.e. the post-novelty env whose name contains the domain
        for env_name in envs:
            if self.domain in env_name.lower() and 'pre_novelty' not in env_name.lower():
                gym_env = suite.make(
                    env_name = env_name,
                    robots = 'Kinova3',
                    controller_configs = load_controller_config(default_controller="OSC_POSE"),
                    has_renderer=True,
                    has_offscreen_renderer=False,
                    ignore_done=True,
                    use_camera_obs=False,
                    control_freq=20,
                    reward_shaping=True,
                    hard_reset=False,
                )
                return gym_env

    
    def _load_detector(self):
        """load the detector based on the problem domain specified in the config file
        """
        detector_module = importlib.import_module(self.config['detection_dir']+'.'+self.domain+'_detector')

        detector = getattr(detector_module, self.domain.capitalize()+'_Detector')
        return detector(self.env)

if __name__ == '__main__':
    agent = HybridPlanningLearningAgent()
    

    