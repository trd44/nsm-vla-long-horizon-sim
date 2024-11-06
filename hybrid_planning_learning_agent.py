import os
import importlib
import execution.executor
import planning
import execution
import learning
import mimicgen
import planning.hybrid_symbolic_llm_planner
import planning.planning_utils
from utils import *
from robosuite.controllers import load_controller_config
from tarski import fstrips as fs

class HybridPlanningLearningAgent:
    def __init__(self, config_file='config.yaml'):
        self.config:dict = load_config(config_file)
        self.domain:str = self.config['domain'].split('_domain.')[0]
        self.planner:planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner = planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner(config_file)
        self.env = self._load_env()
        self.detector = self._load_detector()
    
    def plan(self) -> List[List[fs.Action]]:
        """generate a plan to achieve the goal based on the domain and problem files whose paths are specified in the config file

        Returns:
            List[fs.Action]: a list a sequence of actions to achieve the goal a.k.a the plans
        """
        _, _, plans = self.planner.search()
        # for this project, we just care about the first plan
        if len(plans) > 0:
            return plans[0]
        raise Exception("No plan found")
    
    def execute_operator(self, operator:fs.Action):
        """execute the operator in the simulation environment

        Args:
            operator (fs.Action): the operator to execute
        """
        executor = execution.executor.ExecutorRL(operator, "sac")
        executor.execute(self.detector)
        
    def learn_operator(self, operator:fs.Action):
        """Train an RL agent to learn the operator.

        Args:
            operator (fs.Action): the operator to learn
        """
        learner = learning.Learner(self.env, self.domain, operator)
        return learner.learn()

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
    

    