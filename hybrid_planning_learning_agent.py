import os
import importlib
import planning
import detection
import execution
import learning
import planning.hybrid_symbolic_llm_planner
from utils import *
from robosuite.controllers import load_controller_config

import mimicgen

class HybridPlanningLearningAgent:
    def __init__(self, config_file='config.yaml'):
        self.config:dict = load_config(config_file)
        self.domain:str = self.config['domain'].split('_domain.')[0]
        self.planner = planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner(config_file)
        self.env = self._load_env()
        self.detector = self._load_detector()
    
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
    print(agent.detector)

    