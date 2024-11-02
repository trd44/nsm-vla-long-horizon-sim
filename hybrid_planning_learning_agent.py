import os
import importlib
import planning
import detection
import execution
import learning
import planning.hybrid_symbolic_llm_planner
from utils import *

class HybridPlanningLearningAgent:
    def __init__(self, config_file='config.yaml', env=None):
        self.config:dict = load_config(config_file)
        self.domain:str = self.config['domain'].split('_domain.')[0]
        self.planner = planning.hybrid_symbolic_llm_planner.HybridSymbolicLLMPlanner(config_file)
        self.env = env
        self.detector = self._load_detector()
    
    def _load_detector(self):
        """load the detector based on the problem domain specified in the config file
        """
        detector_module = importlib.import_module(self.config['detection_dir']+'.'+self.domain+'_detector')
        detector = getattr(detector_module, self.domain.capitalize()+'_Detector')
        return detector(self.env)

if __name__ == '__main__':
    agent = HybridPlanningLearningAgent()
    print(agent.detector)

    