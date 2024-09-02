import os
from tarski.search import GroundForwardSearchModel
from tarski.search.model import progress
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import ParsingError, FstripsReader

from utils import *

class HybridSymbolicLLMPlanner:
    def __init__(self, config_file='config.yaml'):
        self.config = load_config(config_file)
        self.reader = FstripsReader(raise_on_error=True)
        self.parse_domain()
        self.problem = self.parse_problem() 
    
    def parse_domain(self):
        """parses the initial domain file
        """
        domain_file_path = self.config['planning_dir'] + os.sep + self.config['init_planning_domain']
        self.reader.parse_domain(domain_file_path)
    
    def parse_problem(self):
        """parses the problem file
        """
        problem_file_path = self.config['planning_dir'] + os.sep + self.config['planning_problem']
        problem = self.reader.parse_instance(problem_file_path)
        return problem
    
    def forward_search(self):
        """performs forward search
        """
        model = GroundForwardSearchModel(self.problem, ground_problem_schemas_into_plain_operators(self.problem))
        print(model.applicable(model.init()))
        print(model.successors(model.init()))
    
if __name__ == '__main__':
    planner = HybridSymbolicLLMPlanner()
    planner.forward_search()