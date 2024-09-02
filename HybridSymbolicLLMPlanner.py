import os
import logging
import typing
from collections import deque

from tarski.search import GroundForwardSearchModel
from tarski.search.model import progress
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader

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
        search = DepthFirstSearch(model, max_expansions=self.config['max_depth'])
        space, stats, plans = search.run()
        return space, stats, plans



class DepthFirstSearch:
    """Full expansion of a problem through Depth-First search. Also returns the plan(s)"""
    def __init__(self, model: GroundForwardSearchModel, max_expansions=-1):
        self.model = model
        self.max_expansions = max_expansions
    
    def run(self):
        return self.search(self.model.init())
    
    def search(self, root):
        # create obj to track state space
        
        space = SearchSpace()
        stats = SearchStats()
        plans = []

        openlist = deque()  # stack storing the nodes which are next to explore
        openlist.append(make_root_node(root))
        closed = {root}

        while openlist:
            stats.iterations += 1
            # logging.debug("dfs: Iteration {}, #unexplored={}".format(iteration, len(open_)))

            node = openlist.pop()
            if self.model.is_goal(node.state):
                stats.num_goals += 1
                plan = reverse_engineer_plan(node)
                plans.append(plan)
                logging.info(f"Goal found after {stats.nexpansions} expansions. {stats.num_goals} goal states found.")

            if 0 <= self.max_expansions <= stats.nexpansions:
                logging.info(f"Max. expansions reached. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                return space, stats

            for operator, successor_state in self.model.successors(node.state):
                if successor_state not in closed:
                    openlist.append(make_child_node(node, operator, successor_state))
                    closed.add(successor_state)
            stats.nexpansions += 1

        logging.info(f"Search space exhausted. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
        space.complete = True
        return space, stats, plans


class BreadthFirstSearch:
    """ Full expansion of a problem through Breadth-First search.
    """
    def __init__(self, model: GroundForwardSearchModel, max_expansions=-1):
        self.model = model
        self.max_expansions = max_expansions

    def run(self):
        return self.search(self.model.init())

    def search(self, root):
        # create obj to track state space
        
        space = SearchSpace()
        stats = SearchStats()
        plans = []

        openlist = deque()  # fifo-queue storing the nodes which are next to explore
        openlist.append(make_root_node(root))
        closed = {root}

        while openlist:
            stats.iterations += 1
            # logging.debug("brfs: Iteration {}, #unexplored={}".format(iteration, len(open_)))

            node = openlist.popleft()
            if self.model.is_goal(node.state):
                stats.num_goals += 1
                plan = reverse_engineer_plan(node)
                plans.append(plan)
                logging.info(f"Goal found after {stats.nexpansions} expansions. {stats.num_goals} goal states found.")

            if 0 <= self.max_expansions <= stats.nexpansions:
                logging.info(f"Max. expansions reached. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                return space, stats

            for operator, successor_state in self.model.successors(node.state):
                if successor_state not in closed:
                    openlist.append(make_child_node(node, operator, successor_state))
                    closed.add(successor_state)
            stats.nexpansions += 1

        logging.info(f"Search space exhausted. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
        space.complete = True
        return space, stats, plans


class SearchNode:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class SearchSpace:
    """ A representation of a search space / transition system corresponding to some planning problem """
    def __init__(self):
        self.nodes = set()
        self.last_node_id = 0
        self.complete = False  # Whether the state space contains all states reachable from the initial state
    #
    # def expand(self, node: SearchNode):
    #     self.nodes.add(node)


class SearchStats:
    def __init__(self):
        self.iterations = 0
        self.num_goals = 0
        self.nexpansions = 0


def make_root_node(state):
    """ Construct the initial root node without parent nor action """
    return SearchNode(state, None, None)


def make_child_node(parent_node, action, state):
    """ Construct an child search node """
    return SearchNode(state, parent_node, action)

def reverse_engineer_plan(node:SearchNode) -> list:
            """Reverse engineer the plan from the goal node back to the root node

            Args:
                node (SearchNode): the goal node with a parent node

            Returns:
                list: the plan from the root node to the goal node
            """
            plan = []
            while node.parent is not None:
                plan.append(node.action)
                node = node.parent
            plan.reverse()
            return plan

    
if __name__ == '__main__':
    planner = HybridSymbolicLLMPlanner()
    planner.forward_search()