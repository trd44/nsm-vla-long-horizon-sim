import os
import logging
import typing
from collections import deque

from tarski.search import GroundForwardSearchModel
from tarski.search.model import progress
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader, FstripsWriter
from tarski.syntax import land, neg, CompoundFormula
from tarski import fstrips as fs
from tarski.evaluators.simple import evaluate

from langchain.tools import tool

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
    
    def add_operator(self, operator_name:str, parameters:List[str], precondition:List[str], effects:List[str]):
        """Adds an action to the current problem

        Args:
            operator_name (str): the name of the operator
            parameters (List[str]): a list of parameters. Example: `['?drawer - drawer', '?gripper - gripper']`
            precondition (List[str]): a list of precondition. Example: `['(not (open ?drawer))', '(free ?gripper)']`
            effects (List[str]): a list of effects. Example: `['(open ?drawer)']`

        Returns:
            str: the name of the action added
        """
        params_dict = {}

        def parse_parameters(parameters:List[str], params_dict:dict):
            """fills the params_dict with the mappings from the name of the parameter to the its variable

            Args:
                parameters (List[str]): list of parameters
                params_dict (dict): dictionary to store the mappings
            """
            for param in parameters:
                param = param.split(' - ')
                # first item is the parameter name, second item is the parameter type
                param_name, param_type_str = param[0], param[1]
                param_type = self.problem.language.get_sort(param_type_str)
                param_var = self.problem.language.variable(param_name, param_type)
                params_dict[param_name] = param_var
            
        
        def parse_predicate(pred:str) -> Tuple[str, List[str], bool]:
            """parse the predicate into (name, parameter list, negated)

            Args:
                pred (str): _description_
                List (_type_): _description_
                bool (_type_): _description_
            """
            
            parts = pred.replace('(', '').replace(')', '').split()
            # check if the first part is 'not'
            if parts[0] == 'not':
                # if it is, then the second part is the name of the predicate
                name = parts[1]
                # the rest of the parts are the arguments
                args = parts[2:]
                return name, args, True
            else:
                name = parts[0]
                args = parts[1:]
                return name, args, False

        def parse_precondition(predicates:List[str]) -> CompoundFormula:
            """parse the predicates into a CompoundFormula

            Args:
                predicates (List[str]): list of predicates
            Returns:
                CompoundFormula: the compound formula
            """
            pred_list = []
            for pred in predicates:
                name, args, negated = parse_predicate(pred)
                args_list = [params_dict[arg] for arg in args]
                pred = self.problem.language.get_predicate(name)
                if negated:
                    pred_list.append(neg(pred(*args_list)))
                else:
                    pred_list.append(pred(*args_list))
            return land(*pred_list)

        def parse_effects(predicates:List[str]) -> List[fs.SingleEffect]:
            """parse the effects into a list of SingleEffect

            Args:
                predicates (List[str]): list of predicates
            Returns:
                List[fs.SingleEffect]: list of SingleEffect
            """
            effects = []
            for pred in predicates:
                name, args, negated = parse_predicate(pred)
                pred = self.problem.language.get_predicate(name)
                if negated:
                    effects.append(fs.DelEffect(pred(*[params_dict[arg] for arg in args])))
                else:
                    effects.append(fs.AddEffect(pred(*[params_dict[arg] for arg in args])))
            return effects
        

        parse_parameters(parameters, params_dict)
        params_list = list(params_dict.values())
        precondition_formula:CompoundFormula = parse_precondition(precondition)
        effects_list:List[fs.SingleEffect] = parse_effects(effects)
        
        self.problem.action(
            name=operator_name,
            parameters=params_list,
            precondition=precondition_formula,
            effects=effects_list
        )
        return operator_name

    def forward_search(self):
        """performs forward search
        """
        # drawer = self.problem.language.get_sort('drawer')
        
        # drawer_var = self.problem.language.variable('?drawer', drawer)
        # gripper = self.problem.language.get_sort('gripper')
        # gripper_var = self.problem.language.variable('?gripper', gripper)
        # open = self.problem.language.get_predicate('open')
        # free = self.problem.language.get_predicate('free')
        # self.problem.action(
        #     name='open-drawer',
        #     parameters=[drawer_var, gripper_var],
        #     precondition=land(neg(open(drawer_var)), free(gripper_var)),
        #     effects=[fs.AddEffect(open(drawer_var))]
        # )
        model = GroundForwardSearchModel(self.problem, ground_problem_schemas_into_plain_operators(self.problem))
        print(list(model.applicable(model.init())))
        print(list(model.successors(model.init())))
        search = BreadthFirstSearch(model, max_expansions=self.config['max_depth'])
        space, stats, plans = search.run()
        domain_file_name = "temp.pddl"
        writer = FstripsWriter(self.problem)
        writer.write_domain(domain_file_name, constant_objects=None)
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
                logging.info(f"Max. expansions reached on one branch. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                stats.nexpansions -= 1
                continue
            else:
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
            # open, drawer1 = self.model.problem.language.get('open', 'drawer1')
            # under, mug1, coffee_pod_holder1 = self.model.problem.language.get('under', 'mug1', 'coffee-pod-holder1')
            # inside, pod1 = self.model.problem.language.get('in', 'coffee-pod1')
            # if evaluate(inside(pod1, coffee_pod_holder1), node.state):
            #     print('pod is inside coffee-pod-holder1')
            # if evaluate(under(mug1, coffee_pod_holder1), node.state):
            #     print('mug1 is under coffee-pod-holder1')
            if self.model.is_goal(node.state):
                stats.num_goals += 1
                plan = reverse_engineer_plan(node)
                plans.append(plan)
                logging.info(f"Goal found after {stats.nexpansions} expansions. {stats.num_goals} goal states found.")

            if 0 <= self.max_expansions <= stats.nexpansions:
                logging.info(f"Max. expansions reached. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                return space, stats, plans

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
    planner.add_operator(
        operator_name='open-drawer',
        parameters=['?drawer - drawer', '?gripper - gripper'],
        precondition=['(not (open ?drawer))', '(free ?gripper)'],
        effects=['(open ?drawer)']
    )
    planner.forward_search()