from typing import *

class OperatorCandidate:
    def __init__(self, name:str, parameters:List[str]=None, precondition:List[str]=None, effects:List[str]=None, grounded_params:List[str]=None):
        self.name = name
        if parameters:
            self.parameters = sorted(parameters)
        else:
            self.parameters = []
        if precondition:
            self.precondition = sorted(precondition)
        else:
            self.precondition = []
        if effects:
            self.effects = sorted(effects)
        else:
            self.effects = []
        if grounded_params:
            self.grounded_params = grounded_params
        else:
            self.grounded_params = []
    
    def is_empty(self):
        return len(self.parameters) == 0
    
    def set_parameters(self, parameters:List[str]):
        self.parameters = sorted(parameters)
    
    def set_precondition(self, precondition:List[str]):
        self.precondition = sorted(precondition)
    
    def set_effects(self, effects:List[str]):
        self.effects = sorted(effects)
    
    def set_grounded_params(self, grounded_params:List[str]):
        self.grounded_params = grounded_params
    
    def name_param_repr(self):
        return f"(:action {self.name}\n\t:parameters ({' '.join(self.parameters)})\n)"
    
    def name_param_precond_repr(self):
        if not self.name:
            return ''
        return f"(:action {self.name}\n\t:parameters ({' '.join(self.parameters)})\n\t:precondition (and {' '.join(self.precondition)})\n)"
    
    def full_repr(self):
        return str(self)
    
    def __str__(self):
        return f"(:action {self.name}\n\t:parameters ({' '.join(self.parameters)})\n\t:precondition (and {' '.join(self.precondition)})\n\t:effect (and {' '.join(self.effects)})\n)"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # two operators are equal if their parameters, precondition, and effects are the same
        return self.parameters == other.parameters and self.precondition == other.precondition and self.effects == other.effects

    def __hash__(self):
        # hash the tuple of the parameters, precondition, and effects
        return hash((tuple(self.parameters), tuple(self.precondition), tuple(self.effects)))

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __le__(self, other):
        return str(self) <= str(other)

    def __ge__(self, other):
        return str(self) >= str(other)

    def __ne__(self, other):
        return str(self) != str(other)

class OperatorCandidateCounter:
    def __init__(self):
        self.operator_candidates = {}
    
    def add_operator_candidate(self, operator_candidate:OperatorCandidate):
        if operator_candidate in self.operator_candidates:
            self.operator_candidates[operator_candidate] += 1
        else:
            self.operator_candidates[operator_candidate] = 1
    
    def get_max_operator_candidate(self) -> OperatorCandidate:
        return max(self.operator_candidates, key=self.operator_candidates.get)

class SearchNode:
    def __init__(self, state, parent, action, depth=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0 if parent is None else parent.depth + 1
        if depth:
            self.depth = depth
    
    # add support for comparator
    def __lt__(self, other):
        if self.depth == other.depth:
            # uses the state as the tiebreaker
            # turn state into string and compare
            return str(self.state) < str(other.state)
        return self.depth < other.depth
    
    def __gt__(self, other):
        if self.depth == other.depth:
            # uses the state as the tiebreaker
            # turn state into string and compare
            return str(self.state) > str(other.state)
        return self.depth > other.depth

        


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
    return SearchNode(state, None, None, 0)


def make_child_node(parent_node, action, state):
    """ Construct an child search node """
    return SearchNode(state, parent_node, action, parent_node.depth + 1)

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