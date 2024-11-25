import os
import logging
import heapq
import copy
import itertools
import dill
from collections import deque

from planning.planning_utils import OperatorCandidate, OperatorCandidateCounter, make_root_node, make_child_node, reverse_engineer_plan, SearchSpace, SearchStats
from tarski.search import GroundForwardSearchModel
from tarski.model import Model
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader, FstripsWriter
from tarski.syntax import land, neg, CompoundFormula, Sort, Constant, Atom
from tarski.syntax.formulas import VariableBinding
from tarski.syntax.builtins import *
from tarski import fstrips as fs
from tarski.evaluators.simple import evaluate
from tarski.model import Model
from tarski.syntax.builtins import BuiltinPredicateSymbol
from VLM.LlmApi import *
from VLM.TreeOfThoughtsPrompts import *
from utils import *


class HybridSymbolicLLMPlanner:
    def __init__(self, config:dict):
        self.config = config
        self.reader = FstripsReader(raise_on_error=True)
        self.parse_domain()
        self.starting_problem = self.parse_problem()
        self.max_new_operators_branching_factor = self.config['planning']['max_new_operators_branching_factor']
        self.llm_calls = 0
        self.max_depth = self.config['planning']['max_depth']
        self.max_num_llm_calls = self.config['planning']['max_num_llm_calls']
        self.novel_objects = self.config['planning']['novel_objects']
    
    def parse_domain(self):
        """parses the initial domain file
        """
        domain_file_path = self.config['planning']['planning_dir'] + os.sep + self.config['planning']['init_planning_domain']
        self.reader.parse_domain(domain_file_path)
    
    def parse_problem(self):
        """parses the problem file
        """
        problem_file_path = self.config['planning']['planning_dir'] + os.sep + self.config['planning']['planning_problem']
        problem = self.reader.parse_instance(problem_file_path)
        return problem
    
    def parse_operator(self, operator:str) -> OperatorCandidate:
        """parse the operators string into a dictionary

        Args:
            operator (str): the operators string with operator separated by newlines

        Returns:
            the parsed OperatorCandidate
        """
        operator = operator.split('(:action')
        operators_dict = {}
        if len(operator) == 0:
            return operators_dict
        operator = operator[-1]
        # the string between `:action` and :parameters is the operator name
        operator_name = operator.split(':parameter')[0].strip().replace('\n', '')
        # the string between `:parameters` and :precondition is the parameters
        param_start = operator.find(':parameter')
        param_end = operator.find(':precondition')
        p = find_parentheses(operator[param_start:param_end])
        parameters = operator[param_start+p[0]:param_start+p[1]]
        # split the parameters into a list of parameters at the space before the `?` but keep the `?`
        parameters = ['?' + param.strip() for param in parameters.split('?') if param]
        # the string between `:precondition` and :effects is the precondition
        precondition_start = operator.find(':precondition')
        precondition_end = operator.find(':effects')
        p = find_parentheses(operator[precondition_start:precondition_end])
        precondition = operator[precondition_start+p[0]:precondition_start+p[1]]
        # split the precondition into a list of predicates by matching parentheses
        precondition = [f'({condition})' for condition in split_by_parentheses(precondition)]
        # the string between `:effects` and the end of the operator is the effects
        effects_start = operator.find(':effect')
        effects_end = operator.find('\n)', effects_start)
        p = find_parentheses(operator[effects_start:effects_end])
        effects = operator[effects_start+p[0]:effects_start+p[1]]
        # split the effects into a list of predicates by matching parentheses
        effects = [f'({condition})' for condition in split_by_parentheses(effects)]
        return OperatorCandidate(operator_name, parameters, precondition, effects)


    def add_operator(self, problem:fs.problem.Problem, operator:OperatorCandidate) -> str:
        """Adds an action to the current problem

        Args:
            problem (fs.problem.Problem): the current problem
            operator (OperatorCandidate): the operator to add

        Returns:
            str: the name of the action added
        """
        # check if the operator is empty
        if operator.is_empty():
            return ''
        
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
                param_type = problem.language.get_sort(param_type_str)
                param_var = problem.language.variable(param_name, param_type)
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
                pred = problem.language.get_predicate(name)
                if negated:
                    pred_list.append(neg(pred(*args_list)))
                else:
                    pred_list.append(pred(*args_list))
            return land(*pred_list, flat=True)

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
                pred = problem.language.get_predicate(name)
                if negated:
                    effects.append(fs.DelEffect(pred(*[params_dict[arg] for arg in args])))
                else:
                    effects.append(fs.AddEffect(pred(*[params_dict[arg] for arg in args])))
            return effects
        

        parse_parameters(operator.parameters, params_dict)
        params_varbinding = VariableBinding(list(params_dict.values()))
        precondition_formula:CompoundFormula = parse_precondition(operator.precondition)
        effects_list:List[fs.SingleEffect] = parse_effects(operator.effects)
        
        problem.action(
            name=operator.name,
            parameters=params_varbinding,
            precondition=precondition_formula,
            effects=effects_list
        )
        return operator.name
    
    def write_domain(self, problem:fs.problem.Problem, domain_file:str):
        """writes the domain to a file

        Args:
            problem (fs.problem.Problem): the problem to write
            domain_file (str): the file to write the domain to
        """
        writer = FstripsWriter(problem)
        writer.write_domain(domain_file, constant_objects=None)

    def prompt_llm_for_new_operator(self, problem:fs.problem.Problem, state:Model) -> OperatorCandidate:
        """prompts the LLM for new operators

        Args:
            problem (fs.problem.Problem): the current problem
            state (fs.State): the current state

        Returns:
            OperatorCandidate: the majority operator candidate
        """
        lang_dump = problem.language.dump()
        current_state = ", ".join(sorted(map(str, state.as_atoms())))
        if hasattr(problem.goal, 'subformulas'):
            g_str = ", ".join(g.replace('(not ', 'not(') for g in sorted(map(str, problem.goal.subformulas)))
        else:
            g_str = str(problem.goal)
        
        # find the dict in lang_dump['sorts'] that has the name 'object' and get the domain
        relevant_objects = "object"
        for s in lang_dump['sorts']:
            if s['name'] == 'object':
                relevant_objects = ', '.join(s['domain']) # currently assumes all objects are relevant. TODO: for future versions, prompt the LLM to determine relevant objects
                break
        novel_objects = ", ".join(self.novel_objects)
        # the `(:objects...)` section of the domain text
        types_start = self.reader.domain_text.find('(:types')
        parentheses = find_parentheses(self.reader.domain_text[types_start:])
        object_types = self.reader.domain_text[types_start:types_start+parentheses[1]]

        # predicates are in the section `:predicates` of domain text
        preds_start = self.reader.domain_text.find('(:predicates')
        preds_end = self.reader.domain_text.find('(:action')
        preds_section = self.reader.domain_text[preds_start:preds_end]
        parentheses = find_parentheses(preds_section)
        preds = preds_section[parentheses[0]:parentheses[1]]

        # get existing operators
        existing_op_params = []
        existing_op_params_precond = []
        existing_op_params_precond_effect = []
        for existing_op_name, existing_op in problem.actions.items():
            op_w_params = \
            f"(:action {existing_op_name}\n\t:parameters ({existing_op.parameters.pddl_repr()})"
            op_w_param_precond = \
            f"{op_w_params}\n\t:precondition ({existing_op.precondition.pddl_repr()})"
            effects_str = ' '.join(['('+ e.pddl_repr() + ')' for e in existing_op.effects])
            op_w_param_precond_effect = \
            f"{op_w_param_precond}\n\t:effect (and {effects_str})"
            existing_op_params.append(op_w_params+'\n)')
            existing_op_params_precond.append(op_w_param_precond+'\n)')
            existing_op_params_precond_effect.append(op_w_param_precond_effect+'\n)')

        existing_op_params_precond_effect_str = '\n'.join(existing_op_params_precond_effect)
        
        true_atoms, false_atoms = self.full_state_description(state, self.novel_objects)

        def extract_operator_from_llm_output(out:str) -> str:
            """parse the `(:action)` section from llm's output

            Args:
                out (str): llm output string
            """
            proposed_operator_start:int = out.find('(:action')
            if proposed_operator_start == -1:
                return ''
            proposed_operator_parentheses:tuple = find_parentheses(out[proposed_operator_start:])
            proposed_operator_str:str = out[proposed_operator_start + proposed_operator_parentheses[0] - 1:proposed_operator_start + proposed_operator_parentheses[1]+1] # include the parantheses
            return proposed_operator_str
        
        def extract_constants_from_llm_output(out:str) -> List[str]:
            """parse the constants from llm's output

            Args:
                out (str): llm output string
            """
            constants_start:int = out.lower().find('ground objects')
            if constants_start == -1:
                return []
            constants_start = constants_start + out[constants_start:].find(':') + 1 
            constants_end = out[constants_start:].find('```')
            constants_str:str = out[constants_start:constants_start + constants_end]
            return constants_str.replace(' ','').replace('*','').replace('`','').split('\n')[0].split(',')
        
        def prompt_llm_for_operator_name_params():
            """prompt the LLM for the operator name and parameters

            Returns:
                str: the operator name and parameters
            """
            # get the full state description of novel objects
             
            prompt = propose_operator_prompt.format(
                current_state = current_state,
                goal_state = g_str,
                relevant_objects = relevant_objects,
                novel_objects = novel_objects,
                true_atoms_novel_obj = ', '.join(true_atoms),
                false_atoms_novel_obj = ', '.join(false_atoms),
                object_types = object_types,
                existing_operators = existing_op_params_precond_effect_str,
            )
            out = chat_completion(prompt)
            self.llm_calls += 1
            return extract_operator_from_llm_output(out), extract_constants_from_llm_output(out)
        

        def fill_operator_precondition(proposed_op:OperatorCandidate, param_constants:List[str]):
            """fill the operator precondition with the true and false atoms of the state"""
            true_relevant_atoms, false_relevant_atoms = self.relevant_objs_only_state_description(state, param_constants)
            proposed_op.set_precondition(list(true_relevant_atoms)+list(false_relevant_atoms))
            return proposed_op

            
        def prompt_llm_for_operator_effects(proposed_operator_w_precond_str:str, param_constants:List[str]):
            """prompt the LLM for the operator effects
            Args:
                proposed_operator_w_precond_str (str): the proposed operator with precondition string
                param_constants (List[str]): the list of constants. In this case the parameter objects.
            Returns:
                str: the operator effects
            """
            # check if the proposed operator is empty
            if proposed_operator_w_precond_str == '':
                return ''
            true_atoms, false_atoms = self.full_state_description(state, param_constants)
            full_param_obj_atoms = ', '.join(true_atoms) + ', ' + ', '.join(false_atoms)
            prompt = define_effect_prompt.format(
                full_param_obj_atoms = full_param_obj_atoms,
                example_operators=existing_op_params_precond_effect_str,
                proposed_operator_with_precondition=proposed_operator_w_precond_str
            )
 
            out = chat_completion(prompt)
            self.llm_calls += 1
            return extract_operator_from_llm_output(out)
        
        # prompt the LLM for `num_self_consistency_candidates` number of operators
        counter = OperatorCandidateCounter()
        # query LLM for the name and parameters of the operator
        for _ in range(self.config['planning']['num_op_candidates']):
            proposed_op_str, grounded_params = prompt_llm_for_operator_name_params()
            if proposed_op_str == '': # LLM decided not to propose an operator
                proposed_op:OperatorCandidate = OperatorCandidate('')
                continue
            else:
                proposed_op:OperatorCandidate = self.parse_operator(proposed_op_str)
                if proposed_op.is_empty():
                    continue
                proposed_op.set_grounded_params(grounded_params)
                # standardize the name of the parameters
                params = self.from_grounded_constants_to_lifted(state, grounded_params)
                lifted_params = [f"?{params[c].name} - {params[c].name}" for c in grounded_params if c != '']
                proposed_op.set_parameters(lifted_params)
                # fill the precondition with the true and false atoms involving the parameter objects
                proposed_op = fill_operator_precondition(proposed_op, grounded_params)
                # prompt the LLM for the effect of the operator
                proposed_op_w_effects_str = prompt_llm_for_operator_effects(proposed_op.name_param_precond_repr(), grounded_params)
                # update the operator with effects
                proposed_op = self.parse_operator(proposed_op_w_effects_str)
            counter.add_operator_candidate(proposed_op)
            # early stopping if there's a majority candidate
            if counter.get_max_operator_candidate_count() > self.config['planning']['num_op_candidates'] // 2:
                break
        
        return counter.get_max_operator_candidate()
        

    def search(self) -> Tuple[SearchSpace, SearchStats, List[List[fs.Action]]]:
        """performs search. Calls the LLM agent to create new operators while searching for a plan. 

        Returns:
            tuple: tje search space, the search stats, and the plans found
        """
        # create obj to track state space
        space = SearchSpace()
        stats = SearchStats()
        plans = []

        start_model = GroundForwardSearchModel(self.starting_problem, ground_problem_schemas_into_plain_operators(self.starting_problem))

        open_list = []  # stack storing the nodes which are next to explore in a priority queue
        root = start_model.init()
        heapq.heappush(open_list, (0, make_root_node(root), self.starting_problem))
        closed = {root}

        while open_list:
            # logging.debug("dfs: Iteration {}, #unexplored={}".format(iteration, len(open_)))

            priority, node, problem = heapq.heappop(open_list)
            num_votes = - priority # priority is negative of the number of votes as we want to prioritize the node with the highest number of votes. Python heapq treats smaller number as higher priority.

            problem_w_added_ops = copy.deepcopy(problem)
            # check if the number of llm calls has reached the maximum
            if self.llm_calls > self.max_num_llm_calls - self.config['planning']['num_op_candidates']:
                logging.info(f"Max. number of LLM calls reached. # calls {self.llm_calls}, # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                break
            else: # ask the llm for new operators and update model's operators
                for _ in range(self.max_new_operators_branching_factor): # prompt the LLM to invent up to `max_new_operators_branching_factor` new operators
                    new_op:OperatorCandidate = self.prompt_llm_for_new_operator(problem_w_added_ops, node.state)
                    _ = self.add_operator(problem_w_added_ops, new_op)
                search_ahead_plan = self.search_ahead(problem_w_added_ops, node, self.max_depth)
                # add the plan to the list of plans if it is not None
                if search_ahead_plan:
                    plans.append(search_ahead_plan)
                if len(plans) >= self.config['planning']['min_plan_candidates']:
                    logging.info(f"Minimum number of plans found. # plans: {len(plans)}, # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                    return space, stats, plans
            
            model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))
            
            model_w_added_ops = GroundForwardSearchModel(problem_w_added_ops, ground_problem_schemas_into_plain_operators(problem_w_added_ops))

            if 0 <= self.max_depth <= node.depth: # reached max depth
                logging.info(f"Max. operator proposal expansions reached on one branch at {node}. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                continue
            else: # expand the node and add its children to the open list
                for operator, successor_state in model_w_added_ops.successors(node.state):
                    if successor_state not in closed:
                        num_votes_op = num_votes - 1 # assume one vote per candidate. Now effectively a BFS. TODO: implement voting mechanism. Currently it's effectively a BFS.
                        if model.has_operator(operator): # check if the operator is in the original problem
                            heapq.heappush(open_list, (-num_votes_op, make_child_node(node, operator, successor_state), problem))
                        elif model_w_added_ops.has_operator(operator): # check if the operator is newly added 
                            heapq.heappush(open_list, (-num_votes_op, make_child_node(node, operator, successor_state), problem_w_added_ops))
                        else:
                            raise ValueError(f"Operator {operator} not found in the problem.")
                        closed.add(successor_state)

        logging.info(f"Search space exhausted. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
        space.complete = True
        return space, stats, plans
    
    def search_ahead(self, problem:fs.problem.Problem, start_node, max_depth:int) -> List[fs.Action]:
        """BFS ahead from the current node for a solution to the problem's goal

        Args:
            problem (fs.problem.Problem): the problem to solve
            node (SearchNode): the current node
            max_depth (int): the maximum depth to search
            space (SearchSpace): the search space
            stats (SearchStats): the search stats used for keeping track of expansions

        Returns:
            List[fs.Action]: the plan found if any
        """
        # create obj to track state space
        space = SearchSpace()
        stats = SearchStats()
        model = GroundForwardSearchModel(problem, ground_problem_schemas_into_plain_operators(problem))
        open_list = deque()
        open_list.append(start_node)
        closed = {start_node}
        while open_list:
            stats.iterations += 1
            # logging.debug("dfs: Iteration {}, #unexplored={}".format(iteration, len(open_)))

            node = open_list.popleft()
            if model.is_goal(node.state): # found a plan
                stats.num_goals += 1
                plan = reverse_engineer_plan(node)
                logging.info(f"Goal found after {stats.nexpansions} expansions from node {start_node}. {stats.num_goals} goal states found.")

                # write the operators that resulted in a plan to a file
                domain_file_name = self.config['planning']['planning_dir'] + os.sep + self.config['planning']['modified_planning_domain']
                # find the `.pddl`, insert the num goal before `.pddl` of the file name
                domain_file_name = domain_file_name[:domain_file_name.find('.pddl')] + f'_{stats.num_goals}.pddl'
                
                self.write_domain(problem, domain_file_name)

                # pickle the SearchNode that resulted in a plan
                with open(self.config['planning']['planning_dir'] + os.sep + f"{self.config['planning']['planning_goal_node']}_{stats.num_goals}.pkl", 'wb') as f:
                    dill.dump(node, f)
                return plan # early return if a plan is found since we only care about plan with the shortest length


            if 0 <= max_depth <= node.depth: # reached max depth
                logging.info(f"Max. expansions reached on one branch. # expanded: {stats.nexpansions} from node {start_node}, # goals: {stats.num_goals}.")
                continue
            else: # expand the node and add its children to the open list
                for operator, successor_state in model.successors(node.state):
                    if successor_state not in closed:
                        open_list.append(make_child_node(node, operator, successor_state))
                        closed.add(successor_state)
                        stats.nexpansions += 1

        logging.info(f"Search space exhausted. # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
        space.complete = True
        return None

    def from_grounded_constants_to_lifted(self, model:Union[fs.problem.Problem, Model], constants:List[str]) -> Dict[str, Sort]:
        """converts the grounded constants to lifted constants

        Args:
            problem (fs.problem.Problem): the problem
            constants (List[str]): the grounded constants

        Returns:
            List[str]: the lifted constants
        """
        constants_to_lifted_mapping = {c: model.language.get_constant(c).sort for c in constants if c != ''}
        return constants_to_lifted_mapping

    def relevant_objs_only_state_description(self, state:Model, obj_constants:List[str], grounded=False) -> Tuple[Set[str], Set[str]]:
        """returns the a description of only the relevant objects in the state in the form of true grounded atoms and negated grounded atoms

        Args:
            state (Model): the state
            obj_constants (List[str]): the list of objects
            grounded (bool): whether to return the grounded atoms or the lifted atoms

        Returns:
            Tuple[List[str], List[str]]: the list of true grounded atoms and the list of negated grounded atoms
        """
        dump:dict = state.language.dump()
        preds:list = dump['predicates']
        true_atoms = []
        false_atoms = []
        for pred_dict in preds:
            if isinstance(pred_dict['symbol'], BuiltinPredicateSymbol):
                continue #ignore the built-in predicates
            pred = state.language.get_predicate(pred_dict['symbol'])
            param_constants = []
            for param_sort in pred.sort:
                sort_constants = set(c.name for c in param_sort.domain())
                param_constants.append(sort_constants.intersection(obj_constants))
            # generate combinations of the sets of constants
            comb = list(itertools.product(*param_constants))
            
            # categorize grounded predicates
            for one_comb in comb:
                if len(set(one_comb)) < len(one_comb): # skip if more than one constants in the combination are the same
                    continue
                
                comb_string = f"{pred.name}({', '.join(one_comb)})"
                negated_comb_string = f"not({comb_string})"
                lifted = self.from_grounded_constants_to_lifted(state,one_comb)
                lifted_comb_string = f"({pred.name} {' '.join([f'?{c.name}' for c in lifted.values()])})"
                negated_lifted_comb_string = f"(not ({pred.name} {' '.join([f'?{c.name}' for c in lifted.values()])}))"
                if evaluate(pred(*(state.language.get_constant(c) for c in one_comb)), state): # add all that evaluate to True to true_atoms
                    if grounded:
                        true_atoms.append(comb_string)
                    else:
                        true_atoms.append(lifted_comb_string)
                else: # add all that evaluate to False to false_atoms
                    if grounded:
                        false_atoms.append(negated_comb_string)
                    else:
                        false_atoms.append(negated_lifted_comb_string)
        return set(true_atoms), set(false_atoms)

    def full_state_description(self, state:Model, obj_constants:List[str]) -> Tuple[Set[str], Set[str]]:
        """returns the a description of the objects in the state in the form of true grounded atoms and negated grounded atoms

        Args:
            state (Model): the state
            obj_constants (List[str]): the list of objects

        Returns:
            Tuple[List[str], List[str]]: the list of true grounded atoms and the list of negated grounded atoms
        """
        dump:dict = state.language.dump()
        preds:list = dump['predicates']
        true_atoms = []
        false_atoms = []
        for pred_dict in preds:
            if isinstance(pred_dict['symbol'], BuiltinPredicateSymbol): # ignore the builtin predicates
                continue
            pred = state.language.get_predicate(pred_dict['symbol'])
            param_constants = []
            for param_sort in pred.sort:
                sort_constants = set(c.name for c in param_sort.domain())
                param_constants.append(sort_constants)
            # generate combinations of the sets of constants
            comb = []
            for i, param_constant_set in enumerate(param_constants):
                # temporarily replace param_constant_set with the intersection of param_constant_set and obj_constants
                param_constants[i] = param_constant_set.intersection(obj_constants)
                # add combinations of the sets of constants
                comb.extend(list(itertools.product(*param_constants)))
                # revert param_constant_set back to its original value
                param_constants[i] = param_constant_set
            
            # categorize grounded predicates
            for one_comb in comb:
                if len(set(one_comb)) < len(one_comb): # skip if more than one constants in the combination are the same
                    continue
                comb_string = f"{pred.name}({', '.join(one_comb)})"
                negated_comb_string = f"not({comb_string})"
                if evaluate(pred(*(state.language.get_constant(c) for c in one_comb)), state): # add all that evaluate to True to true_atoms
                    true_atoms.append(comb_string)
                else: # add all that evaluate to False to false_atoms
                    false_atoms.append(negated_comb_string)
        return set(true_atoms), set(false_atoms)

    
if __name__ == '__main__':
    # testing
    planner = HybridSymbolicLLMPlanner(config=load_config('config.yaml'))
    planner.search()