import os
import logging
import heapq
import copy
import itertools
from collections import deque

from tarski.search import GroundForwardSearchModel
from tarski.search.model import progress
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader, FstripsWriter
from tarski.syntax import land, neg, CompoundFormula
from tarski.syntax.formulas import VariableBinding
from tarski.syntax.builtins import *
from tarski import fstrips as fs
from tarski.evaluators.simple import evaluate
from tarski.model import Model
from tarski.syntax.builtins import BuiltinPredicateSymbol
from VLM.TreeOfThoughts import *
from VLM.TreeOfThoughtsPrompts import *
from utils import *

class HybridSymbolicLLMPlanner:
    def __init__(self, config_file='config.yaml'):
        self.config = load_config(config_file)
        self.reader = FstripsReader(raise_on_error=True)
        self.parse_domain()
        self.starting_problem = self.parse_problem()
        self.max_new_operators_branching_factor = self.config['max_new_operators_branching_factor']
        self.llm_calls = 0
        self.max_depth = self.config['max_depth']
        self.max_num_llm_calls = self.config['max_num_llm_calls']
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.novel_objects = self.config['novel_objects']
    
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
    
    def parse_operators(self, operators:str) -> Dict[str, Dict[str, List[str]]]:
        """parse the operators string into a dictionary

        Args:
            operators (str): the operators string with operators separated by newlines

        Returns:
            Dict[str, Dict[str, List[str]]]: the dictionary with the operators
        """
        operators = operators.split('(:action')
        operators_dict = {}
        if len(operators) == 0:
            return operators_dict
        for operator in operators[1:]:
            # the string between `:action` and :parameters is the operator name
            operator_name = operator.split(':parameter')[0].strip().replace('\n', '')
            # the string between `:parameters` and :precondition is the parameters
            param_start = operator.find(':parameter')
            param_end = operator.find(':precondition')
            p = self._find_parentheses(operator[param_start:param_end])
            parameters = operator[param_start+p[0]:param_start+p[1]]
            # split the parameters into a list of parameters at the space before the `?` but keep the `?`
            parameters = ['?' + param.strip() for param in parameters.split('?') if param]
            # the string between `:precondition` and :effects is the precondition
            precondition_start = operator.find(':precondition')
            precondition_end = operator.find(':effects')
            p = self._find_parentheses(operator[precondition_start:precondition_end])
            precondition = operator[precondition_start+p[0]:precondition_start+p[1]]
            # split the precondition into a list of predicates by matching parentheses
            precondition = self._split_by_parentheses(precondition)
            # the string between `:effects` and the end of the operator is the effects
            effects_start = operator.find(':effect')
            effects_end = operator.find('\n)', effects_start)
            p = self._find_parentheses(operator[effects_start:effects_end])
            effects = operator[effects_start+p[0]:effects_start+p[1]]
            # split the effects into a list of predicates by matching parentheses
            effects = self._split_by_parentheses(effects)
            operators_dict[operator_name] = {
                'parameters': sorted(parameters),
                'precondition': sorted(precondition),
                'effects': sorted(effects)
            }
        return operators_dict


    def add_operator(self, problem:fs.problem.Problem, operator_name:str, parameters:List[str], precondition:List[str], effects:List[str]):
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
        

        parse_parameters(parameters, params_dict)
        params_varbinding = VariableBinding(list(params_dict.values()))
        precondition_formula:CompoundFormula = parse_precondition(precondition)
        effects_list:List[fs.SingleEffect] = parse_effects(effects)
        
        problem.action(
            name=operator_name,
            parameters=params_varbinding,
            precondition=precondition_formula,
            effects=effects_list
        )
        return operator_name

    def update_operators(self, problem:fs.problem.Problem, operators:str) -> Dict[str, Dict[str, List[str]]]:
        """update the problem with the operators

        Args:
            problem (fs.problem.Problem): the problem to update
            operators (str): the operators string with operators separated by newlines
        Returns:
            dict: the dictionary representation of the operators added
        """
        parsed_dict = self.parse_operators(operators)
        for operator_name, operator in parsed_dict.items():
            self.add_operator(
                problem=problem,
                operator_name=operator_name,
                parameters=operator['parameters'],
                precondition=operator['precondition'],
                effects=operator['effects']
            )
        return parsed_dict
    
    def write_domain(self, problem:fs.problem.Problem, domain_file:str):
        """writes the domain to a file

        Args:
            problem (fs.problem.Problem): the problem to write
            domain_file (str): the file to write the domain to
        """
        writer = FstripsWriter(problem)
        writer.write_domain(domain_file, constant_objects=None)

    def prompt_llm_for_new_operator(self, problem:fs.problem.Problem, state:Model) -> str:
        """prompts the LLM for new operators

        Args:
            problem (fs.problem.Problem): the current problem
            state (fs.State): the current state

        Returns:
            str: the operators string
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
        parentheses = self._find_parentheses(self.reader.domain_text[types_start:])
        object_types = self.reader.domain_text[types_start:types_start+parentheses[1]]

        # predicates are in the section `:predicates` of domain text
        preds_start = self.reader.domain_text.find('(:predicates')
        preds_end = self.reader.domain_text.find('(:action')
        preds_section = self.reader.domain_text[preds_start:preds_end]
        parentheses = self._find_parentheses(preds_section)
        preds = preds_section[parentheses[0]:parentheses[1]]
        available_predicates = '\n'.join(self._split_by_parentheses(preds, type='predicates'))

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
        existing_op_params_str = '\n'.join(existing_op_params)
        existing_op_params_precond_str = '\n'.join(existing_op_params_precond)
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
            proposed_operator_parentheses:tuple = self._find_parentheses(out[proposed_operator_start:])
            proposed_operator_str:str = out[proposed_operator_start + proposed_operator_parentheses[0] - 1:proposed_operator_start + proposed_operator_parentheses[1]+1] # include the parantheses
            return proposed_operator_str
        
        def extract_constants_from_llm_output(out:str) -> List[str]:
            """parse the constants from llm's output

            Args:
                out (str): llm output string
            """
            constants_start:int = out.find('ground objects: ')
            if constants_start == -1:
                return []
            constants_start += len('ground objects: ')
            constants_end = out[constants_start:].find('```')
            constants_str:str = out[constants_start:constants_start + constants_end]
            return constants_str.replace('\n','').replace(' ','').split(',')
        
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
                existing_operators = existing_op_params_str,
            ) 
            out = self.thought_generator.invoke([SystemMessage(content=prompt)])
            self.llm_calls += 1
            return extract_operator_from_llm_output(out.content), extract_constants_from_llm_output(out.content)
        
        def prompt_llm_for_operator_precondition(proposed_operator_str:str, param_constants:List[str]):
            """prompt the LLM for the operator precondition
            Args:
                proposed_operator_str (str): the proposed operator string
                param_constants (List[str]): the list of constants. In this case the parameter objects.
            Returns:
                str: the operator precondition
            """
            true_atoms, false_atoms = self.full_state_description(state, param_constants)
            full_param_obj_atoms = ', '.join(true_atoms) + ', ' + ', '.join(false_atoms)
            prompt = define_precondition_prompt.format(
                full_param_obj_atoms = full_param_obj_atoms,
                example_operators=existing_op_params_precond_str,
                proposed_operator=proposed_operator_str
            ) 
            out = self.thought_generator.invoke([SystemMessage(content=prompt)])
            self.llm_calls += 1
            return extract_operator_from_llm_output(out.content)
            
        def prompt_llm_for_operator_effects(proposed_operator_w_precond_str:str, param_constants:List[str]):
            """prompt the LLM for the operator effects
            Args:
                proposed_operator_w_precond_str (str): the proposed operator with precondition string
                param_constants (List[str]): the list of constants. In this case the parameter objects.
            Returns:
                str: the operator effects
            """
            true_atoms, false_atoms = self.full_state_description(state, param_constants)
            full_param_obj_atoms = ', '.join(true_atoms) + ', ' + ', '.join(false_atoms)
            prompt = define_effect_prompt.format(
                full_param_obj_atoms = full_param_obj_atoms,
                example_operators=existing_op_params_precond_effect_str,
                proposed_operator_with_precondition=proposed_operator_w_precond_str
            ) 
            out = self.thought_generator.invoke([SystemMessage(content=prompt)])
            self.llm_calls += 1
            return extract_operator_from_llm_output(out.content)
        
        # prompt the LLM for `num_self_consistency_candiates` number of operators
        proposed_op_name_params_count = {}
        # query LLM for the name and parameters of the operator
        for _ in range(self.config['num_op_candidates']):
            proposed_operator_str, grounded_params = prompt_llm_for_operator_name_params()
            if len(grounded_params) == 0 or proposed_operator_str == '':
                name_params = ''
            else:
                proposed_op = self.parse_operators(proposed_operator_str)
                name = list(proposed_op.keys())[0]
                params = proposed_op[name]['parameters']
                name_params = \
                f"(:action {name}\n\t:parameters ({' '.join(sorted(params))})\n)"
            proposed_op:dict = proposed_op_name_params_count.get(name_params, {})
            proposed_op['grounded_params'] = grounded_params
            proposed_op['count'] = proposed_op.get('count', 0) + 1
            proposed_op_name_params_count[name_params] = proposed_op
            if proposed_op_name_params_count[name_params]['count'] > self.config['num_op_candidates'] // 2: #already found majority candidate
                break
        # find the operator with the highest `count`
        proposed_operator_str = max(proposed_op_name_params_count, key=lambda x: proposed_op_name_params_count[x]['count'])
        grounded_params = proposed_op_name_params_count[proposed_operator_str]['grounded_params']
        # check if no operator was proposed
        if not proposed_operator_str or proposed_operator_str == '':
            return ''
        
        # otherwise, count different proposals of preconditions
        op_precond_count = {}
        # query LLM for the precondition of the operator
        for _ in range(self.config['num_precond_candidates']):
            proposed_operator_w_precond_str = prompt_llm_for_operator_precondition(proposed_operator_str, grounded_params)
            proposed_op = self.parse_operators(proposed_operator_w_precond_str)
            name = list(proposed_op.keys())[0]
            params = proposed_op[name]['parameters']
            precond = [f"({p})" for p in proposed_op[name]['precondition']]
            name_params_precond = \
            f"(:action {name}\n\t:parameters ({' '.join(sorted(params))})\n\t:precondition (and {' '.join(sorted(precond))})\n)"
            op_precond_count[name_params_precond] = op_precond_count.get(name_params_precond, 0) + 1
            if op_precond_count[name_params_precond] > self.config['num_precond_candidates'] // 2: # already found the majority candidate
                break
        # find the precondition with the highest count
        proposed_operator_w_precond_str = max(op_precond_count, key=op_precond_count.get)

        # query LLM for the effects of the operator
        op_effect_count = {}
        for _ in range(self.config['num_effect_candidates']):
            proposed_operator_w_precond_effects_str = prompt_llm_for_operator_effects(proposed_operator_w_precond_str, grounded_params)
            proposed_op = self.parse_operators(proposed_operator_w_precond_effects_str)
            name = list(proposed_op.keys())[0]
            params = proposed_op[name]['parameters']
            precond = [f"({p})" for p in proposed_op[name]['precondition']]
            effect = [f"({p})" for p in proposed_op[name]['effects']]
            name_params_precond_effects = \
            f"(:action {name}\n\t:parameters ({' '.join(sorted(params))})\n\t:precondition (and {' '.join(sorted(precond))})\n\t:effect (and {' '.join(sorted(effect))})\n)"
            op_effect_count[name_params_precond_effects] = op_effect_count.get(name_params_precond_effects, 0) + 1
            if op_effect_count[name_params_precond_effects] > self.config['num_effect_candidates'] // 2: # already found the majority candidate
                break
        # find the effect with the highest count
        proposed_operator_w_precond_effects_str = max(op_effect_count, key=op_effect_count.get)
        return proposed_operator_w_precond_effects_str
        

    def search(self) -> List[List[fs.Action]]:
        """performs search. Calls the LLM agent to create new operators while searching for a plan. 
        
        Args:
            starting_problem (fs.problem.Problem): the starting problem with the original operators

        Returns:
            list: the list of plans
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
            if self.llm_calls >= self.max_num_llm_calls:
                logging.info(f"Max. number of LLM calls reached. # calls {self.llm_calls}, # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                break
            else: # ask the llm for new operators and update model's operators
                for _ in range(self.max_new_operators_branching_factor): # prompt the LLM to invent up to `max_new_operators_branching_factor` new operators
                    new_operators_str = self.prompt_llm_for_new_operator(problem_w_added_ops, node.state)
                    _ = self.update_operators(problem_w_added_ops, new_operators_str)
                search_ahead_plan = self.search_ahead(problem_w_added_ops, node, self.max_depth)
                # add the plan to the list of plans if it is not None
                if search_ahead_plan:
                    plans.append(search_ahead_plan)
                if len(plans) >= self.config['min_plan_candidates']:
                    logging.info(f"Minimum number of plans found. # plans: {len(plans)}, # expanded: {stats.nexpansions}, # goals: {stats.num_goals}.")
                    return plans
            
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
    
    def search_ahead(self, problem:fs.problem.Problem, start_node, max_depth:int) -> List[List[fs.Action]]:
        """BFS ahead from the current node for a solution to the problem's goal

        Args:
            problem (fs.problem.Problem): the problem to solve
            node (SearchNode): the current node
            max_depth (int): the maximum depth to search
            space (SearchSpace): the search space
            stats (SearchStats): the search stats

        Returns:
            List[List[fs.Action]]: _description_
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
                domain_file_name = self.config['planning_dir'] + os.sep + self.config['modified_planning_domain']
                # find the `.pddl`, insert the num goal before `.pddl` of the file name
                domain_file_name = domain_file_name[:domain_file_name.find('.pddl')] + f'_{stats.num_goals}.pddl'
                
                self.write_domain(problem, domain_file_name)
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


    def _find_parentheses(self, s:str) -> Tuple[int, int]:
            """returns the indices of the first opening and matching closing parentheses

            Args:
                s (str): the string to search

            Returns:
                Tuple[int, int]: the indices of the first opening and matching closing parentheses
            """
            count = 0
            start = 0
            for i, c in enumerate(s):
                if c == '(':
                    if count == 0:
                        start = i
                    count += 1
                elif c == ')':
                    count -= 1
                    if count == 0:
                        return start + 1, i
            return -1, -1

    def _split_by_parentheses(self, s:str, type='operator_predicates') -> List[str]:
        """splits a string by parentheses

        Args:
            s (str): the string to split

        Returns:
            List[str]: the list of strings
        """
        if type=='operator_predicates':
            if s.find('and') == -1:
                return [s]
        parts = []
        start = 0
        while start < len(s):
            part_start, part_end = self._find_parentheses(s[start:])
            if part_start == -1:
                break
            # add the part inside the parenthesis
            parts.append(s[start + part_start:start + part_end])
            start += part_end + 1
        return parts


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

    
if __name__ == '__main__':
    planner = HybridSymbolicLLMPlanner()
    planner.search()