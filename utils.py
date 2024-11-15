import os
import yaml
import copy
import json
import re
import copy
import numpy as np
import subprocess
import base64
from typing import *
from PIL import Image
from langchain.tools import tool
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

config_file = "config.yaml"

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_policy(env, path, lr=0.0003, log_dir=None, seed=0):
    # Load the model
    set_random_seed(seed, using_cuda=True)
    model = SAC.load(path, env=env, learning_rate=lr, tensorboard_log=log_dir, seed=seed)
    return model

def deepcopy_env(env, seed=0):
    saved_sim_state = env.sim.get_state()
    import mimicgen
    env_copy = 
    env_copy.reset()
    env_copy.sim.set_state(saved_sim_state)
    env_copy.sim.forward()
    return env_copy

def find_parentheses(s:str) -> Tuple[int, int]:
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

def split_by_parentheses(s:str, type='operator_predicates') -> List[str]:
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
        part_start, part_end = find_parentheses(s[start:])
        if part_start == -1:
            break
        # add the part inside the parenthesis
        parts.append(s[start + part_start:start + part_end])
        start += part_end + 1
    return parts

@tool
def verify_predicates_domain(old_domain:str, new_domain:str, structure="pddl"):
    """Given two PDDL domain files, this function checks whether the two domain files contain the same predicates. If not, it raises a ValueError with the difference in predicates.

    Args:
        old_domain (str): old domain file name
        new_domain (str): new domain file name
        structure (str, optional): Structure of the domain files. Defaults to "pddl".
    """
    planning_dir = load_config(config_file)["planning_dir"]
    old_domain_path =  planning_dir + os.sep + old_domain
    new_domain_path = planning_dir + os.sep + new_domain
    if structure == "pddl":
        with open(old_domain_path, 'r') as file:
            old_domain_content = file.read()
        with open(new_domain_path, 'r') as file:
            new_domain_content = file.read()
        old_predicates = set(extract_predicates(old_domain_content))
        new_predicates = set(extract_predicates(new_domain_content))
        assert old_predicates == new_predicates, f"Predicates in the new domain file are different from the old domain file. The difference is {old_predicates - new_predicates}. Please make sure the new domain file contains the same predicates from the old domain file."
    return True


@tool
def verify_predicates_problem(domain:str, problem:str, structure="pddl"):
    """Given a PDDL domain and problem file, this function checks whether the problem file only contains predicates that are in the domain file

    Args:
        domain (str): _description_
        problem (str): _description_
        structure (str, optional): _description_. Defaults to "pddl".
    """
    planning_dir = load_config(config_file)["planning_dir"]
    domain_path =  planning_dir + os.sep + domain 
    problem_path = planning_dir + os.sep + problem
    if structure == "pddl":
        with open(domain_path, 'r') as file:
            domain_content = file.read()
        with open(problem_path, 'r') as file:
            problem_content = file.read()
        domain_predicates = set(extract_predicates(domain_content))
        problem_predicates = set(extract_predicates(problem_content, file_type='problem'))
        assert check_predicates_subset(problem_predicates, domain_predicates), f"Predicates in the problem file are not a subset of the domain file. The difference is {problem_predicates - domain_predicates}. Please make sure the problem file only contains predicates that are in the domain file."
    return True

@tool
def call_planner(domain:str, problem:str, structure="pddl"):
    """Given a domain and a problem file, this function return the ffmetric Planner output in the action format

    Args:
        domain (str): domain file name
        problem (str): problem file name
        structure (str, optional): The type of the files for planning. Defaults to "pddl".

    Returns:
        _type_: _description_
    """
    planning_dir = load_config(config_file)["planning_dir"]
    domain_path =  planning_dir + os.sep + domain
    problem_path = planning_dir + os.sep + problem
    if structure == "pddl":
        run_script = f"{planning_dir}/Metric-FF-v2.1/./ff -o {domain_path} -f {problem_path} -s 0"
        output = subprocess.getoutput(run_script)
        
        if "unsolvable" in output or "goal can be simplified to FALSE" in output: # unsolvable
            return "the planner did not find a plan given the problem specification in the problem file and available actions in the domain file. Please double check the actions in the domain file.", []
        elif 'ff: found legal plan as follows\n' not in output: # symbolic planning specifications have errors
            return "The planner failed due to errors in the domain and/or problem specifications. Please double check the syntax and semantics in the domain and problem files:\n{}".format(output), []
        try:
            output = output.split('ff: found legal plan as follows\n')[1]
            output = output.split('\ntime spent:')[0]
            # Remove empty lines
            output = os.linesep.join([s for s in output.splitlines() if s])
        except Exception as e:
            return "The planner failed.\nThe output of the planner was:\n{}".format(output), []

        plan, _ = _output_to_plan(output, structure=structure)
        return "successfully found a plan", plan
    elif structure == "hddl":
        run_script = f"{planning_dir}/lilotane/build/lilotane {domain_path} {problem_path} -v=0 -cs"# | cut -d' ' -f2- | sed 1,2d | head -n -2" # > + sub_plan_name
        output = subprocess.getoutput(run_script)
        #TODO：implement logic for processing the output
        return "planner output processing not implemented yet", []


@tool          
def read_file(file_path:str):
    """read the file from the given file_path
    """
    # read the file from the given file_path
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            return file_content
    except Exception as e:
        return e

@tool
def write_file(file_path:str, content:str):
    """write the `content` to file_path"""
    try:
        with open(file_path, 'w') as file:
            file.write(content)
            return True
    except Exception as e:
        return e

def extract_name_params_from_grounded(grounded_operator:str):
    """extract the name and parameters from a grounded operator such as `pick-up-from-tabletop(mug1, table1, gripper1)`

    Args:
        grounded_operator (str): the grounded operator
    """
    # extract the name
    name = grounded_operator.split('(')[0]
    # extract the parameters into a list
    params = grounded_operator.split('(')[1].replace(')', '').split(', ')
    return name, params


def parse_pddl_types(pddl_content):
    # Regular expression to match the :types section
    types_section_pattern = re.compile(r'\(:types\s+(.*?)\s+\)', re.DOTALL)
    match = types_section_pattern.search(pddl_content)
    
    if not match:
        raise ValueError("No :types section found in the PDDL content.")
    
    types_section = match.group(1).strip()
    
    # Split the types section into lines and parse each line
    types_lines = types_section.split('\n')
    types_hierarchy = {}
    
    for line in types_lines:
        line = line.strip()
        if '-' in line:
            types, parent = line.split('-')
            parent = parent.strip()
            types = [t.strip() for t in types.split()]
        else:
            types = [t.strip() for t in line.split()]
            parent = None
        
        for t in types:
            if parent:
                if parent not in types_hierarchy:
                    types_hierarchy[parent] = []
                types_hierarchy[parent].append(t)
            else:
                if t not in types_hierarchy:
                    types_hierarchy[t] = []
    
    return types_hierarchy

def build_hierarchical_json(types_hierarchy):
    def build_tree(node):
        if node not in types_hierarchy or not types_hierarchy[node]:
            return {}
        return {child: build_tree(child) for child in types_hierarchy[node]}
    
    root_nodes = [node for node in types_hierarchy if not any(node in children for children in types_hierarchy.values())]
    hierarchical_json = {root: build_tree(root) for root in root_nodes}
    
    return hierarchical_json


def check_predicates_subset(problem_predicates, domain_predicates):
    # Parse predicates
    parsed_problem_predicates = {parse_predicate(pred, grounded=True) for pred in problem_predicates}
    parsed_domain_predicates = {parse_predicate(pred) for pred in domain_predicates}
    
    # check if the problem_predicates are a subset of the domain_predicates by comparing the name of each predicate and the number of arguments
    return parsed_problem_predicates.issubset(parsed_domain_predicates)


def parse_predicate(predicate, grounded=False):
    """
    Parse a predicate string to extract its name and the number of arguments.
    Example: 'holding ?obj - holdable' -> ('holding', 1)
    Example: 'not (free gripper)' -> ('free', 1)
    """
    # replace parentheses with spaces and split the predicate string
    predicate = predicate.replace('(', '').replace(')', '')
    parts = predicate.split()
    name = parts[0]
    if name == 'not':
        parts = parts[1:]
        name = parts[0]
    # count the number of arguments in a predicate like '(free gripper)'
    if grounded:
        num_args = len(parts) - 1
    else:
        num_args = len([part for part in parts if part.startswith('?')])
    return (name, num_args)

def parse_ground_predicate(predicate:str, problem_objects:dict) -> dict:
    """parse a ground predicate string to extract its name and arguments

    Args:
        predicate (str): a ground predicate string in lisp format
        problem_objects (dict): a dictionary of objects in the problem file
    Returns:
        dict: dictionary containing the predicate name, value and arguments
    """
    def find_arg_type(arg_name:str, problem_objects:dict) -> str:
        for obj_type, obj_list in problem_objects.items():
            if arg_name in obj_list:
                return obj_type
        return None
    
    predicate = predicate.replace('(', '').replace(')', '')
    parts = predicate.split()
    name = parts[0]
    val = True
    arg_start_index = 1
    if name == 'not':
        name = parts[1]
        val = False
        arg_start_index = 2
    args = {}
    for i in range(arg_start_index, len(parts)):
        arg_name = parts[i]
        arg_type = find_arg_type(arg_name, problem_objects)
        args[arg_name] = arg_type
    return {'name': name, 'value': val, 'args': args}

def extract_predicates(file_content, file_type='domain', scrape_action_conditions=False) -> List[str]:
    """extracts predicates from a symbolic planning file

    Args:
        file_content (str): content of the symbolic planning file
        file_type (str, optional): type of the file. Defaults to 'domain'.
        check_action_conditions (bool, optional): whether to scrape the predicates in actions preconditions and effects. Defaults to False.

    Returns:
        list: list of extracted predicates
    """
    if file_type == 'domain':
        start_index = file_content.find("(:predicates")
    else:
        start_index = file_content.find("(:init")
        scrape_action_conditions = False

    # find the next matching ')' after the start_index
    stack = []
    end_index = start_index + 1
    while end_index < len(file_content):
        if file_content[end_index] == '(':
            stack.append('(')
        elif file_content[end_index] == ')':
            if len(stack) == 0:
                break
            stack.pop()
        end_index += 1
    if len(stack) > 0:
        raise ValueError("Mismatched parentheses in file_content")
    
    predicates_str = file_content[start_index:end_index+1]
    # remove eight spaces before predicates
    predicates_str = predicates_str.replace("        ", "")
    predicates_str = predicates_str.replace("\t", "")
    predicates = predicates_str.split("\n")[1:-1]

    if scrape_action_conditions: # extract predicates from action preconditions and effects
        action_conditions = file_content.split("(:action")
    return predicates


def _output_to_plan(output, structure):
    '''
    Helper function to perform regex on the output from the planner.
    ### I/P: Takes in the ffmetric output and
    ### O/P: converts it to a action sequence list.
    '''
    if structure == "pddl":
        action_set = []
        for action in output.split("\n"):
            #if action.startswith('step'):
            try:
                action_set.append(''.join(action.split(": ")[1]))
            except IndexError:
                return False, False
        
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)
        return action_set, game_action_set
    return [], []



def save_agent_view_image(image:np.array):
    """Save the image of the agent's view to a file

    Args:
        image (np.array): image of the agent's view
    """
    image = Image.fromarray(image)
    # PIL image is flipped, so flip it back
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(config['image_path'])

def encode_image(image_path:str):
    """Encode the image at `image_path`

    Args:
        image_path (str): path to the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_type_parent_mapping(pddl_domain_file):
    """Extract the types from the :types section of a PDDL domain file. The types are extracted as a dictionary where the key is the type and the value is the parent type. 

    Args:
        pddl_domain_file (str): path to the domain file

    Returns:
        dict: type -> parent_type mapping
    """
    type_hierarchy = {}
    
    with open(pddl_domain_file, 'r') as file:
        content = file.read()
        
        # Extract the :types section using regex
        types_section = re.findall(r":types(.*?)(?=\))", content, re.DOTALL)
        
        if types_section:
            # Split by newlines to get the individual lines in the types section
            lines = types_section[0].strip().split("\n")

            for line in lines:
                # Split by " - " to get the child and parent types
                if " - " in line:
                    children, parent = line.split(" - ")
                    children = children.split()
                    for child in children:
                        type_hierarchy[child.strip()] = parent.strip()
                else:
                    child = line.strip()
                    parent = None
                    type_hierarchy[child.strip()] = parent
            
    
    return type_hierarchy

def matches_type(obj_type:str, expected_type:str, type_parent_mapping:dict) -> bool:
    """Given an object and an expected type, this function checks if the object matches the expected type by traversing the type hierarchy.

    Args:
        obj_type (str): type of the object   
        expected_type (str): expected type
        type_parent_mapping (dict): mapping from type to parent type

    Returns:
        bool: True if the object matches the expected type, False otherwise
    """
    current_type = obj_type
    while current_type:
        if current_type == expected_type:
            return True
        current_type = type_parent_mapping.get(current_type)
    return False

def extract_objects_from_problem(problem_file:str) -> dict:
    """Extracts the objects from a problem file in lisp format.

    Args:
        problem_file (str): the path to the problem file

    Returns:
        dict: object to type mapping
    """
    res = dict()
    with open(problem_file, 'r') as file:
        content = file.read()
        objects_section = re.findall(r":objects(.*?)(?=\))", content, re.DOTALL)
        if objects_section:
            lines = objects_section[0].strip().split("\n")
            for line in lines:
                # split the object into objects before ' - ' and its type after ' - '
                objs, obj_type = line.split(" - ")
                res[obj_type] = [obj.strip() for obj in objs.split()]
        return res

def extract_ground_predicates_from_init(problem_file:str) -> List[dict]:
    """Extract the ground applicable predicates from the `:init` section of a problem file.` Example:
    {
        "name": "in",
        "value": true,
        "args": {
            "?coffee-pod1": "coffee-pod",
            "?drawer1": "drawer"
        }
    },

    Args:
        problem_file (str): path to the problem file
    Returns:
        list: list of ground predicates
    """
    # get the problem objects
    problem_objects = extract_objects_from_problem(problem_file)
    # read the file content
    with open(problem_file, 'r') as file:
        content = file.read()
        # extract the :init section using regex
        preds:List[str] = extract_predicates(content, file_type='problem')
        res = [parse_ground_predicate(pred, problem_objects) for pred in preds]
    return res


def extract_predicates_from_domain(domain_file:str) -> dict:
    """Extracts the predicates from a domain file in list format.

    Args:
        domain_file (str): file path

    Returns:
        list: dict
    """
    res = dict()
    # Read the domain file
    with open(domain_file, 'r') as file:
        content = file.read()
        # Extract the predicates
        extracted_predicates = extract_predicates(content)
        for pred in extracted_predicates:
            pred_dict = extract_predicate(pred)
            res[pred_dict['name']] = pred_dict
        return res


def extract_predicate(predicate_str_lisp:str) -> dict:
    """Given a predicate in lisp format, this function extracts the predicate name and arguments and returns them as a dictionary. Example: '(holding ?obj - holdable)' -> {'name': 'holding', 'args': {'?obj': 'holdable'}}

    Args:
        predicate_str_lisp (str): the predicate in lisp format

    Returns:
        dict: dictionary containing the predicate name and arguments
    """
    predicate_str_lisp = predicate_str_lisp.replace('(', '').replace(')', '')
    parts = predicate_str_lisp.split()
    name = parts[0]
    args = {}
    for i in range(1, len(parts)):
        if parts[i].startswith('?'):
            arg_name = parts[i]
            arg_type = parts[i+2]
            args[arg_name] = arg_type
    return {'name': name, 'args': args}

def find_applicable_predicates(type_name:str, type_parent_mapping:dict, predicates:List[dict]) -> set:
    """Given a type name, this function finds the applicable predicates for that type by traversing the type hierarchy.

    Args:
        type_name (str): the name of the type
        type_parent_mapping (dict): mapping from type to parent type
        predicates (list): a list of predicates in dictionary format

    Returns:
        set: the set of all applicable predicates
    """
    # Initialize a set to store applicable predicates
    applicable_predicates = []
    
    # Function to traverse the type hierarchy
    def traverse_type_hierarchy(current_type):
        # Iterate through all predicates
        for predicate in predicates.values():
            for arg, arg_type in predicate['args'].items():
                if arg_type == current_type:
                    applicable_pred = copy.deepcopy(predicate)
                    applicable_pred['args'][arg] = type_name
                    applicable_predicates.append(applicable_pred)
        
        # Recursively traverse the parent type
        parent_type = type_parent_mapping.get(current_type)
        if parent_type:
            traverse_type_hierarchy(parent_type)
    
    # Start traversal from the given type
    traverse_type_hierarchy(type_name)
    
    return applicable_predicates

def extract_applicable_truth_assignments(problem_file:str, applicable_predicates:set) -> dict:
    """Given the problem file and a set of applicable predicates, this function extracts the truth assignments for the applicable predicates from the problem file

    Args:
        problem_file (str): _description_
        applicable_predicates (set): _description_

    Returns:
        dict: _description_
    """
    pass


def generate_complete_value_assignments_relevant_to_obj(obj:str, obj_type:str, applicable_predicates:List[dict], ground_init_preds:List[dict],problem_objects:dict) -> List[dict]:
    """Given an object, a list of applicable predicates, the ground init predicates, and the problem objects, this function generates the complete value assignments to the applicable predicates

    Args:
        obj (str): the object
        onj_type (str): the type of the object
        applicable_predicates (List[dict]): the list of applicable predicates for the object
        ground_init_preds (List[dict]): the ground init predicates from a problem file
        problem_objects (dict): the objects in the problem file

    Returns:
        List[dict]: value assignment to the applicable predicates
    """
    # get the type parent mapping
    type_parent_mapping = extract_type_parent_mapping('Planning/PDDL/llm_success_trial1_domain.pddl')
    # iterate over the applicable predicates
    res = []
    for pred in applicable_predicates:
        true_pred = {'name': pred['name'], 'value': True, 'args': {}}
        false_pred = {'name': pred['name'], 'value': False, 'args': {}}
        # fill the object in the appropriate argument
        args = pred['args']
        for arg, arg_type in args.items():
            if arg_type == obj_type:
                true_pred['args'][obj] = arg_type
                false_pred['args'][obj] = arg_type
            else: # find an object of the type in the problem objects

                for obj_type in problem_objects:
                    if matches_type(obj_type, arg_type, type_parent_mapping):
                    # iterate over the actual objects for each obj type
                        detected_objs = problem_objects[obj_type]
                        for detected_obj in detected_objs:
                            true_pred['args'][detected_obj] = obj_type
                            false_pred['args'][detected_obj] = obj_type
                            break


def save_to_json(hierarchy, json_file):
    with open(json_file, 'w') as file:
        json.dump(hierarchy, file, indent=4)

if __name__ == "__main__":
    # type_parent_mapping = extract_type_parent_mapping('Planning/PDDL/llm_success_trial1_domain.pddl')
    # save_to_json(type_parent_mapping, 'type_parent_mapping.json')
    # preds = extract_predicates_from_domain('Planning/PDDL/llm_success_trial1_domain.pddl')
    # save_to_json(preds, 'predicates.json')
    # applicable_preds = find_applicable_predicates('drawer', type_parent_mapping, preds)
    # save_to_json(applicable_preds, 'applicable_preds.json')

    # objects = extract_objects_from_problem('Planning/PDDL/llm_success_trial1_problem.pddl')
    # save_to_json(objects, 'problem_objects.json')

    ground_preds = extract_ground_predicates_from_init('Planning/PDDL/llm_success_trial1_problem.pddl')
    save_to_json(ground_preds, 'ground_init_preds.json')

    

