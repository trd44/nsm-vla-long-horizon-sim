import os
import yaml
import copy
import subprocess
from langchain.tools import tool

config_file = "config.yaml"

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
            return "the planner did not find a plan given the problem specification in the problem file and available actions in the domain file. Please double check the actions in the domain file and the init and goal specifications in the problem file", []
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

def extract_predicates(file_content, file_type='domain', scrape_action_conditions=False) -> list:
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

        #for i in range(len(game_action_set)):
        #   game_action_set[i] = applicator[game_action_set[i].split(" ")[0]]
        #for i in range(len(game_action_set)):
        #    for j in range(len(game_action_set[i])):
        #        if game_action_set[i][j] in applicator.keys():
        #            game_action_set[i][j] = applicator[game_action_set[i]]
        return action_set, game_action_set
    return [], []

if __name__ == "__main__":
    config = load_config('config.yaml')
    #verify_predicates_domain("domain.pddl", "domain.pddl")
    # verify_predicates_problem("domain.pddl", "problem.pddl")
    call_planner(config['generic_planning_domain'], config['new_planning_problem'])
