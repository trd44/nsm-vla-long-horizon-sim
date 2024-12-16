import os
import importlib
from stable_baselines3 import SAC
from utils import *
from robosuite.wrappers import GymWrapper
from robosuite.devices import Keyboard
from learning.learner import OperatorWrapper
from tarski import fstrips as fs

def find_grounded_operator(plan:List[fs.Action], operator_name:str) -> Union[fs.Action, List[fs.Action]]:
    """find the grounded operator in the plan based on the operator name and return the operator and the operators that are executed before the operator to find in the plan

    Args:
        plan (List[fs.Action]): a list of grounded operators
        operator_name (str): the name of the operator to find

    Returns:
        fs.Action: the grounded operator
        List[fs.Action]: the operators that are executed before the operator to find in the plan
    """
    prior_ops = [] # operators that are executed before the operator to find in the plan
    for grounded_operator in plan:
        op_name, _ = extract_name_params_from_grounded(grounded_operator.ident())
        if op_name == operator_name:
            return grounded_operator, prior_ops
        else:
            prior_ops.append(grounded_operator)
    raise Exception(f"Operator {operator_name} not found in the plan")

def choose_policy(config:dict) -> Tuple[SAC, GymWrapper]:
    """Prints out the RL policies available in stable_baselines3, and returns the selected policy choice
    Args:
        config (dict): the configuration dictionary containig the domain and planning information
    Returns:
        SAC: the selected policy
        GymWrapper: the environment for the selected policy
    """
    plan = load_plan(config)
    domain = config['planning']['domain']
    # read directories under `learning/policies` to get the list of policies
    policies_dir = f"learning{os.sep}policies"

    # select operator to visualize
    for i, operator in enumerate(os.listdir(f"{policies_dir}{os.sep}{domain}")):
        print(f"[{i}] {operator}")
    print()
    try:
        operator = int(input(f"Choose an operator to visualize (enter a number from 0 to {len(os.listdir(f'{policies_dir}{os.sep}{domain}')) - 1}): "))
        operator = min(max(operator, 0), len(os.listdir(f"{policies_dir}{os.sep}{domain}")))
    except:
        operator = 0
        print(f"Input is not valid. Use {os.listdir(f'{policies_dir}{os.sep}{domain}')[operator]} by default.\n")
    chosen_operator = os.listdir(f"{policies_dir}{os.sep}{domain}")[operator]
    print(f"Chosen operator: {chosen_operator}\n")

    # select seed to visualize
    for i, seed in enumerate(os.listdir(f"{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}")):
        print(f"[{i}] {seed}")
    print()
    try:
        seed = int(input(f"Choose a seed to visualize (enter a number from 0 to {len(os.listdir(f'{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}')) - 1}): "))
        seed = min(max(seed, 0), len(os.listdir(f"{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}")))
    except:
        seed = 0
        print(f"Input is not valid. Use {os.listdir(f'{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}')[seed]} by default.\n")
    chosen_seed = os.listdir(f"{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}")[seed]
    print(f"Chosen seed: {chosen_seed}\n")

    # TODO: select model to visualize
    # for i, model in enumerate(os.listdir(f"{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}{os.sep}{chosen_seed}")):
    #     print(f"[{i}] {model}")
    
    # load the environment for the selected policy
    # make sure `has_renderer` is set to True in the config file
    config['eval_simulation']['has_renderer'] = True
    env = load_env(domain, config['eval_simulation'])
    env = GymWrapper(env)
    op, prior_ops = find_grounded_operator(plan, operator_name=chosen_operator)
    prior_ops_executors_mapping = OrderedDict({op: load_executor(config, op) for op in prior_ops})
    env = OperatorWrapper(env=env, grounded_operator=op, executed_operators=prior_ops_executors_mapping, config=config)
    device = Keyboard()
    env.viewer.add_keypress_callback(device.on_press)
    device.start_control()
    # load the policy
    model = SAC.load(f"{policies_dir}{os.sep}{domain}{os.sep}{chosen_operator}{os.sep}{chosen_seed}{os.sep}best_model{os.sep}best_model.zip", env=env)
    return model, env

def visualize_policy(config:dict):
    """Visualize the learned policy
    Args:
        config (dict): the configuration dictionary containig the domain and planning information
    """
    model, env = choose_policy(config)
    obs, _ = env.reset()
    n_success = 0
    for _ in range(config['learning']['eval']['n_eval_episodes']):
        done = False
        truncated = False
        while not (done or truncated):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            if done or truncated:
                obs, _ = env.reset()
                n_success += 1 if reward == 1 else 0
    print(f"ran {config['learning']['eval']['n_eval_episodes']} episodes, was successful in {n_success} episodes")

if __name__ == "__main__":
    config = load_config("config.yaml")
    visualize_policy(config)
