# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics: Dict[str, Union[bool, float, np.array]], grounded_effect: str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicating percentage of completion towards the grounded effect
    '''
    # Parse the grounded effect
    effect = grounded_effect.strip()
    is_negated = False

    # Check if the effect is negated
    if effect.startswith('(not '):
        is_negated = True
        # Remove '(not ' from the start and ')' from the end
        effect = effect[5:-1].strip()
    elif effect.startswith('not '):
        is_negated = True
        effect = effect[4:].strip()

    # Remove outer parentheses if they exist
    if effect.startswith('(') and effect.endswith(')'):
        effect = effect[1:-1].strip()

    # Split the effect into predicate and arguments
    tokens = effect.split()
    predicate = tokens[0]
    arguments = tokens[1:]

    # Construct the key to look up in observation_with_semantics
    key = f"{predicate}({', '.join(arguments)})"

    # Get the value from observation_with_semantics
    value = observation_with_semantics.get(key, None)

    # Compute the reward based on the value
    if isinstance(value, bool):
        reward = float(value)
    elif isinstance(value, (float, int)):
        reward = float(value)
    else:
        # Default reward if value is not found or invalid
        reward = 0.0

    # Invert the reward if the effect is negated
    if is_negated:
        reward = 1.0 - reward

    # Ensure the reward is between 0 and 1
    reward = max(0.0, min(1.0, reward))

    return reward
