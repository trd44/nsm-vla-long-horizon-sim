# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicate percentage of completion towards the grounded effect
    '''
    import re

    # Parse the grounded effect
    effect = grounded_effect.strip()
    is_negation = False
    if effect.startswith('(not'):
        is_negation = True
        # Remove '(not ' and trailing ')'
        effect = effect[4:].strip()
        if effect.startswith('(') and effect.endswith(')'):
            effect = effect[1:-1].strip()
    else:
        if effect.startswith('(') and effect.endswith(')'):
            effect = effect[1:-1].strip()

    tokens = effect.split()
    predicate = tokens[0]
    args = tokens[1:]

    progress = 0.0

    if predicate == 'exclusively-occupying-gripper':
        # Check if the gripper is holding the object
        object_name = args[0]
        gripper_name = args[1]
        # Use distance between gripper and object
        dist_key = f'{gripper_name}_to_{object_name}_dist'
        if dist_key in observation_with_semantics:
            dist = np.linalg.norm(observation_with_semantics[dist_key])
            max_dist = np.linalg.norm(observation_with_semantics[f'{gripper_name}_to_obj_max_possible_dist'])
            progress = 1.0 - min(dist / max_dist, 1.0)
        else:
            progress = 0.0

    elif predicate == 'directly-on-table':
        # Check the height of the object above the table
        object_name = args[0]
        table_name = args[1]
        height_key = f'height_of_{object_name}_lowest_point_above_{table_name}_surface'
        if height_key in observation_with_semantics:
            height_above_table = observation_with_semantics[height_key]
            max_height = observation_with_semantics['obj_max_possible_height_above_table1']
            progress = 1.0 - min(height_above_table / max_height, 1.0)
        else:
            progress = 0.0

    elif predicate == 'inside':
        # Check the percent overlap of the object within another object
        inner_object_name = args[0]
        outer_object_name = args[1]
        overlap_key = f'percent_overlap_of_{inner_object_name}_with_{outer_object_name}'
        if overlap_key in observation_with_semantics:
            progress = float(observation_with_semantics[overlap_key])
        else:
            progress = 0.0

    else:
        # Unhandled predicate
        progress = 0.0

    # If the effect is negated, invert the progress
    if is_negation:
        progress = 1.0 - progress

    # Ensure progress is between 0 and 1
    progress = max(0.0, min(progress, 1.0))

    return progress
