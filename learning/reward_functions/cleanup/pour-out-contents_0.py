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
    # Parse the grounded_effect string
    grounded_effect = grounded_effect.strip()
    is_negation = False

    if grounded_effect.startswith('(not '):
        is_negation = True
        # Remove '(not ' at the start and ')' at the end
        grounded_effect = grounded_effect[5:-1].strip()
    
    # Remove outer parentheses if present
    if grounded_effect.startswith('(') and grounded_effect.endswith(')'):
        grounded_effect = grounded_effect[1:-1].strip()
    
    tokens = grounded_effect.split()
    predicate = tokens[0]
    args = tokens[1:]

    # Handle different predicates
    if predicate == 'exclusively-occupying-gripper':
        # Get positions of gripper and object
        obj = args[0]
        gripper = args[1]
        gripper_pos = observation_with_semantics.get(f'{gripper}_pos', None)
        obj_pos = observation_with_semantics.get(f'{obj}_pos', None)
        max_possible_dist = np.linalg.norm(observation_with_semantics.get('gripper1_to_obj_max_possible_dist', np.array([1, 1, 1])))

        if gripper_pos is None or obj_pos is None:
            return 0.0  # Cannot compute reward without positions

        # Compute distance and normalize
        distance = np.linalg.norm(gripper_pos - obj_pos)
        progress = 1 - min(distance / max_possible_dist, 1.0)

        if is_negation:
            progress = 1 - progress
        return progress

    elif predicate == 'directly-on-table':
        # Get height of object's lowest point above table surface
        obj = args[0]
        table = args[1]
        height_key = f'height_of_{obj}_lowest_point_above_{table}_surface'
        height = observation_with_semantics.get(height_key, None)
        max_possible_height = observation_with_semantics.get('obj_max_possible_height_above_table1', None)

        if height is None or max_possible_height is None:
            return 0.0  # Cannot compute reward without height

        # Normalize height
        normalized_height = min(height / max_possible_height, 1.0)
        progress = 1 - normalized_height

        if is_negation:
            progress = normalized_height
        return progress

    elif predicate == 'inside':
        # Get percent overlap between objects
        obj1 = args[0]
        obj2 = args[1]
        overlap_key1 = f'percent_overlap_of_{obj1}_with_{obj2}'
        overlap_key2 = f'percent_overlap_of_{obj2}_with_{obj1}'
        percent_overlap = observation_with_semantics.get(overlap_key1, None)
        if percent_overlap is None:
            percent_overlap = observation_with_semantics.get(overlap_key2, None)
        
        if percent_overlap is None:
            progress = 0.0
        else:
            progress = float(percent_overlap)

        if is_negation:
            progress = 1 - progress
        return progress

    else:
        # For unhandled predicates, return 0.0
        return 0.0
