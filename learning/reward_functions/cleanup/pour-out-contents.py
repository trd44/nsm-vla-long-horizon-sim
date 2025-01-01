# llm generated reward shaping function
from typing import *
import numpy as np

def parse_grounded_effect(grounded_effect: str):
    # Remove surrounding spaces
    grounded_effect = grounded_effect.strip()
    is_negated = False
    if grounded_effect.startswith('(not '):
        is_negated = True
        # Remove the '(not ' prefix and remove the closing ')'
        predicate_string = grounded_effect[5:-1].strip()
    else:
        # Remove the '(' and ')' at the start and end.
        predicate_string = grounded_effect[1:-1].strip()
    # Now extract the predicate and arguments
    tokens = predicate_string.split()
    predicate = tokens[0]
    args = tokens[1:]
    return predicate, args, is_negated

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicate percentage of completion towards the grounded effect
    '''
    predicate, args, is_negated = parse_grounded_effect(grounded_effect)
    
    # Initialize progress to 0
    progress = 0.0
    
    if predicate == 'exclusively-occupying-gripper':
        # args: [object_name, gripper_name]
        obj_name, gripper_name = args
        # Compute distance between gripper and object
        gripper_to_obj_dist_key = f'{gripper_name}_to_{obj_name}_dist'
        gripper_to_obj_max_absolute_dist = observation_with_semantics['gripper1_to_obj_max_absolute_dist']
        max_distance = np.linalg.norm(gripper_to_obj_max_absolute_dist)
        if gripper_to_obj_dist_key in observation_with_semantics:
            gripper_to_obj_dist_vector = observation_with_semantics[gripper_to_obj_dist_key]
            distance = np.linalg.norm(gripper_to_obj_dist_vector)
            progress = max(0.0, 1.0 - (distance / max_distance))
        else:
            # If key not found, cannot compute progress
            progress = 0.0
            
        # If negative literal, progress is inverted
        if is_negated:
            progress = 1.0 - progress
    
    elif predicate == 'directly-on-table':
        # args: [object_name, table_name]
        obj_name, table_name = args
        height_of_obj_key = f'height_of_{obj_name}_lowest_point_above_{table_name}_surface'
        max_height_key = f'max_height_above_{table_name}'
        if height_of_obj_key in observation_with_semantics and max_height_key in observation_with_semantics:
            height = observation_with_semantics[height_of_obj_key]
            max_height = observation_with_semantics[max_height_key]
            progress = max(0.0, 1.0 - (height / max_height))
        else:
            # Cannot compute progress without height info
            progress = 0.0
        # If negated literal, invert progress
        if is_negated:
            progress = 1.0 - progress
    
    elif predicate == 'inside':
        # args: [object1, object2]
        obj1_name, obj2_name = args
        # Look for percent_overlap_of_obj1_with_obj2
        overlap_key = f'percent_overlap_of_{obj1_name}_with_{obj2_name}'
        if overlap_key in observation_with_semantics:
            overlap = observation_with_semantics[overlap_key]
            # Assuming overlap is between 0 and 1
            progress = overlap
        else:
            # If no overlap data, progress is 0
            progress = 0.0
        
        # For 'not (inside ...)', we need progress towards not being inside,
        # So progress should be higher when overlap is lower.
        if is_negated:
            progress = 1.0 - progress
    else:
        # If predicate not recognized, return progress 0
        progress = 0.0
    
    # Ensure progress is between 0 and 1
    progress = np.clip(progress, 0.0, 1.0)

    return progress
