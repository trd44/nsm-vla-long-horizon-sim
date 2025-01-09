# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, numpy.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicating percentage of completion towards the grounded effect
    '''
    # Parse the grounded effect
    grounded_effect = grounded_effect.strip()
    if grounded_effect.startswith('(not '):
        is_negative = True
        # Remove '(not ' and the outer ')'
        effect = grounded_effect[5:-1].strip()
    else:
        is_negative = False
        # Remove outer '()'
        effect = grounded_effect[1:-1]

    components = effect.strip().split()
    predicate = components[0]
    args = components[1:]

    if predicate == 'exclusively-occupying-gripper':
        # Progress towards gripper holding the object
        object_name = args[0]
        gripper_name = args[1]
        
        # Compute the distance between gripper and object
        dist_vector = observation_with_semantics[f'{gripper_name}_to_{object_name}_dist']
        distance = np.linalg.norm(dist_vector)
        
        # Get max possible distance to normalize
        max_dist_vector = observation_with_semantics[f'{gripper_name}_to_obj_max_possible_dist']
        max_distance = np.linalg.norm(max_dist_vector)
        
        # Calculate progress
        progress = 1 - (distance / max_distance)
        # Clip progress to [0,1]
        progress = np.clip(progress, 0, 1)
        
        return float(progress)
    
    elif predicate == 'on-peg':
        object_name = args[0]
        peg_name = args[1]
        
        # Access the heights
        object_bottom_height_above_peg_base = observation_with_semantics[f'{object_name}_bottom_height_above_{peg_name}_base']
        peg_height = observation_with_semantics[f'{peg_name}_height']

        # Calculate height difference
        height_difference = object_bottom_height_above_peg_base - peg_height
        
        # Get gripper and peg positions
        gripper_name = 'gripper1'  # Assuming gripper1
        gripper_z = observation_with_semantics[f'{gripper_name}_pos'][2]
        peg_z = observation_with_semantics[f'{peg_name}_pos'][2]
        
        # Maximum possible lift height
        max_lift_height = gripper_z - peg_z

        # Progress towards lifting the object off the peg
        progress = height_difference / max_lift_height
        # Clip progress to [0,1]
        progress = np.clip(progress, 0, 1)
        
        if is_negative:
            # If the effect is negated (not on-peg), progress is towards lifting off
            return float(progress)
        else:
            # If the effect is positive (on-peg), progress is reverse
            return float(1 - progress)
    else:
        # For other predicates, return 0
        return 0.0
