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
    # Parse the grounded effect
    effect = grounded_effect.strip().strip('()')
    negated = False
    if effect.startswith('not '):
        negated = True
        inner_effect = effect[len('not '):].strip()
        inner_effect = inner_effect.strip('()').strip()
    else:
        inner_effect = effect.strip()
    tokens = inner_effect.split()
    predicate = tokens[0]
    args = tokens[1:]
    
    # Initialize reward
    reward = 0.0

    # Handle the predicates
    if predicate == 'exclusively-occupying-gripper':
        # Predicate: exclusively-occupying-gripper object gripper
        object_name = args[0]
        gripper_name = args[1]
        # Compute distance between gripper and object
        dist_key = f'{gripper_name}_to_{object_name}_dist'
        if dist_key in observation_with_semantics:
            dist_vector = observation_with_semantics[dist_key]
            dist = np.linalg.norm(dist_vector)
            # Assuming max possible distance is available
            max_dist_key = f'{gripper_name}_to_obj_max_possible_dist'
            if max_dist_key in observation_with_semantics:
                max_dist_vector = observation_with_semantics[max_dist_key]
                max_dist = np.linalg.norm(max_dist_vector)
                # Compute progress as (1 - (current_distance / max_distance))
                progress = 1 - min(dist / max_dist, 1.0)
                # If the object is in the gripper, progress = 1
                reward = progress
    elif predicate == 'on-peg':
        # Predicate: on-peg object peg
        object_name = args[0]
        peg_name = args[1]
        # Get the height of the object above the peg base
        height_key = f'{object_name}_bottom_height_above_{peg_name}_base'
        if height_key in observation_with_semantics:
            height = observation_with_semantics[height_key]
            # Get the gripper height above the peg base for max possible height
            gripper_name = 'gripper1'  # assuming gripper1
            gripper_pos_key = f'{gripper_name}_pos'
            peg_pos_key = f'{peg_name}_pos'
            if gripper_pos_key in observation_with_semantics and peg_pos_key in observation_with_semantics:
                gripper_pos = observation_with_semantics[gripper_pos_key]
                peg_pos = observation_with_semantics[peg_pos_key]
                max_height = gripper_pos[2] - peg_pos[2]  # Z-axis height difference
                # Compute progress as current height over max height
                progress = min(height / max_height, 1.0) if max_height > 0 else 0.0
                if negated:
                    # Since the goal is not on-peg, higher height means more progress
                    reward = progress
                else:
                    # For on-peg, lower height means more progress
                    reward = 1 - progress
    else:
        # Handle other predicates if necessary
        reward = 0.0

    # Ensure reward is between 0 and 1
    reward = max(0.0, min(reward, 1.0))
    return reward
