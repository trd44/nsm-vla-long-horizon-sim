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
    if grounded_effect == 'exclusively-occupying-gripper square-nut1 gripper1':
        # Calculate the Euclidean distance between gripper1 and square-nut1
        current_dist_vector = observation_with_semantics['gripper1_to_square-nut1_dist']
        current_dist = np.linalg.norm(current_dist_vector)
        # Get the max possible distance
        max_dist_vector = observation_with_semantics['gripper1_to_obj_max_possible_dist']
        max_dist = np.linalg.norm(max_dist_vector)
        # Compute progress
        progress = max(0.0, min(1.0, 1.0 - (current_dist / max_dist)))
        return progress
    elif grounded_effect == '(not (on-peg square-nut1 round-peg1))':
        # Get current height above the peg
        current_height = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base'].item()
        # Get threshold height
        threshold_height = observation_with_semantics['height_threshold_required_to_be_on_or_off_round-peg1'].item()
        # Compute progress
        progress = min(1.0, current_height / threshold_height)
        return progress
    else:
        return 0.0
