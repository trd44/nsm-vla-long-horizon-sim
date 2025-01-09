# llm generated reward shaping function
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicating percentage of completion towards the grounded effect
    '''
    if grounded_effect == '(exclusively-occupying-gripper square-nut1 gripper1)':
        # Compute distance between gripper1 and square-nut1
        gripper1_to_square_nut1_dist = observation_with_semantics['gripper1_to_square-nut1_dist']
        distance = np.linalg.norm(gripper1_to_square_nut1_dist)
        # Compute maximum possible distance
        max_possible_dist_vector = observation_with_semantics['gripper1_to_obj_max_possible_dist']
        max_distance = np.linalg.norm(max_possible_dist_vector)
        # Compute progress towards gripping the nut
        progress = 1 - (distance / max_distance)
        progress = np.clip(progress, 0.0, 1.0)
        return float(progress)
    
    elif grounded_effect == '(not (on-peg square-nut1 round-peg1))':
        # Get the height of the square-nut above the round-peg base
        square_nut1_height = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base']
        round_peg1_height = observation_with_semantics['round-peg1_height']
        # Maximum possible height difference
        max_height = observation_with_semantics['gripper1_to_obj_max_possible_dist'][2]
        # Compute progress towards lifting the nut off the peg
        if square_nut1_height >= round_peg1_height:
            progress = (square_nut1_height - round_peg1_height) / (max_height - round_peg1_height)
            progress = np.clip(progress, 0.0, 1.0)
            return float(progress)
        else:
            return 0.0
    else:
        # If the grounded effect is not recognized, return zero progress
        return 0.0
