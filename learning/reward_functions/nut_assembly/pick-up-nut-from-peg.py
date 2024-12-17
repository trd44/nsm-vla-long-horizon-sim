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
    # Extract relevant information
    if grounded_effect == 'exclusively-occupying-gripper square-nut1 gripper1':
        gripper1_pos = observation_with_semantics['gripper1_pos']
        square_nut1_pos = observation_with_semantics['square-nut1_pos']
        gripper1_to_square_nut1_dist_vector = observation_with_semantics['gripper1_to_square-nut1_dist']
        gripper1_to_square_nut1_dist = np.linalg.norm(gripper1_to_square_nut1_dist_vector)
        max_possible_distance = np.linalg.norm(observation_with_semantics['gripper1_to_obj_max_possible_dist'])
        progress_grasp = 1 - (gripper1_to_square_nut1_dist / max_possible_distance)
        progress_grasp = np.clip(progress_grasp, 0, 1)
        return progress_grasp
    
    elif grounded_effect == 'not (on-peg square-nut1 round-peg1)':
        square_nut1_bottom_height_above_round_peg1_base = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base']
        max_possible_height = observation_with_semantics['gripper1_to_obj_max_possible_dist'][2]  # The maximum z-distance
        progress_lift = square_nut1_bottom_height_above_round_peg1_base / max_possible_height
        progress_lift = np.clip(progress_lift, 0, 1)
        return progress_lift

