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
    # Compute the reward based on the grounded effect
    if grounded_effect == '(exclusively-occupying-gripper square-nut1 gripper1)':
        # Goal: gripper1 is holding square-nut1
        # We'll use the distance between gripper1 and square-nut1 to measure progress
        
        # Get the distance vector between gripper1 and square-nut1
        gripper_to_nut_dist_vec = observation_with_semantics['gripper1_to_square-nut1_dist']
        # Compute the Euclidean distance
        gripper_to_nut_dist = np.linalg.norm(gripper_to_nut_dist_vec)
        # Get the maximum possible distance for normalization
        max_possible_dist_vec = observation_with_semantics['gripper1_to_obj_max_possible_dist']
        max_possible_dist = np.linalg.norm(max_possible_dist_vec)
        # Calculate progress as inverse of normalized distance
        progress = 1 - (gripper_to_nut_dist / max_possible_dist)
        # Ensure the progress is between 0 and 1
        progress = np.clip(progress, 0.0, 1.0)
        return float(progress)
    
    elif grounded_effect == '(not (on-peg square-nut1 round-peg1))':
        # Goal: square-nut1 is no longer on round-peg1
        # We'll use the height of square-nut1 above the base of round-peg1
        
        # Get the current height of square-nut1 above round-peg1 base
        nut_height_above_peg = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base']
        # Get the threshold height required to be considered off the peg
        height_threshold = observation_with_semantics['height_threshold_required_to_be_on_or_off_round-peg1']
        # Calculate progress as the ratio of current height to required height
        progress = nut_height_above_peg / height_threshold
        # Ensure the progress is between 0 and 1
        progress = np.clip(progress, 0.0, 1.0)
        return float(progress)
    
    else:
        # If the grounded effect is unrecognized, return 0 progress
        return 0.0
