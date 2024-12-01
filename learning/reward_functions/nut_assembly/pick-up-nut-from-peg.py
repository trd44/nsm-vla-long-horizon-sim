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
        # Compute distance between gripper and square-nut1
        gripper_pos = observation_with_semantics['gripper1_pos']
        nut_pos = observation_with_semantics['square-nut1_pos']
        distance = np.linalg.norm(gripper_pos - nut_pos)
        # Compute maximum possible distance
        max_distance = np.linalg.norm(observation_with_semantics['gripper1_to_obj_max_absolute_dist'])
        # Calculate progress
        progress = 1 - (distance / max_distance)
        # Clip progress to [0,1]
        progress = np.clip(progress, 0.0, 1.0)
        return float(progress)
    elif grounded_effect == 'not (on-peg square-nut1 round-peg1)':
        # Compute height above round-peg1 base
        height_above_base = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base']
        # Get the height of the round-peg1
        peg_height = observation_with_semantics['round-peg1_height']
        # Calculate progress
        progress = height_above_base / peg_height
        # Clip progress to [0,1]
        progress = np.clip(progress, 0.0, 1.0)
        return float(progress)
    else:
        # If the grounded effect is not recognized, return 0 progress
        return 0.0
