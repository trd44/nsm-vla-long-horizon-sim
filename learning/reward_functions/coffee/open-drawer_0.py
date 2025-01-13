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
    if grounded_effect == "(open drawer1)":
        # Calculate progress towards opening the drawer
        drawer1_travel_distance = float(observation_with_semantics['drawer1_travel_distance'])
        drawer1_cabinet_side_length = float(observation_with_semantics['drawer1_cabinet_side_length'])
        progress = drawer1_travel_distance / drawer1_cabinet_side_length
        progress = max(0.0, min(1.0, progress))  # Ensure progress is within [0,1]
        return progress
    elif grounded_effect == "(exclusively-occupying-gripper drawer1 gripper1)":
        # Calculate progress based on gripper's proximity to the drawer
        gripper1_to_drawer1_dist = np.array(observation_with_semantics['gripper1_to_drawer1_dist'])
        distance = np.linalg.norm(gripper1_to_drawer1_dist)
        gripper1_to_any_obj_max_absolute_dist = np.array(observation_with_semantics['gripper1_to_any_obj_max_absolute_dist'])
        max_distance = np.linalg.norm(gripper1_to_any_obj_max_absolute_dist)
        progress = 1.0 - (distance / max_distance)
        progress = max(0.0, min(1.0, progress))  # Ensure progress is within [0,1]
        return progress
    else:
        # If the grounded effect is not recognized, return 0 progress
        return 0.0
