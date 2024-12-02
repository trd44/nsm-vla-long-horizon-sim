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
    if grounded_effect == '(open drawer1)':
        # Calculate progress towards opening the drawer
        travel_distance = observation_with_semantics['drawer1_travel_distance']
        side_length = observation_with_semantics['drawer1_cabinet_side_length']
        progress = float(travel_distance / side_length)
        progress = min(max(progress, 0.0), 1.0)
        return progress
    elif grounded_effect == '(exclusively-occupying-gripper drawer1 gripper1)':
        # Calculate progress towards gripper exclusively occupying drawer
        distance_vector = observation_with_semantics['gripper1_to_drawer1_dist']
        distance = np.linalg.norm(distance_vector)
        max_distance_vector = observation_with_semantics['gripper1_to_any_obj_max_absolute_dist']
        max_distance = np.linalg.norm(max_distance_vector)
        progress = max(0.0, 1.0 - (distance / max_distance))
        progress = min(max(progress, 0.0), 1.0)
        return progress
    else:
        # Unknown grounded effect
        return 0.0
