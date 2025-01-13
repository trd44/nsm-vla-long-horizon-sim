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
    if grounded_effect.strip() == '(exclusively-occupying-gripper drawer1 gripper1)':
        # Compute the Euclidean distance between gripper1 and drawer1
        gripper_to_drawer_dist = np.linalg.norm(observation_with_semantics['gripper1_to_drawer1_dist'])
        # Get the maximum possible distance between gripper1 and any object
        max_distance = np.linalg.norm(observation_with_semantics['gripper1_to_any_obj_max_absolute_dist'])
        # Calculate progress as the inverse of the normalized distance
        progress = max(0.0, min(1.0, 1.0 - (gripper_to_drawer_dist / max_distance)))
        return progress
    elif grounded_effect.strip() == '(open drawer1)':
        # Calculate the displacement of drawer1 along its moving axis (assuming x-axis)
        # Since we don't have the initial position, we'll assume it starts at x = 0
        drawer_current_pos = observation_with_semantics['drawer1_pos'][0]  # x-coordinate
        drawer_travel_distance = observation_with_semantics['drawer1_travel_distance']
        # Progress is displacement divided by maximum travel distance
        progress = max(0.0, min(1.0, drawer_current_pos / drawer_travel_distance))
        return progress
    else:
        # Unrecognized grounded effect
        return 0.0
