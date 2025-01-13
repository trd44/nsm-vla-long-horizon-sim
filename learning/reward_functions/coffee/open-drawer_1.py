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
    if grounded_effect == '(exclusively-occupying-gripper drawer1 gripper1)':
        # Compute the Euclidean distance between gripper1 and drawer1
        gripper_to_drawer_dist = observation_with_semantics['gripper1_to_drawer1_dist']
        distance = np.linalg.norm(gripper_to_drawer_dist)
        
        # Compute the maximum possible distance using 'gripper1_to_any_obj_max_absolute_dist'
        max_dist_vector = observation_with_semantics['gripper1_to_any_obj_max_absolute_dist']
        max_distance = np.linalg.norm(max_dist_vector)
        
        # Calculate reward as 1 minus the normalized distance
        reward = 1.0 - (distance / max_distance)
        
        # Ensure the reward is between 0 and 1
        reward = np.clip(reward, 0.0, 1.0)
        return reward

    elif grounded_effect == '(open drawer1)':
        # Assume the drawer opens along the Y-axis; calculate displacement from initial position
        drawer_pos = observation_with_semantics['drawer1_pos']
        initial_drawer_y = -0.35  # Assuming this is the closed position from 'drawer1_pos'
        drawer_y = drawer_pos[1]  # Y position of the drawer
        
        # Use 'drawer1_travel_distance' to normalize the displacement
        max_travel = observation_with_semantics['drawer1_travel_distance']
        displacement = initial_drawer_y - drawer_y  # Since movement is along negative Y-axis
        
        # Calculate reward as the fraction of the maximum travel distance
        reward = displacement / max_travel
        
        # Ensure the reward is between 0 and 1
        reward = np.clip(reward, 0.0, 1.0)
        return reward

    else:
        # Unknown grounded effect
        return 0.0
