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
    import numpy as np

    if grounded_effect == '(exclusively-occupying-gripper mug1 gripper1)':
        # Calculate the Euclidean distance between gripper1 and mug1
        d = np.linalg.norm(observation_with_semantics['gripper1_to_mug1_dist'])
        max_d = np.linalg.norm(observation_with_semantics['gripper1_to_obj_max_possible_dist'])
        reward = 1.0 - (d / max_d)
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)
    elif grounded_effect == '(not (directly-on-table mug1 table1))':
        h = observation_with_semantics['height_of_mug1_lowest_point_above_table1_surface']
        max_h = observation_with_semantics['obj_max_possible_height_above_table1_surface']
        reward = h / max_h
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)
    elif grounded_effect == '(not (inside block1 mug1))':
        overlap = observation_with_semantics['percent_overlap_of_block1_with_mug1']
        reward = 1.0 - overlap
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)
    elif grounded_effect == '(directly-on-table block1 table1)':
        h = observation_with_semantics['height_of_block1_lowest_point_above_table1_surface']
        max_h = observation_with_semantics['obj_max_possible_height_above_table1_surface']
        reward = 1.0 - (h / max_h)
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)
    else:
        # If the grounded effect is not recognized, return 0
        return 0.0
