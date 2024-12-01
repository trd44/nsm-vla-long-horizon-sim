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
    if grounded_effect == 'directly-on-table block1 table1':
        height = observation_with_semantics['height_of_block1_lowest_point_above_table1_surface']
        height_threshold = 0.05  # Threshold for considering the block directly on the table
        progress = max(0.0, min(1.0, (height_threshold - height) / height_threshold))
        return progress

    elif grounded_effect == 'exclusively-occupying-gripper mug1 gripper1':
        distance = np.linalg.norm(observation_with_semantics['gripper1_to_mug1_dist'])
        max_possible_distance = 1.5  # Maximum possible distance in the environment
        progress = max(0.0, min(1.0, 1 - distance / max_possible_distance))
        return progress

    elif grounded_effect == 'not (directly-on-table mug1 table1)':
        height = observation_with_semantics['height_of_mug1_lowest_point_above_table1_surface']
        height_threshold = 0.05  # Threshold height to consider mug1 lifted from the table
        progress = max(0.0, min(1.0, height / height_threshold))
        return progress

    elif grounded_effect == 'not (inside block1 mug1)':
        percent_overlap = observation_with_semantics.get('percent_overlap_of_mug1_bounding_box_with_block1_bounding_box', 0.0)
        progress = max(0.0, min(1.0, 1 - percent_overlap / 1.0))
        return progress

    else:
        return 0.0
