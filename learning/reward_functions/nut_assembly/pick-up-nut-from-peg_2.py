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
    # Parse the grounded effect to determine which effect we're considering
    if grounded_effect == 'exclusively-occupying-gripper square-nut1 gripper1':
        # Calculate the distance between gripper1 and square-nut1
        gripper_to_square_nut_vec = observation_with_semantics['gripper1_to_square-nut1_dist']
        gripper_to_square_nut_dist = np.linalg.norm(gripper_to_square_nut_vec)

        # Get the maximum possible distance between gripper1 and any object
        gripper_to_obj_max_possible_vec = observation_with_semantics['gripper1_to_obj_max_possible_dist']
        gripper_to_obj_max_possible_dist = np.linalg.norm(gripper_to_obj_max_possible_vec)

        # Calculate progress towards picking up square-nut1
        # The closer the gripper is to the square nut, the higher the reward
        reward = 1 - (gripper_to_square_nut_dist / gripper_to_obj_max_possible_dist)
        reward = np.clip(reward, 0.0, 1.0)
        return float(reward)

    elif grounded_effect == 'not (on-peg square-nut1 round-peg1)':
        # Get the height difference between square-nut1 and the base of round-peg1
        height_difference = observation_with_semantics['square-nut1_bottom_height_above_round-peg1_base']

        # Get the height threshold required to be considered off the peg
        height_threshold = observation_with_semantics['height_threshold_required_to_be_on_or_off_round-peg1']

        # Calculate progress towards removing square-nut1 from round-peg1
        # The higher the square nut is above the peg base, the higher the reward
        reward = float(height_difference / height_threshold)
        reward = np.clip(reward, 0.0, 1.0)
        return reward

    else:
        # If the grounded effect is not recognized, return zero reward
        return 0.0
