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
    # Helper function to parse the grounded effect
    def parse_grounded_effect(effect_str):
        effect_str = effect_str.strip()
        if effect_str.startswith('not ('):
            negated = True
            inner_effect = effect_str[4:-1].strip()
        else:
            negated = False
            inner_effect = effect_str
        tokens = inner_effect.strip('()').split()
        predicate = tokens[0]
        args = tokens[1:]
        return predicate, args, negated

    # Parse the grounded effect
    predicate, args, negated = parse_grounded_effect(grounded_effect)

    if predicate == 'exclusively-occupying-gripper':
        obj = args[0]
        gripper = args[1]
        # Compute progress based on distance between gripper and object
        dist_vector = observation_with_semantics[f'{gripper}_to_{obj}_dist']
        current_distance = np.linalg.norm(dist_vector)
        max_distance_vector = observation_with_semantics[f'{gripper}_to_obj_max_absolute_dist']
        max_distance = np.linalg.norm(max_distance_vector)
        progress = 1.0 - (current_distance / max_distance)
        progress = np.clip(progress, 0.0, 1.0)
        if negated:
            progress = 1.0 - progress
        return float(progress)

    elif predicate == 'free':
        gripper = args[0]
        # Compute progress based on distances to all objects
        distances = []
        for key in observation_with_semantics.keys():
            if key.startswith(f'{gripper}_to_') and key.endswith('_dist') and key != f'{gripper}_to_obj_max_absolute_dist':
                dist_vector = observation_with_semantics[key]
                current_distance = np.linalg.norm(dist_vector)
                distances.append(current_distance)
        if distances:
            min_distance = min(distances)
            max_distance_vector = observation_with_semantics[f'{gripper}_to_obj_max_absolute_dist']
            max_distance = np.linalg.norm(max_distance_vector)
            progress = min_distance / max_distance
            progress = np.clip(progress, 0.0, 1.0)
            if negated:
                progress = 1.0 - progress
            return float(progress)
        else:
            return 0.0  # Unable to compute progress

    elif predicate == 'on-peg':
        nut = args[0]
        peg = args[1]
        # Compute progress based on vertical distance between nut and peg
        vector_key = f'{nut}_bottom_to_{peg}_top'
        if vector_key in observation_with_semantics:
            dist_vector = observation_with_semantics[vector_key]
            vertical_distance = abs(dist_vector[2])  # Z-component
            max_vertical_key = f'{nut}_to_{peg}_top_max_vertical_dist'
            max_vertical_distance = observation_with_semantics[max_vertical_key]
            progress = vertical_distance / max_vertical_distance
            progress = np.clip(progress, 0.0, 1.0)
            if not negated:
                progress = 1.0 - progress  # Progress increases as vertical distance decreases
            return float(progress)
        else:
            return 0.0  # Unable to compute progress

    else:
        # Unknown predicate
        return 0.0
