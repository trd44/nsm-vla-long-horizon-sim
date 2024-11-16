def reward_shaping_fn(observation_with_semantics: Dict[str, Union[bool, float, np.array]], grounded_effect: str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicating percentage of completion towards the grounded effect
    '''
    if grounded_effect == 'exclusively-occupying-gripper square-nut1 gripper1':
        # Compute progress towards gripper1 holding square-nut1
        dist_vector = observation_with_semantics['gripper1_to_square-nut1_dist']
        dist_norm = np.linalg.norm(dist_vector)
        max_dist = 0.05  # Maximum distance to consider the nut grasped
        progress = max(0.0, min(1.0, (max_dist - dist_norm) / max_dist))
        return progress
    elif grounded_effect == 'not (free gripper1)':
        # Similar to above, gripper is not free when holding the nut
        dist_vector = observation_with_semantics['gripper1_to_square-nut1_dist']
        dist_norm = np.linalg.norm(dist_vector)
        max_dist = 0.05
        progress = max(0.0, min(1.0, (max_dist - dist_norm) / max_dist))
        return progress
    elif grounded_effect == 'not (on-peg square-nut1 round-peg1)':
        # Compute progress towards lifting square-nut1 off the peg
        pos_diff = observation_with_semantics['square-nut1_to_round-peg1_top']
        z_diff = pos_diff[2]  # Height difference
        initial_z = 0.01  # Initial height when nut is on the peg
        target_z = 0.05  # Height considered off the peg
        progress = max(0.0, min(1.0, (z_diff - initial_z) / (target_z - initial_z)))
        return progress
    else:
        return 0.0
