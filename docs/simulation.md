action
```
index 0: [-1, 1]: x backward - forward
index 1: [-1, 1]: y left - right (human perspective)
index 2: [-1, 1]: z lower - higher
index 3: [-1, 1]: yaw left - right (human perspective)
index 4: [-1, 1]: pitch forward - backward
index 5: [-1, 1]: roll clockwise - counter (human)
index 6: [-1, 1]: open - close
```
cleanup observation
```
OrderedDict with keys
odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'mug_pos', 'mug_quat', 'mug_to_robot0_eef_pos', 'mug_to_robot0_eef_quat', 'drawer_pos', 'drawer_quat', 'drawer_to_robot0_eef_pos', 'drawer_to_robot0_eef_quat', 'cube_pos', 'cube_quat', 'cube_to_robot0_eef_pos', 'cube_to_robot0_eef_quat', 'drawer_joint_pos', 'robot0_proprio-state', 'object-state'])
```
coffee observation
```
```

nut—assembly
```
OrderedDict with keys
odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'gripper1_pos', 'gripper1_quat', 'square-nut1_pos', 'square-nut1_quat', 'round-nut1_pos', 'round-nut1_quat', 'gripper1_to_square-nut1_dist', 'gripper1_to_square-nut1_quat', 'gripper1_to_round-nut1_dist', 'gripper1_to_round-nut1_quat', 'square-peg1_pos', 'square-peg1_quat', 'round-peg1_pos', 'round-peg1_quat', 'gripper1_to_obj_max_absolute_dist', 'gripper1_to_square-peg1_dist', 'gripper1_to_square-peg1_quat', 'gripper1_to_round-peg1_dist', 'gripper1_to_round-peg1_quat', 'square-peg1_height', 'round-peg1_height', 'square-nut1_bottom_height_above_square-peg1_base', 'square-nut1_bottom_height_above_round-peg1_base', 'round-nut1_bottom_height_above_square-peg1_base', 'round-nut1_bottom_height_above_round-peg1_base', 'robot0_proprio-state', 'object-state'])
```