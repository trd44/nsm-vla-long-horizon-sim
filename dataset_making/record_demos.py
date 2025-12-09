# record_demos.py

import os
import copy
import math
import gym
import cv2
import numpy as np
# Commenting bc its breaking pi inference
# from openpi.src.openpi.planning.hanoi_vlm_planner import query_model 

from dataset_making.args import Args
from dataset_making.utils import to_datestring
from dataset_making.tasks import (
    PickOperation, PlaceOperation, TurnOnOperation, TurnOffOperation
)
from planning.planner import (
    add_predicates_to_pddl, call_planner, define_goal_in_pddl
)

# Define which predicates to include per domain and planner mode
planning_predicates = {
    "Hanoi": ['on', 'clear', 'grasped', 'smaller'],
    "Hanoi4x3": ['on', 'clear', 'grasped', 'smaller'],
    "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
    "NutAssembly": ['on', 'clear', 'grasped'],
    "CubeSorting": ['on', 'clear', 'grasped', 'small'],
    "HeightStacking": ['on', 'clear', 'grasped', 'smaller'],
    "AssemblyLineSorting": ['on', 'clear', 'grasped', 'type_match'],
    "PatternReplication": ['on', 'clear', 'grasped'],
}
planning_mode = {
    "Hanoi": 0,
    "Hanoi4x3": 0,
    "KitchenEnv": 1,
    "NutAssembly": 0,
    "CubeSorting": 0,
    "HeightStacking": 0,
    "AssemblyLineSorting": 0,
    "PatternReplication": 1,
}

class RecordDemos(gym.Wrapper):
    """Gym wrapper to record demonstrations by symbolically planning 
    and executing.
    
    Args:
        args: Args object containing the arguments for the dataset making script
        env: gym.Env object representing the environment
        detector: Detector object representing the detector
        pddl_path: str path to the PDDL files
        randomize: bool whether to randomize the actions
    """
    def __init__(
        self,
        args: Args,
        env,
        detector,
        pddl_path: str,
        randomize: bool=True,
    ):
        super().__init__(env)
        self.args = args
        self.env = env
        self.detector = detector
        self.pddl_path = pddl_path
        self.randomize = randomize

        # Attributes set from args
        self.total_episodes = args.episodes
        self.verbose = args.verbose
        
        # Randomization / noise controls        
        self.noise_std = args.noise_std
        # Fraction of episodes that should be noisy 
        # (deterministic schedule: last X%)
        self.noisy_fraction = args.noisy_fraction
        self.schedule_override = False       # Deterministic scheduling state
        self.randomize_this_episode = False  # Computed per-episode in reset()
        self.scheduled_ep_index = -1         # Not sure why necessary
        self.reset_count = 0
        
        # Buffer for recording
        self.sequential_episode_buffer = []
        self.successes = 0

        # Generated in reset()
        self.plan = None          # Generated during reset()
        self.natural_plan = None  # Generated during reset()
        self.non_noisy_eps = self.set_non_noisy_eps()

        # TODO: Remove? Unnecessary AI slop?
        # # Ensure trailing slash for pddl path so planner joins files correctly
        # if not self.pddl_path.endswith(os.sep):
        #     self.pddl_path += os.sep

    def set_non_noisy_eps(self) -> int:
        """
        Determine the number of episodes that should be non-noisy 
        deteministically based on the number of episodes recorded when 
        randomize is True.
        """
        # If not randomizing, return the total number of episodes.
        if not self.randomize:
            return self.total_episodes
        
        non_noisy_frac = 1.0 - self.noisy_fraction
        non_noisy_eps = math.floor(non_noisy_frac * self.total_episodes)
        return non_noisy_eps

    def reset(self, successes: int) -> dict:
        """Reset environment and clear buffers."""
        self.successes = successes
        self.reset_count += 1
        
        # Clear any previous trajectory data
        self.sequential_episode_buffer = []

        # Check if this episode should be noisy
        self.randomize_this_episode = (self.successes >= self.non_noisy_eps)
        if self.verbose and self.randomize_this_episode:
            print(f"[RecordDemos] Episode {self.successes} is noisy.")
        elif self.verbose and not self.randomize_this_episode:
            print(f"[RecordDemos] Episode {self.successes} is non-noisy.")
        
        raw = self.env.reset()
        
        # Extract observation if reset returns tuple or list
        if isinstance(raw, (tuple, list)):
            obs = raw[0]
        else:
            obs = raw
        
        # Ensure obs is a dict
        if not isinstance(obs, dict):
            obs = self.env._get_observations()

        # Forward the simulator state
        self.env.sim.forward()
        
        # Let the simulation settle for 50 timesteps
        action_space = self.env.action_space  # Determine action shape
        neutral_action = np.zeros(action_space.shape, dtype=action_space.dtype)
        for _ in range(50):
            obs, *_ = self.env.step(neutral_action)

        # Print body positions for debugging
        if self.verbose:
            print("After env.reset(), block positions:")
            for name in ["cube1_main", "cube2_main", "cube3_main"]:
                body_id = self.env.sim.model.body_name2id(name)
                print(f"{name}: {self.env.sim.data.body_xpos[body_id]}")

        # Detect init state
        state = self.detector.get_groundings(
            as_dict=True, binary_to_float=False, return_distance=False
        )
        if self.verbose:
            print("Detector groundings:", state)
        
        # Include only TRUE predicates that are in planning_predicates
        init_predicates = {}
        for predicate, value in state.items():
            pp = planning_predicates[self.args.env]
            if value and predicate.split('(')[0] in pp:
                init_predicates[predicate] = True

        # After add_predicates_to_pddl:
        # Pass detected objects to dynamically generate PDDL
        # Filter predicates to only include those that involve active cubes 
        #   (for Hanoi-like environments)
        # Get detector attributes safely, as not all detectors have them
        detector_objects = getattr(self.detector, 'objects', [])
        object_areas = getattr(self.detector, 'object_areas', [])
        
        # Get environment name early for filtering logic
        env_name = self.args.env
        
        active_cubes = []
        if detector_objects:
            for cube in detector_objects:
                # Check if cube is part of tower by looking for 'on' predicates
                is_active = False
                for pred, value in init_predicates.items():
                    if pred.startswith('on(') and value:
                        parts = pred.split('(')[1].split(',')
                        obj1 = parts[0]
                        obj2 = parts[1].rstrip(')')
                        if obj1 == cube or obj2 == cube:
                            is_active = True
                            break
                if is_active:
                    active_cubes.append(cube)
        
        # For PatternReplication: 
        # remove all reference cube predicates and table clear predicates
        # - ref_cubes are only used for goal generation
        # - table should not be a placement target
        if env_name == "PatternReplication":
            filtered_predicates = {}
            for predicate in init_predicates.keys():
                # Skip ref_cube predicates 
                if "ref_cube" in predicate:
                    continue
                # Clear predicates involving table
                if predicate.startswith("clear(") and "table" in predicate:
                    continue
                filtered_predicates[predicate] = True
        
        # Filter predicates to only include those involving active cubes or pegs
        # (This is primarily for Hanoi-like environments)
        elif active_cubes or object_areas:
            filtered_predicates = {}
            # Apply filtering if we have objects to filter on
            for predicate, value in init_predicates.items():
                if value:
                    # Only include smaller predicates if both objects are active
                    if predicate.startswith('smaller('):
                        parts = predicate.split('(')[1].split(',')
                        obj1 = parts[0]
                        obj2 = parts[1].rstrip(')')
                        if (obj1 in active_cubes or obj1 in object_areas) and \
                           (obj2 in active_cubes or obj2 in object_areas):
                            filtered_predicates[predicate] = value
                    else:
                        # Check if predicate involves any active cubes
                        predicate_involves_active_cube = False
                        for cube in active_cubes:
                            if cube in predicate:
                                predicate_involves_active_cube = True
                                break
                        
                        # Check if predicate involves any pegs
                        if not predicate_involves_active_cube and object_areas:
                            for peg in object_areas:
                                if peg in predicate:
                                    predicate_involves_active_cube = True
                                    break
                        
                        if predicate_involves_active_cube:
                            filtered_predicates[predicate] = value
        else:
            # No filtering needed - use all True predicates
            filtered_predicates = {k: v for k, v in init_predicates.items() if v}
        
        # Detected objects are embedded in the filtered predicates
        # The PDDL files should define objects in their problem files
        
        add_predicates_to_pddl(self.pddl_path, filtered_predicates)
        
        # Generate environment-specific goals for certain environments
        envs_with_goals = [
            "CubeSorting", 
            "HeightStacking", 
            "AssemblyLineSorting", 
            "PatternReplication"
        ]
        if env_name in envs_with_goals:
            goal_predicates = _generate_env_specific_goal(
                env_name, state, self.detector, self.pddl_path
            )
            if self.verbose:
                print(f"Generated goal predicates for {env_name}: \
                    {goal_predicates}")
        
        # Generate a fresh plan for this episode based on current state
        # check 'planner' in self.args.planner
        if self.args.planner == 'pddl':
            self.plan, _ = call_planner(
                self.pddl_path, 
                problem="problem_dummy.pddl", 
                mode=planning_mode[env_name]
            )
        else:  # VLM Planner
            # TODO: this obs does not have agentview_image. 
            # Figure out how to get the obs with agentview_image
            # I don't know if I wrote this or an accidental tab complete.
            init_image = obs.get('agentview_image') 
            cv2.imshow("init_image", init_image)
            #TODO: save the goal image in this path
            # I don't know if I wrote this or an accidental tab complete.
            goal_image = self.args.goal_image_path 
            self.plan = query_model(
                init_image, goal_image, model=self.args.planner
            )
        
        if self.verbose:
            print(f"Plan: {self.plan}")
        
        if not self.plan:
            print("There is no plan, resetting")
            return 
        
        # Convert PDDL plan to natural language
        self.natural_plan = self._convert_plan_to_natural_language(self.plan)
        
        return obs

    def _convert_plan_to_natural_language(self, plan):
        """Convert PDDL plan to natural language commands."""
        natural_commands = []
        print("Natural language plan:")
        for i, op_str in enumerate(plan):
            nl_instruction = symbolic_to_natural_instruction(op_str, self.env)
            natural_commands.append(nl_instruction)
            print(f"  {i+1}. {nl_instruction}")
        print()
        return natural_commands

    def record_step(self, obs_dict: dict, action: np.ndarray):
        """Store a single step's data."""
        # Calculate actual gripper width from finger positions in sim
        try:
            left_finger_pos = self.env.sim.data.body_xpos[
                self.env.sim.model.body_name2id("gripper0_left_inner_finger")
            ]
            right_finger_pos = self.env.sim.data.body_xpos[
                self.env.sim.model.body_name2id("gripper0_right_inner_finger")
            ]
            gripper_width = float(np.linalg.norm(left_finger_pos - right_finger_pos))
        except:
            # Fallback: use first joint position as proxy
            gripper_width = float(obs_dict.get('robot0_gripper_qpos', [0.0])[0])
        
        self.sequential_episode_buffer.append({
            'obs_dict': copy.deepcopy(obs_dict),
            'action': np.array(action, dtype=np.float32),
            'language_instruction': str(self.current_instruction),
            'gripper_width': gripper_width
        })

    def _map_operator(self, op_str: str):
        """Map a PDDL operator string to a TaskOperation subclass and kwargs."""
        parts = op_str.lower().split()
        cmd = parts[0]
        if cmd == 'pick':
            return PickOperation, {
                'object_id': parts[1],
            }
        if cmd == 'place':
            return PlaceOperation, {
                'object_id': parts[1],
                'placement_id': parts[2],
            }
        if cmd == 'turn-on':
            return TurnOnOperation, {
                'object_id': parts[1],
            }
        if cmd == 'turn-off':
            return TurnOffOperation, {
                'object_id': parts[1],
            }
        raise ValueError(f"Unsupported operator: {op_str}")

    def run_trajectory(self, obs: dict) -> bool:
        """Execute the full symbolic plan, recording each step."""
        # Iterate over the plan and execute each operation
        for i, op_str in enumerate(self.plan):  
            OpClass, params = self._map_operator(op_str)  # Map to operator
            op = OpClass(
                self.args,
                self.env,
                self.detector,
                self.randomize_this_episode,
                self.noise_std,
                **params
            )
            
            self.current_instruction = self.natural_plan[i]
            
            success, obs = op.execute(obs)  # Execute the operation
            
            if not success:
                print(f"Failed to perform task: {op_str}")
                return False
        
        return True

    def save_trajectory(self, episode_idx: int) -> bool:
        """Format and save the recorded trajectory in RLDS .npy format."""
        from dataset_making.utils import quaternion_to_euler  # Ensure this is imported

        steps = []
        buffer = self.sequential_episode_buffer
        num_steps = len(buffer)

        if not buffer:
            print(f"No data to save for episode {episode_idx}.")
            return False

        for i, step_data in enumerate(buffer):
            obs = step_data['obs_dict']
            action = step_data['action']
            instr = step_data['language_instruction']
            gripper_width = step_data.get('gripper_width', 0.0)

            # MAIN IMAGE
            agent_img = obs.get('agentview_image')
            if agent_img is None:
                print("Agent image is none")
                agent_img = np.zeros((256,256,3), dtype=np.uint8)
            else:
                agent_img = np.ascontiguousarray(agent_img[::-1, ::-1], dtype=np.uint8)
                # agent_img = np.asarray(agent_img, dtype=np.uint8)

            # WRIST IMAGE
            wrist_img = obs.get('robot0_eye_in_hand_image')
            if wrist_img is None:
                print("Wrist image is none")
                wrist_img = np.zeros((256,256,3), dtype=np.uint8)
            else:
                # wrist_img = np.asarray(wrist_img, dtype=np.uint8)
                wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1], dtype=np.uint8)

            # JOINT STATE
            # Reconstruct joint positions from cos/sin if needed
            if 'robot0_joint_pos' in obs:
                joint_state = np.asarray(obs['robot0_joint_pos'], dtype=np.float32)
            elif 'robot0_joint_pos_cos' in obs and 'robot0_joint_pos_sin' in obs:
                joint_cos = obs['robot0_joint_pos_cos']
                joint_sin = obs['robot0_joint_pos_sin']
                joint_state = np.arctan2(joint_sin, joint_cos).astype(np.float32)
            else:
                raise KeyError("No joint position keys found in observation!")

            # STATE (END EFFECTOR)
            eef_pos = obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32))
            eef_quat = obs.get('robot0_eef_quat', np.array([0., 0., 0., 1.], dtype=np.float32))
            
            # Use the gripper width calculated from actual finger positions in sim
            eef_gripper = np.array([gripper_width], dtype=np.float32)
            
            # Convert quaternion to axis angle
            eef_axis_angle = _quat2axisangle(eef_quat)

            eef_state = np.concatenate((eef_pos, eef_axis_angle, eef_gripper, np.array([0.0], dtype=np.float32))).astype(np.float32)

            # State is the concatenation of joint state and gripper opening
            joint_state = np.concatenate((joint_state, eef_gripper)).astype(np.float32)

            # RLDS step dict
            step_dict = {
                "action": action.astype(np.float32),
                "is_terminal": (i == num_steps - 1),
                "is_last": (i == num_steps - 1),
                "is_first": (i == 0),
                "reward": 1.0 if (i == num_steps - 1) else 0.0,  # or set as appropriate
                "discount": 1.0,
                "language_instruction": instr,
                "observation": {
                    "wrist_image": wrist_img,
                    "image": agent_img,
                    "state": joint_state,        
                    "ee_state": eef_state,
                }
            }
            steps.append(step_dict)

        # Compose RLDS episode dict
        out_dict = {
            "steps": steps,
            "episode_metadata": {
                "file_path": f"episode_{episode_idx}.npy"
            }
        }

        out_dir = os.path.join(self.args.env_dir, 'data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'episode_{episode_idx}.npy')
        try:
            np.save(out_path, out_dict)
            print(f"Saved RLDS episode {episode_idx} ({len(steps)} steps) to {out_path}")
            self.sequential_episode_buffer = []
            return True
        except Exception as e:
            print(f"Error saving episode {episode_idx}: {e}")
            return False


def _generate_env_specific_goal(env_name: str, state: dict, detector, pddl_path: str):
    """Generate environment-specific goals for CubeSorting, HeightStacking, AssemblyLineSorting, and PatternReplication."""
    goal_predicates = []
    
    if env_name == "CubeSorting":
        # Find all small cubes and assign them to target zones
        for predicate in state.keys():
            if "small" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(", ")
                goal_predicates.append(f'on {objs[0]} platform1')
            elif "small" in predicate and not state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(", ")
                goal_predicates.append(f'on {objs[0]} platform2')
        if goal_predicates:
            define_goal_in_pddl(pddl_path, goal_predicates)
        
    elif env_name == "HeightStacking":
        # Create stacking order based on sizes
        sizes = {}
        for predicate in state.keys():
            if "smaller" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(",")
                sizes[objs[0]] = objs[1]
        # Create stacking order based on sizes
        sorted_sizes = sorted(sizes.items(), key=lambda x: x[1])
        for i in range(len(sorted_sizes)-1):
            goal_predicates.append(f'on {sorted_sizes[i][0]} {sorted_sizes[i+1][0]}')
        # Add largest cube on platform
        if sorted_sizes:
            goal_predicates.append(f'on {sorted_sizes[-1][0]} platform')
        if goal_predicates:
            define_goal_in_pddl(pddl_path, goal_predicates)
        
    elif env_name == "AssemblyLineSorting":
        # Match types
        types = {}
        for predicate in state.keys():
            if "type_match" in predicate and state[predicate]:
                objs = predicate[predicate.find("(")+1:predicate.find(")")].split(",")
                types[objs[0]] = objs[1]
        
        # Collect all cubes with their positions and target bins
        cube_info = []
        for obj, bin_name in types.items():
            try:
                body_name = detector.object_id[obj]
                body_id = detector.env.sim.model.body_name2id(body_name)
                y_pos = detector.env.sim.data.body_xpos[body_id][1]
                cube_info.append((y_pos, obj, bin_name))
            except (KeyError, ValueError):
                cube_idx = int(obj.replace('cube', '')) if 'cube' in obj else 999
                cube_info.append((float(cube_idx), obj, bin_name))
        
        # Sort all cubes by Y position (left to right)
        cube_info.sort(key=lambda x: x[0])
        
        # Generate predicates in left-to-right order (cube_info is already sorted)
        for y_pos, cube_name, bin_name in cube_info:
            goal_predicates.append(f'on {cube_name} {bin_name}')
        if goal_predicates:
            define_goal_in_pddl(pddl_path, goal_predicates)
        
    elif env_name == "PatternReplication":
        # Get pattern from detector
        goal_predicates = detector.get_pattern_replication_goal()
        if goal_predicates:
            define_goal_in_pddl(pddl_path, goal_predicates)
    
    return goal_predicates


def symbolic_to_natural_instruction(op_str, env=None):
    """Convert a symbolic PDDL operator string to a natural language 
    instruction.
    """
    # Check if op_str is already in natural language. 
    # Natural language instructions start with 'pick up the' or 'place the'
    op_str_l = op_str.lower()
    if op_str_l.startswith("pick up the") or op_str_l.startswith("place the"):
        return op_str  # Already in natural language
    
    # Build dynamic color mapping from environment if available
    colors = {}
    cube_colors = hasattr(env, 'cube_colors')
    color_cats = hasattr(env, 'color_categories')
    cube_sizes = hasattr(env, 'cube_sizes')
    rgba_semantic_colors = hasattr(env, 'rgba_semantic_colors')
    
    if env is not None and cube_colors and color_cats:  # AssemblyLine env  
        for i in range(len(env.cube_colors)):
            color_idx = env.cube_colors[i]
            color_name = env.color_categories[color_idx][0]
            colors[f"cube{i}"] = f"{color_name} block"
    
    elif env is not None and cube_sizes:  # CubeSorting env
        # Get colors from sizes
        # Small cubes are blue, large cubes are red
        for i in range(len(env.cube_sizes)):
            size_type = env.cube_sizes[i]
            color_name = "blue" if size_type == "small" else "red"
            colors[f"cube{i}"] = f"{color_name} block"
    
    elif env is not None and cube_colors and rgba_semantic_colors:
        # For PatternReplication and HeightStacking environments
        # Map RGBA values to color names
        rgba_to_name = {tuple(v[:3]): k for k, v in env.rgba_semantic_colors.items()}
        for i in range(len(env.cube_colors)):
            rgba = tuple(env.cube_colors[i][:3])
            color_name = rgba_to_name.get(rgba, "unknown")
            colors[f"cube{i}"] = f"{color_name} block"
    else:
        # Fallback to hardcoded colors for environments without dynamic colors
        colors = {
            "cube1": "blue block", "cube2": "red block", "cube3": "green block",
            "cube4": "yellow block"}
    
    areas  = {
        "peg1": "left area", "peg2": "middle area", "peg3": "right area", 
        "bin0": "red bin", "bin1": "green bin", "bin2": "blue bin",
        "platform1": "blue zone", "platform2": "red zone", 
        "platform": "gray zone",
        "reference_platform": "reference area", "target_platform": "gray zone", 
        "table": "table"
        }
    op = op_str.lower().split()
    if not op: return ""
    
    if op[0] == "pick":
        block = colors.get(op[1], op[1])
        if len(op) > 2:  # Could add picking from object. Not implemented.
            # from_obj = colors.get(op[2], op[2])
            return f"Pick up the {block}."
        else:
            return f"Pick up the {block}."
    
    if op[0] == "place":
        block = colors.get(op[1], op[1])
        # Place target can be area or another block
        if op[2].startswith("cube"):
            target = colors.get(op[2], op[2])
            return f"Place the {block} on top of the {target}."
        else:
            area = areas.get(op[2], op[2])
            return f"Place the {block} on the {area}."
    return op_str

def _quat2axisangle(quat):
    """
    Copied from robosuite: 
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den