import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import robosuite_task_zoo
from datetime import datetime
import gymnasium as gym
from PIL import Image

from robosuite.utils.detector import HanoiDetector, KitchenDetector, NutAssemblyDetector

from planning.planner import *
from planning.executor import *

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

planning_predicates = {"Hanoi": ['on', 'clear', 'grasped'],
                          "KitchenEnv": ['on', 'clear', 'grasped', 'stove_on'],
                          "NutAssembly": ['on', 'clear', 'grasped']}  

planning_mode = {"Hanoi": 0,
                 "KitchenEnv": 1,
                 "NutAssembly": 0}

class RecordDemos(gym.Wrapper):
    def __init__(self, 
                 env,
                 vision_based,
                 detector,
                 pddl_path,
                 args, 
                 render=False,
                 randomize=True,
                 noise_std_factor=0.5):
        
        # Run super method
        super().__init__(env=env)
        
        # Set args
        self.env = env
        self.vision_based = vision_based
        self.detector = detector
        self.pddl_path = pddl_path
        self.args = args
        self.render = render
        self.randomize = randomize
        self.noise_std_factor = noise_std_factor

        # Set up the environment
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')

        self.sequential_episode_buffer = []

        # Init buffer
        self.data_buffer = dict()
        self.action_steps = []

        # Detect init state
        self.reset()
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        # Filter and keep only the predicates that are in planning_predicates[args.env] and are True and map them to the PDDL format
        init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate in planning_predicates[args.env]}

        # Usage

        add_predicates_to_pddl(pddl_path, init_predicates)
        # Generate a plan
        self.plan, _ = call_planner(pddl_path, mode=planning_mode[args.env])
        print("Task demonstrated: ", self.plan)

    def get_task(self):
        """
        Returns the task
        """
        return self.plan[self.operator_step]

    # def operator_to_function(self, operation):
    #     """
    #     A function that maps the operation to the corresponding function
    #     returns the function, the semantic description of the operation and the goal
    #     """
    #     map_color = {"cube1": "blue", "cube2": "red", "cube3": "green",}
    #     operation = operation.lower().split(' ')
    #     if 'pick' in operation[0]:
    #         if self.args.env == "Hanoi":
    #             return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
    #         else:
    #             return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
    #     elif 'place' in operation[0]:
    #         if self.args.env == "Hanoi":
    #             return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
    #         else:
    #             return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
    #     elif 'turn-on' in operation[0]:
    #         return self.turn_on_button, f'switch on button', "button"
    #     elif 'turn-off' in operation[0]:
    #         return self.turn_off_button, f'switch off button', "button"
    #     else:
    #         return None

    def reset(self, seed=None):
        """
        The reset function that resets the environment
        """
        print("[RecordDemos] Resetting environment...") # Debug print
        self.operator_step = 0
        self.sequential_episode_buffer = []

        # old buffers
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.task_buffer = list()
        try:
            obs, _ = self.env.reset()
        except:
            obs = self.env.reset()
        
        # Ensure obs is a dictionary, especially if not using GymWrapper or vision
        if not isinstance(obs, dict):
             # This might happen if vision=False and GymWrapper isn't used correctly
             # You might need to manually get the observation dictionary
             print("Warning: env.reset() did not return a dictionary. Attempting manual obs fetch.")
             obs = self.env._get_observations() # Access internal method (use with caution)

        self.env.sim.forward() # Ensure sim state is updated
        return obs # Return the observation dictionary

    def run_trajectory(self, obs):
        """
        Runs the trajectory
        """
        done = False
        for operation in self.plan:
            try:
                function, self.task, goal = self.operator_to_function(operation)
            except:
                continue
            print(f'Performing task: {self.task}, with goal: {goal}')
            done, obs = function(obs, goal)
            if not(done):
                print("Failed to perform task")
                return False
            print("Successful task")
        print("Successful episode?: ", done)
        return done

    def save_trajectory(self, num_recorded_eps):
        """
        Saves the trajectory to the buffer
        """
        print(f"Attempting to save trajectory for episode {num_recorded_eps}...") # Debug
        episode_for_lerobot = []
        
        # Check if the sequential buffer actually contains data
        if not self.sequential_episode_buffer:
            print(f"Warning: Skipping saving episode {num_recorded_eps} - sequential buffer is empty.")
            # No need to reset here, reset() clears the buffer for the next episode
            return False # Indicate saving did not happen

        # --- Iterate through the sequential buffer ---
        for step_index, step_data in enumerate(self.sequential_episode_buffer):
            try:
                obs_dict = step_data['obs_dict']
                action_7d = step_data['action']
                language_instruction = step_data['language_instruction']
                print(f"Step {step_index}: {language_instruction}") # Debug

                # --- 1. Extract Wrist Image ---
                # Use the correct key for the wrist camera specified in suite.make
                wrist_image = obs_dict.get('robot0_eye_in_hand_image')
                if wrist_image is None:
                    # Handle missing wrist image: Print warning and use placeholder
                    print(f"Warning: Wrist image ('robot0_eye_in_hand_image') not found in step {step_index} of episode {num_recorded_eps}. Using placeholder.")
                    # Define placeholder shape based on your camera config or desired output
                    wrist_placeholder_shape = (256, 256, 3) # Example: Make sure this matches LeRobot feature def
                    wrist_image = np.zeros(wrist_placeholder_shape, dtype=np.uint8)
                else:
                    # Ensure correct dtype
                    wrist_image = np.asarray(wrist_image, dtype=np.uint8)
                    # Optional: Add shape check/resizing if needed
                    # expected_wrist_shape = (256, 256, 3)
                    # if wrist_image.shape != expected_wrist_shape:
                    #     print(f"Warning: Wrist image shape is {wrist_image.shape}, resizing to {expected_wrist_shape}.")
                    #     wrist_image = np.array(Image.fromarray(wrist_image).resize(expected_wrist_shape[:2]))
            
                # --- 2. Extract Main Camera Image ---
                # Use the correct key for the main camera specified in suite.make
                agent_image = obs_dict.get('agentview_image')
                if agent_image is None:
                    # print(f"Warning: Main image ('agentview_image') not found in step {step_index} of episode {num_recorded_eps}. Using placeholder.")
                    agent_placeholder_shape = (256, 256, 3) # Example: Make sure this matches LeRobot feature def
                    agent_image = np.zeros(agent_placeholder_shape, dtype=np.uint8)
                else:
                    # Ensure correct dtype
                    agent_image = np.asarray(agent_image, dtype=np.uint8)
                    # Optional: Add shape check/resizing
                    # expected_agent_shape = (256, 256, 3)
                    # if agent_image.shape != expected_agent_shape:
                    #     print(f"Warning: Agent image shape is {agent_image.shape}, resizing to {expected_agent_shape}.")
                    #     agent_image = np.array(Image.fromarray(agent_image).resize(expected_agent_shape[:2]))
            
                # --- 3. Extract/Construct Proprioceptive State ---
                # Reduced form for proprioceptive state - 7 joint positions and 1
                EXPECTED_STATE_DIM = 8
                # --- Gripper QPOS interpretation (NEEDS USER INPUT FROM XML) ---
                # Example: Assume first joint in qpos is main actuator
                GRIPPER_QPOS_INDEX = 0
                # Find these limits in your specific Kinova gripper XML file!
                GRIPPER_JOINT_MIN = 0.0 # Example Placeholder
                GRIPPER_JOINT_MAX = 0.8 # Example Placeholder (e.g., Robotiq 2F-85 range)
                # ---
                try:
                    # --- Define keys and expected dimensions (EXAMPLE for Panda) ---
                    # --- ADJUST KEYS AND DIMS FOR KINOVA3 ---
                    # print(obs_dict.keys())
                    joint_sin = obs_dict['robot0_joint_pos_sin']
                    joint_cos = obs_dict['robot0_joint_pos_cos']
                    # Calculate the 7 joint angles in radians
                    arm_joint_positions = np.arctan2(joint_sin, joint_cos)

                    # --- Get 1 Gripper State value by interpreting qpos ---
                    gripper_qpos = obs_dict.get('robot0_gripper_qpos')
                    kinova_gripper_state_norm = 0.0 # Default value

                    if gripper_qpos is not None and len(gripper_qpos) > GRIPPER_QPOS_INDEX:
                        current_joint_val = gripper_qpos[GRIPPER_QPOS_INDEX]
                        # Normalize based on known limits (USER MUST PROVIDE)
                        if GRIPPER_JOINT_MAX > GRIPPER_JOINT_MIN:
                            kinova_gripper_state_norm = (current_joint_val - GRIPPER_JOINT_MIN) / (GRIPPER_JOINT_MAX - GRIPPER_JOINT_MIN)
                            kinova_gripper_state_norm = np.clip(kinova_gripper_state_norm, 0.0, 1.0)
                        else:
                            print(f"Warning: Invalid gripper joint limits [{GRIPPER_JOINT_MIN}, {GRIPPER_JOINT_MAX}] in step {step_index}.")
                    else:
                        print(f"Warning: 'robot0_gripper_qpos' not found or too short in step {step_index}. Using 0.0 for gripper.")
                    kinova_gripper_state = np.array([kinova_gripper_state_norm], dtype=np.float32) # Shape (1,)
                    # --- Concatenate into 8-element state vector ---
                    proprio_state = np.concatenate([
                        arm_joint_positions,    # 7 elements
                        kinova_gripper_state    # 1 element
                    ]).astype(np.float32)
                    # ---

                    # Verify final shape
                    if proprio_state.shape[0] != EXPECTED_STATE_DIM:
                        print(f"Warning: Constructed proprio_state shape {proprio_state.shape} != expected ({EXPECTED_STATE_DIM},). Using zeros.")
                        proprio_state = np.zeros(EXPECTED_STATE_DIM, dtype=np.float32)

                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error constructing proprio state manually in step {step_index}: {e}. Using zeros.")
                    proprio_state = np.zeros(EXPECTED_STATE_DIM, dtype=np.float32)


                # --- 4. Append the formatted step data ---
                episode_for_lerobot.append({
                    'image': agent_image,           # uint8 image
                    'wrist_image': wrist_image,     # uint8 image
                    'state': proprio_state,         # float32 vector
                    'action': action_7d,            # float32 vector (already 7D)
                    'language_instruction': language_instruction, # string
                })
                
            except KeyError as e:
                print(f"Warning: Missing key {e} in step data. Skipping this step.")
                continue
            except Exception as e:
                print(f"Error processing step {step_index} in episode {num_recorded_eps}: {type(e).__name__}: {e}. Skipping step.")
                continue # Skip this step

        # --- 5. Save the completed episode list ---
        if not episode_for_lerobot:
            print(f"Warning: No valid steps processed for episode {num_recorded_eps}. Not saving file.")
            saved_successfully = False
        else:
            # Define output path (consider making directory structure more robust)
            # Example: Use args.env_dir defined in __main__
            # output_dir = os.path.join(self.args.env_dir, "hanoi_dataset") # Save within experiment dir
            output_dir = f'data/{self.args.env.lower()}_dataset/data' # Old path - choose one
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'episode_{num_recorded_eps}.npy')

            try:
                np.save(output_path, episode_for_lerobot)
                print(f"Saved intermediate episode data ({len(episode_for_lerobot)} steps) to {output_path}")
                saved_successfully = True
            except Exception as e:
                print(f"Error saving episode {num_recorded_eps} to {output_path}: {e}")
                saved_successfully = False

        # --- 6. Resetting environment ---
        # Reset is handled by the main loop calling env.reset()
        # We just need to clear the buffer here
        self.sequential_episode_buffer = []
        print("-" * 20) # Separator after saving attempt

        # Return True if saving was successful, False otherwise (or handle differently)
        # The return value of save_trajectory might need adjustment based on how the main loop uses it.
        # If the main loop simply increments recorded_eps if done=True, this return might not be needed.
        # Returning obs from reset() here is definitely not needed.
        return saved_successfully # Or simply return None if the main loop doesn't use the return value
        # for step in self.action_steps:
        #     if step in self.episode_buffer.keys():
        #         for i in range(0,len(self.episode_buffer[step]),2): 
        #             # Convert action from dx,dy,dz,gripper to dx,dy,dz,d_roll,d_pitch,d_yaw,gripper
        #             action_4dim = self.episode_buffer[step][i]
        #             action = np.concatenate((action_4dim[:3], np.zeros(3), action_4dim[3:]))
        #             image = self.episode_buffer[step][i+1]
        #             wrist_image = None
        #             episode.append({
        #                 'image': image,
        #                 'wrist_image': wrist_image,
        #                 'state': None, #Needs to be robot proprio
        #                 'action': action,
        #                 'language_instruction': step,
        #             })
        # np.save(f'data/hanoi_dataset/data/episode_{num_recorded_eps}.npy', episode)
        
        # obs = self.reset()
        # return obs

    def operator_to_function(self, operation):
        """
        Maps the PDDL operation to a function, a natural language instruction
        (with descriptive name substitutions), and the goal parameters.
        """
        # --- Define the mapping for descriptive names ---
        name_mapping = {
            # Hanoi Mappings
            "cube1": "blue cube",
            "cube2": "red cube",
            "cube3": "green cube",
            "peg1": "area 1",
            "peg2": "area 2",
            "peg3": "area 3",
            # Kitchen Mappings (Add/adjust as needed)
            "button": "button",
            "pot": "pot",
            "stove": "stove", # Example, if 'stove' is used as a location
            # NutAssembly Mappings (Add/adjust as needed)
            "roundnut": "round nut",
            "squarenut": "square nut",
            "roundpeg": "round peg", # Location/object
            "squarepeg": "square peg", # Location/object
            # Add any other object/location names from your PDDL domains
        }
        # ---

        original_operation_str = operation # Keep original for logging if needed
        operation_parts = operation.lower().split(' ')
        op_type = operation_parts[0]

        try:
            # Default values if parsing fails
            func, instruction, goal = None, None, None

            if 'pick' in op_type:
                # Expected format: pick <object> <source_location>
                if len(operation_parts) >= 3:
                    obj_original = operation_parts[1]
                    loc_original = operation_parts[2]
                    # Get mapped names, defaulting to original if no mapping exists
                    obj_mapped = name_mapping.get(obj_original, obj_original)
                    loc_mapped = name_mapping.get(loc_original, loc_original)
                    # Construct the natural language instruction
                    instruction = f'pick the {obj_mapped} from {loc_mapped}'
                    goal = obj_original # Goal for pick is usually the object's original ID
                    func = self.pick
                else:
                    print(f"Warning: Could not parse 'pick' operation: '{original_operation_str}'")

            elif 'place' in op_type:
                # Expected format: place <object> <destination_location>
                if len(operation_parts) >= 3:
                    obj_original = operation_parts[1]
                    loc_original = operation_parts[2]
                    # Get mapped names
                    obj_mapped = name_mapping.get(obj_original, obj_original)
                    loc_mapped = name_mapping.get(loc_original, loc_original)
                     # Construct the natural language instruction
                    instruction = f'place the {obj_mapped} on {loc_mapped}'
                    # Goal for place needs both original IDs
                    goal = (obj_original, loc_original)
                    func = self.place
                else:
                    print(f"Warning: Could not parse 'place' operation: '{original_operation_str}'")

            elif 'turn-on' in op_type:
                # Expected format: turn-on <object> (e.g., 'button')
                if len(operation_parts) >= 2:
                    target_original = operation_parts[1] # Get target from operation
                else:
                    target_original = "button" # Fallback if format is just 'turn-on'
                    print(f"Warning: Assuming 'turn-on' target is 'button' for operation: '{original_operation_str}'")

                target_mapped = name_mapping.get(target_original, target_original)
                instruction = f'switch on the {target_mapped}'
                goal = target_original # Goal is the original ID
                func = self.turn_on_button

            elif 'turn-off' in op_type:
                # Expected format: turn-off <object> (e.g., 'button')
                if len(operation_parts) >= 2:
                    target_original = operation_parts[1] # Get target from operation
                else:
                    target_original = "button" # Fallback if format is just 'turn-off'
                    print(f"Warning: Assuming 'turn-off' target is 'button' for operation: '{original_operation_str}'")

                target_mapped = name_mapping.get(target_original, target_original)
                instruction = f'switch off the {target_mapped}'
                goal = target_original # Goal is the original ID
                func = self.turn_off_button

            else:
                # Operation type not recognized
                print(f"Warning: Unhandled operation type '{op_type}' in operator_to_function for operation: '{original_operation_str}'.")

            # Return the found function, the generated instruction, and the goal
            return func, instruction, goal

        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"Error processing operation '{original_operation_str}': {type(e).__name__}: {e}")
            return None, None, None # Return None for all parts on error

    def zip_buffer(self, dir_path):
        # Decompose the data buffer into action steps
        for step in self.action_steps:
            if step not in self.data_buffer.keys():
                continue
            # Convert the data buffer to bytes
            data_bytes = pickle.dumps(self.data_buffer[step])
            file_path = dir_path + step + '.zip'
            # Write the bytes to a zip file
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)

    def record_step(self, obs_dict, action_7d):
        """
        Records the current step's data into the sequential episode buffer.

        Args:
            obs_dict (dict): Observation dictionary *before* action (optional, might be useful later).
            action_7d (np.ndarray): The 7D action [dx, dy, dz, dr, dp, dy, gripper] applied.
            next_obs_dict (dict): The full observation dictionary *after* action, containing
                                  images and proprioceptive state.
        """
        # Store the essential data for this step
        step_data = {
            # 'prev_obs_dict': copy.deepcopy(obs_dict), # Optional: Store previous obs if needed
            'action': np.array(action_7d, dtype=np.float32),
            'obs_dict': copy.deepcopy(obs_dict), # Crucial: contains images, state
            'language_instruction': str(self.task) # Current semantic task from the planner
        }
        self.sequential_episode_buffer.append(step_data)

        # action_step = self.task
        # if action_step not in self.action_steps:
        #     self.action_steps.append(action_step)
        # if action_step not in self.episode_buffer.keys():
        #     # self.episode_buffer[action_step] = [obs, action, next_obs] # Why 3 things?
        #     self.episode_buffer[action_step] = [action, next_obs]
        # else:
        #     self.episode_buffer[action_step] += [action, next_obs]
        # return state_memory


    def cap(self, eps, max_val=0.12, min_val=0.01):
        """
        Caps the displacement
        """
        # If the displacement is greater than the max value, cap it
        if np.linalg.norm(eps) > max_val:
            eps = eps / np.linalg.norm(eps) * max_val
        # If the displacement is smaller than the min value, cap it
        if np.linalg.norm(eps) < min_val:
            eps = eps / np.linalg.norm(eps) * min_val
        return eps

    def to_osc_pose(self, action):
        """
        Converts the action to the OSC pose
        """
        # Add [0, 0, 0] to the action to make it a 6D action and the gripper aperture at the end
        # Insert [0, 0, 0] from action[3] to action[5]
        action = np.insert(action, 3, [0, 0, 0])
        return action

    def relative_obs_mapping(self, goal):
        """
        Get the relative observation between the gripper and the goal + the gripper aperture
        """
        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
        dist = np.linalg.norm(object_pos - gripper_pos)

        gripper_quat = self.env.sim.data.body_xquat[self.gripper_body]
        object_quat = self.env.sim.data.body_xquat[goal]
        angle = self.quaternion_to_euler(gripper_quat)[2] - self.quaternion_to_euler(object_quat)[2]

        left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)

        return np.array([dist, angle, aperture])
        
    def quaternion_to_euler(self, quat):
        """
        Converts a quaternion to euler angles
        """
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def pick(self, obs, goal):
        """
        Transitons the environment to a state where the gripper is has picked the object
        """

        # --- Fetch initial observation manually ---
        try:
            current_obs_dict = self.env.env._get_observations() # Get dict s_t
            if not isinstance(current_obs_dict, dict): raise TypeError("Initial fetch failed")
            # print(f"[Pick Init] Fetched obs keys: {list(current_obs_dict.keys())}") # Debug keys
        except Exception as e_fetch_init:
            print(f"Error fetching initial obs in pick: {e_fetch_init}. Cannot start task.")
            return False, obs # Return original obs on failure
        
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")
        reset_step_count = 0

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        # Moving gripper 10 cm above the object
        ref_z = 1.1
        z_pos = 0
        loop_stage = "Move Z High"
        while z_pos < ref_z:

            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails

            z_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])[2]
            dist_z = abs(ref_z - z_pos)
            dist_z = self.cap([dist_z])
            action = 5*np.concatenate([[0, 0], dist_z, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z), 1), [0]])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.env.render() if self.render else None
            
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
            dist_z = dist_z[0]
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over object...")
        gripper_goal = "pot_handle" if goal_str == "pot" else goal_str
    
        while not state['over(gripper,{})'.format(gripper_goal)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if goal_str == 'pot':
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, -0.09, 0])
            elif self.args.env == "NutAssembly":
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)            
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)  
            # if self.state_memory is None:
            #     return False, obs

            obs, state = next_obs, next_state # Update obs for the next loop iteration
            

            reset_step_count += 1

            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 100:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs

            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            action = np.asarray([0,0,0,1])
            action = self.to_osc_pose(action)

            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination
            
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs

            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.1, 3), [0]])
            action = 5*self.cap(action)
            action = self.to_osc_pose(action)

            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 300:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def place(self, obs, goal):
        """
        Transitons the environment to a state where the gripper is object is placed at the place to drop
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")
        reset_step_count = 0

        goal_str = goal[1]
        pick_str = goal[0]
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])
        pick_body = self.env.sim.model.body_name2id(self.detector.object_id[pick_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        if 'peg' in goal_str and self.args.env == "Hanoi":
            goal_pos = self.env.sim.data.body_xpos[goal][:3] - np.array([0.1, 0.04, 0])
        else:
            goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        #print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            if pick_str == 'pot':
                gripper_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("PotObject_root")]
            elif self.args.env == "NutAssembly":
                if goal_str == "roundpeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([0, -0.01, 0]) #+np.array([0, -0.05, 0])
                elif goal_str == "squarepeg":
                    gripper_pos = np.asarray(self.env.sim.data.body_xpos[pick_body]) +np.array([+0.015, +0.01, 0])#+ np.array([0.01, -0.025, 0])
            else:
                gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in goal_str and self.args.env == "Hanoi":
                object_pos = self.detector.area_pos[goal_str]
            # elif self.args.env == "NutAssembly" and goal_str == "roundpeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
            # elif self.args.env == "NutAssembly" and goal_str == "squarepeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+ np.array([-0.01, 0.025, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)

            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination
            
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down picked object on place to drop...")
        while not state['on({},{})'.format(pick_str, goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            #object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            place_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [- (gripper_pos[2] - place_pos[2])]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 200:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("dropping object...")
        while not(state['open_gripper(gripper)']):#state['grasped({})'.format(goal_str)]:
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            
            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Resetting gripper")
        # First move up
        for _ in range(15):
            try:
                obs = self.env.env._get_observations() # s_t
                if not isinstance(obs, dict): raise TypeError("Loop fetch failed")
            except Exception as e_fetch_loop:
                 print(f"Error fetching obs in {loop_stage} step {reset_step_count}: {e_fetch_loop}.")
                 return False, obs # Fail if obs fetch fails
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)

            self.record_step(obs, action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated # Check for termination

            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            # self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str)
            # if self.state_memory is None:
            #     return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_on_button(self, obs, goal):
        """
        Transitons the environment to a state where button is turned on
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        #print("Moving gripper over button...")
        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to swith level ...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Turning on button...")
        while not state['stove_on()']:
            action = np.asarray([0,0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs
    
    def turn_off_button(self, obs, goal):
        """
        Transitons the environment to a state where button is turned off
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        #print("\n\n\t\t---------------------PICK GOAL IS: ", goal)
        #print("\n\n")

        goal_str = goal
        goal = self.env.sim.model.body_name2id(self.detector.object_id[goal_str])

        goal_pos = self.env.sim.data.body_xpos[goal][:3]
        goal_quat = self.env.sim.data.body_xquat[goal]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        reset_step_count = 0

        #print("Moving gripper over button...")
        while not state['over(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to swith level ...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Turning off button...")
        while state['stove_on()']:
            action = np.asarray([0,-0.3,0,0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 50:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1

        return True, obs

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly'], help='Name of the environment to run the experiment in')
    parser.add_argument('--dir', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(100), help='Number of trajectories to record')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the recording')
    parser.add_argument('--vision', action='store_true', help='Use vision based observation')
    parser.add_argument('--relative_obs', action='store_true', help='Use relative gripper-goal observation')
    parser.add_argument('--vla', action='store_true', help='Store the data in VLA friendly format')

    args = parser.parse_args()
    args.vla = True
    # Set the random seed
    np.random.seed(args.seed)
    # Define the directories
    dir = args.dir
    experiment_name = args.env + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid 
    if args.name is not None:
        experiment_id = args.name
    args.env_dir = os.path.join(dir, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

    # Create the directories
    args.traces = args.env_dir + '/traces/'
    os.makedirs(args.env_dir, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)

    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    # Create the environment
    env = suite.make(
        args.env,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=100000000,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )

    env.reset()

    # Wrap the environment
    if args.env == 'Hanoi':
        detector = HanoiDetector(env)
        pddl_path = './planning/PDDL/hanoi/'
    elif args.env == 'KitchenEnv':
        detector = KitchenDetector(env)
        pddl_path = './planning/PDDL/kitchen/'
    elif args.env == 'NutAssembly':
        detector = NutAssemblyDetector(env)
        pddl_path = './planning/PDDL/nut_assembly/'
    env = GymWrapper(env, proprio_obs=not(args.vision))
    env = RecordDemos(env, args.vision, detector, pddl_path, args, render=args.render, randomize=True)

    os.makedirs(f'data/hanoi_dataset/data', exist_ok=True)

    # Run the recording of the demonstrations
    num_recorded_eps = 0
    recorded_eps = 0
    episode = 1
    while recorded_eps < args.episodes: 
        obs = env.reset()
        print("Episode: {}".format(episode))
        keys = list(env.data_buffer.keys())
        done = env.run_trajectory(obs)
        if done:
            obs = env.save_trajectory(recorded_eps)
            recorded_eps += 1
        episode += 1
        if len(keys) > 0:
            num_recorded_eps = len(env.data_buffer[keys[0]])
            print("Number of recorded episodes: {}".format(num_recorded_eps))
            print("\n\n")