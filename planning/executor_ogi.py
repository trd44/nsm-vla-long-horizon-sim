'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import dill
import torch
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import TrainDiffusionTransformerLowdimWorkspace
import copy

import cv2
cv2.destroyAllWindows = lambda: None

class Executor():
	def __init__(self, id, mode, Beta=None):
		super().__init__()
		self.id = id
		self.Beta = Beta
		self.mode = mode
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}
    
class Executor_Diffusion(Executor):
    def __init__(self, 
                 id, 
                 policy, 
                 Beta, 
                 count=0,
                 nulified_action_indexes=[], 
                 oracle=False, 
                 horizon=None, 
                 use_yolo=True, 
                 save_data=False,
                 tracked_positions={}
                 ):
        super().__init__(id, "RL", Beta)
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.horizon = horizon
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.oracle = oracle
        self.use_yolo = use_yolo
        self.save_data = save_data
        self.image_buffer = []
        self.map_id_semantic = {
                "blue cube": "cube1",
                "red cube": "cube2",
                "green cube": "cube3",
                "yellow cube": "cube4",
        }
        self.tracked_positions = tracked_positions
        self.detected_positions = {}
        # Store bboxes centers and groundtruth positions of cubes
        self.bboxes_centers = []
        self.count = count

    def load_policy(self, detector=None, yolo_model=None, regressor_model=None, image_size=256):
        path = self.policy
        # load checkpoint
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = TrainDiffusionTransformerLowdimWorkspace
        cfg.policy.num_inference_steps = 10
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        #device = torch.device(self.device)
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.model = policy

        if detector is not None:
            self.detector = detector
        if yolo_model is not None:
            self.yolo_model = yolo_model
            self.image_size = image_size
        if regressor_model is not None:
            self.regressor_model = regressor_model


    def pixel_to_world_dual(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        # Load linear Regression models for cube positions
        models_dual = self.regressor_model
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        try:
            features = np.array([[float(cls_id),
                                float(px1), float(py1), float(w1), float(h1), float(conf1),
                                float(px2), float(py2), float(w2), float(h2), float(conf2),
                                float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            x = reg_x_dual.predict(features)[0]
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        except:
            features = np.array([[
                      float(px1), float(py1), float(w1), float(h1), float(conf1),
                      float(px2), float(py2), float(w2), float(h2), float(conf2),
                      float(ee_x), float(ee_y), float(ee_z)]], dtype=np.float64)
            x = reg_x_dual.predict(features)[0]
            y = reg_y_dual.predict(features)[0]
            z = reg_z_dual.predict(features)[0]
        return x, y, z

    def yolo_estimate(self, image1, image2, save_video=False, cubes_obs=None, ee_pos=None):
        # Resize the image to fit YOLO input requirements
        cubes_predicted_xyz = {}
        confidence_1 = {}
        try:
            image1 = cv2.resize(image1, (256, 256))
        except Exception as e:
            print("Error resizing image: ", e, image1.shape, image1.dtype)
        try:
            image2 = cv2.resize(image2, (256, 256))
        except Exception as e:
            print("Error resizing image2: ", e, image2.shape, image2.dtype)
            # Concatenate the two images side by side
        # Mirror the image (top to bottom)
        image1 = cv2.flip(image1, 0)
        # Convert the image to BGR format if it is not already
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # Run YOLO model on the image
        predictions1 = self.yolo_model.predict(image1, verbose=False)[0]
        image2 = cv2.flip(image2, 0)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        predictions2 = self.yolo_model.predict(image2, verbose=False)[0]

        # Ensure ndarray
        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1)
        if image2 is not None and not isinstance(image2, np.ndarray):
            image2 = np.array(image2)

        # Draw bounding boxes from Roboflow JSON
        for pred in predictions1.boxes:#.get("predictions", []):
            confidence_2 = {}

            cls_id = int(pred.cls)
            cls = self.yolo_model.names[cls_id]
            x, y, w, h = pred.xywhn.tolist()[0]
            conf = pred.conf
            # Convert normalized coordinates to pixel coordinates
            x = int(x * image1.shape[1])
            y = int(y * image1.shape[0])
            w = int(w * image1.shape[1])
            h = int(h * image1.shape[0])

            # Convert to pixel coordinates
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            if cls_id not in confidence_1:
                confidence_1[cls_id] = conf
            else:
                if conf > confidence_1[cls_id]:
                    confidence_1[cls_id] = conf
                else:
                    continue

            # Get the ground truth position of the cube
            ground_truth_xyz = cubes_obs[self.map_id_semantic[cls]]
            
            if image2 is not None:
                found_match = False
                for pred in predictions2.boxes:
                    cls_id2 = int(pred.cls)
                    if cls_id2 not in confidence_2:
                        confidence_2[cls_id2] = pred.conf
                    else:
                        if pred.conf > confidence_2[cls_id2]:
                            confidence_2[cls_id2] = pred.conf
                        else:
                            continue
                    if cls_id2 == cls_id:
                        found_match = True
                        x_cam2, y_cam2, w_cam2, h_cam2 = pred.xywhn.tolist()[0]
                        conf_cam2 = pred.conf
                        # Convert normalized coordinates to pixel coordinates
                        x_cam2 = int(x_cam2 * image2.shape[1])
                        y_cam2 = int(y_cam2 * image2.shape[0])
                        w_cam2 = int(w_cam2 * image2.shape[1])
                        h_cam2 = int(h_cam2 * image2.shape[0])

                        self.bboxes_centers.append({
                            "px_cam1": x,
                            "py_cam1": y,
                            "w_cam1": w,
                            "h_cam1": h,
                            "conf_cam1": float(conf),
                            "cls": cls,
                            "px_cam2": x_cam2,
                            "py_cam2": y_cam2,
                            "w_cam2": w_cam2,
                            "h_cam2": h_cam2,
                            "conf_cam2": float(conf_cam2),
                            "ee_x": ee_pos[0] if ee_pos is not None else None,
                            "ee_y": ee_pos[1] if ee_pos is not None else None,
                            "ee_z": ee_pos[2] if ee_pos is not None else None,
                            "world_x": ground_truth_xyz[0],
                            "world_y": ground_truth_xyz[1],
                            "world_z": ground_truth_xyz[2],
                        })
                if not found_match:
                    x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2 = 0, 0, 0, 0, 0
                    self.bboxes_centers.append({
                                "px_cam1": x,
                                "py_cam1": y,
                                "w_cam1": w,
                                "h_cam1": h,
                                "conf_cam1": float(conf),
                                "cls": cls,
                                "px_cam2": 0,
                                "py_cam2": 0,
                                "w_cam2": 0,
                                "h_cam2": 0,
                                "conf_cam2": 0,
                                "ee_x": ee_pos[0] if ee_pos is not None else None,
                                "ee_y": ee_pos[1] if ee_pos is not None else None,
                                "ee_z": ee_pos[2] if ee_pos is not None else None,
                                "world_x": ground_truth_xyz[0],
                                "world_y": ground_truth_xyz[1],
                                "world_z": ground_truth_xyz[2],
                            })
                # Use dual camera regression to estimate the position
                predicted_xyz = self.pixel_to_world_dual(cls_id, x, y, w, h, conf, x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2, ee_pos[0], ee_pos[1], ee_pos[2])
            else:
                # Use single camera regression to estimate the position
                predicted_xyz = self.pixel_to_world(x, y)

            self.detected_positions.update({self.map_id_semantic[cls]: predicted_xyz})
            cubes_predicted_xyz.update({self.map_id_semantic[cls]: predicted_xyz})

            if save_video:
                # Draw box + label
                cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image1, f"{cls}:{float(conf):.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_video:
            # Append to buffer for video saving
            if not hasattr(self, "image_buffer"):
                self.image_buffer = []

            #print("Image shape:", image1.shape, "dtype:", image1.dtype)
            self.image_buffer.append(image1.copy())

        return cubes_predicted_xyz

    def save_video(self, output_path="output.mp4", fps=10):
        if not self.image_buffer:
            print("No frames to save.")
            return
        
        height, width, _ = self.image_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in self.image_buffer:
            out.write(frame)

        out.release()
        print(f"Video saved at {output_path}")

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        import pandas as pd
        if not self.bboxes_centers:
            print("No bounding boxes data to save.")
            return
        
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        print(f"YOLO data saved at {output_path}")

    def action_obs_mapping(self, obs, action_step="PickPlace", relative=False):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "obj_to_pick_pos": (4,7), "place_to_drop_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (6,7), "place_to_drop_z": (9,10)}
        # trace_obs_list = obj_to_pick_pos - gripper_pos, aperture, place_to_drop_pos - gripper_pos
        # reach_pick_obs_list = obj_to_pick_pos - gripper_pos
        # pick_obs_list = obj_to_pick_z - gripper_z, aperture
        # reach_drop_obs_list = place_to_drop_pos - gripper_pos
        # drop_obs_list = place_to_drop_z - gripper_z, aperture

        oracle = np.array([])
        if action_step == "PickPlace":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]]])
        elif action_step == "ReachPick":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]]
        elif action_step == "Grasp" or action_step == "Pick":
            if relative:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        elif action_step == "ReachDrop":
            if relative:
                oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
            else:
                oracle = obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]]
        elif action_step == "Drop":
            if relative:
                oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
            else:
                oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        else:
            oracle = obs
        return oracle


    def prepare_obs(self, obs, action_step="PickPlace"):
        obs_dim = {"PickPlace": 7, "ReachPick": 3, "Grasp": 2, "ReachDrop": 3, "Drop": 2, "Pick": 2}
        #print(len(obs), obs_dim[action_step])
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((len(obs), obs_dim[action_step]))
        for j, env_n_obs in enumerate(obs):
            # Prepare the observation for the policy
            obs_policy = self.action_obs_mapping(env_n_obs, action_step=action_step, relative=False)
            #keypoint_policy = self.keypoint_mapping(obs_step, action_step=action_step)
            #concatenated_obs = np.concatenate([keypoint_policy, obs_policy], axis=-1)
            
            # Resize env_n_obs[i] to match the new shape
            #returned_obs[j][i] = concatenated_obs
            returned_obs[j] = obs_policy
        #print("Returned obs shape: ", returned_obs.shape)
        #print("Original obs shape: ", obs.shape)
        return returned_obs

    def get_object_obs(self, env, objects_pos, predicted_pos, obj_to_pick, place_to_drop, relative_obs=True):
        gripper_pos = objects_pos["gripper"]
        left_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_left_inner_finger")])
        right_finger_pos = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_right_inner_finger")])
        aperture = np.linalg.norm(left_finger_pos - right_finger_pos)*1000.
        #print(obj_to_pick)
        try:
            obj_to_pick_pos = predicted_pos[obj_to_pick] if obj_to_pick in predicted_pos else objects_pos[obj_to_pick]
        except:
            obj_to_pick_pos = np.array([0.0, 0.0, 0.0])
        try:
            place_to_drop_pos = predicted_pos[place_to_drop] if place_to_drop in predicted_pos else objects_pos[place_to_drop]
        except:
            place_to_drop_pos = np.array([0.0, 0.0, 0.0])
        if relative_obs:
            rel_obj_to_pick_pos = gripper_pos - obj_to_pick_pos
            rel_place_to_drop_pos = gripper_pos - place_to_drop_pos
            obs = np.concatenate([gripper_pos, [aperture], -rel_obj_to_pick_pos*1000, -rel_place_to_drop_pos*1000])
        else:
            obs = np.concatenate([gripper_pos, [aperture], obj_to_pick_pos, place_to_drop_pos])
        return obs

    def insert_yolo_estimate(self, obs, cubes_xyz, obj_to_pick, place_to_drop):
        # Insert the yolo estimated positions in the obs
        # cubes_obs is a dictionary of shape {cube_id: [x, y, z]}

        for key in self.tracked_positions.keys():
                cubes_xyz[key] = self.tracked_positions[key]

        if obj_to_pick in cubes_xyz:
            obj_to_pick_xyz = cubes_xyz[obj_to_pick]
            obs[7:10] = np.asarray(obj_to_pick_xyz)
        if place_to_drop in cubes_xyz:
            place_to_drop_xyz = cubes_xyz[place_to_drop]
            obs[4:7] = np.asarray(place_to_drop_xyz)
        return obs
                    
    def obs_base_from_info(self, info):
        obs_base = []
        for i in range(len(info)):
            obs_base.append(info[i]["obs_base"])
        return np.array(obs_base)

    def image_from_info(self, info, camera="agentview"):
        images = []
        for i in range(len(info)):
            if camera in info[i]:
                images.append(info[i][camera])
        return np.array(images)
    
    def track_positions(self, state, ee_pos, obs, symgoal):
        # Track the position of the cubes, if the cube is grasped its position is the ee position
        #print("State: ", state)
        for relation, value in state.items():
            if 'grasped' in relation and value:
                # Get the cube id
                cube_id = relation.split('(')[1].split(',')[0].split(')')[0]
                self.tracked_positions[cube_id] = np.asarray(ee_pos)
                if cube_id == symgoal[0]:
                    # If the cube is the one to pick, update the obs
                    obs[7:10] = np.asarray(ee_pos)
        return obs


    def valid_state_f(self, state):
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        if len(state) != 3:
            return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        return True

    def map_gripper(self, action):
        # Change last value of the action (called gripper_action) to a mapped discretized value of the gripper opening
        # -0.5 < gripper_action < 0.5 is mapped to 0
        # gripper_action <= -0.5 is mapped to 0.1
        # gripper_action >= -0.5 is mapped to -0.1
        # Returns the modified action array
        action_gripper = action[-1]
        if -0.5 < action_gripper < 0.5:
            # Do nothing
            action_gripper = np.array([0])
        if action_gripper <= -0.5:
            # Close gripper
            action_gripper = np.array([0.1])
        elif action_gripper >= 0.5:
            # Open gripper
            action_gripper = np.array([-0.1])
        action = np.concatenate([action[:3], action_gripper])
        return action

    def execute(self, env, observations, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        self.image_buffer = []
        self.detected_positions = {}
        self.tracked_positions = {}
        horizon = self.horizon if self.horizon is not None else 500
        print("\tTask goal: ", symgoal)

        step_executor = 0
        done = False
        success = False 
        print("\tStarting executor for step: ", self.id)
        while not done:
            processed_obs = []
            # Prepare the observation for the policy
            for observation in observations:
                #print("Observation keys: ", observation.keys())
                if self.use_yolo or self.save_data:
                    cubes_xyz = {}
                    objects_pos = observation["objects_pos"]
                    state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                    # agentview_image = np.array(observation["agentview_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                    # wrist_image = np.array(observation["robot0_eye_in_hand_image"].reshape((self.image_size, self.image_size, 3)), dtype=np.uint8)
                    # ee_pos = observation["robot0_eef_pos"]
                    # cubes_obs = {k: np.asarray(v, dtype=np.float32).copy() for k, v in objects_pos.items() if 'cube' in k}
                    # #print("cubes_obs: ", cubes_obs)
                    # #print("Image shape: ", image.shape)
                    # if len(self.detected_positions) >= 3 and not(self.save_data) and self.id in ["Grasp", "Drop"]:
                    #     cubes_xyz = copy.deepcopy(self.detected_positions)
                    # else:
                    #     cubes_xyz = self.yolo_estimate(image1 = agentview_image, 
                    #                                 image2 = wrist_image, 
                    #                                 save_video=self.save_data, 
                    #                                 cubes_obs=cubes_obs,
                    #                                 ee_pos=ee_pos)
                        #if len(self.tracked_positions) >= 3:
                        #    cubes_xyz = copy.deepcopy(self.tracked_positions)

                    
                    obs = self.get_object_obs(env, objects_pos, cubes_xyz, symgoal[0], symgoal[1], relative_obs=self.oracle)
                    # obs_copy = copy.deepcopy(obs)
                    # if len(cubes_xyz) > 0:
                    #     if self.use_yolo:
                    #         o = self.insert_yolo_estimate(obs_copy, cubes_xyz=cubes_xyz, obj_to_pick=symgoal[0], place_to_drop=symgoal[1])
                    #         o = self.track_positions(state=state, ee_pos=ee_pos, obs=o, symgoal=symgoal)
                    #         # insert o into obs
                    #         obs = o
                processed_obs.append(obs)
            processed_obs = np.array(processed_obs)
            if self.oracle:
                processed_obs = self.prepare_obs(processed_obs, action_step=self.id)
            processed_obs = np.array([processed_obs])  # scale to mm
            # if obs_base:
            #     obs = self.obs_base_from_info(info)
            #print(processed_obs)
            #print(processed_obs.shape)
            # create obs dict
            #print("Observation, ", obs)
            np_obs_dict = {
                'obs': processed_obs.astype(np.float32)
            }
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=self.device))
            # run policy
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            actions = np_action_dict['action']/1000.0  # scale to m
            if len(actions[0][0]) < 4:
                for index in self.nulified_action_indexes:
                    actions = np.insert(actions, index, 0, axis=2)
            observations = []
            for action in actions[0]:
                #print("Action: ", action)
                action = self.map_gripper(action)
                _, _, done, info = env.step(action)
                if render:
                    env.render()
                obs = env._get_observations()
                objects_pos = self.detector.get_all_objects_pos()
                obs['objects_pos'] = objects_pos
                observations.append(obs)
            if done:
                print("Environment terminated")
            step_executor += 1
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            success = self.Beta(state, symgoal)
            success = success #or info['is_success']
            if success:
                done = True
            if step_executor > horizon:
                print("Reached executor horizon")
                done = True 
        if self.save_data:
            self.save_csv_yolo(output_path=f"{self.id}_dualcam_{self.count}.csv")
            self.count += 1
        return observations, success
