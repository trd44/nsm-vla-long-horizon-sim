import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
import robosuite_task_zoo
from datetime import datetime
import gymnasium as gym
# import gym
import cv2
import sys

from robosuite.wrappers import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from robosuite.utils.detector import HanoiDetector, KitchenDetector, NutAssemblyDetector
from robosuite.wrappers.nutassembly.object_state import AssembleStateWrapper
from robosuite.wrappers.kitchen.object_state import KitchenStateWrapper
from robosuite.wrappers.hanoi.object_state import HanoiStateWrapper
from robosuite.wrappers.nutassembly.vision import AssembleVisionWrapper
from robosuite.wrappers.kitchen.vision import KitchenVisionWrapper
from robosuite.wrappers.hanoi.vision import HanoiVisionWrapper

from planning.planner import *
from planning.executor import *

from ultralytics import YOLO
from roboflow import Roboflow
import joblib
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*env.*is deprecated.*")


object_state_wrapper = {"Hanoi": HanoiStateWrapper, "KitchenEnv": KitchenStateWrapper, "NutAssembly": AssembleStateWrapper}
vision_wrapper = {"Hanoi": HanoiVisionWrapper, "KitchenEnv": KitchenVisionWrapper, "NutAssembly": AssembleVisionWrapper}

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
                 yolo_model=None,
                 regressor_model=None,
                 map_id_semantic=None,
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
        self.yolo_model = yolo_model
        self.regressor_model = regressor_model
        self.render = render
        self.randomize = randomize
        self.noise_std_factor = noise_std_factor

        # Set up the environment
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.recorded_eps = 0

        # Init buffer
        self.data_buffer = dict()
        self.action_steps = []
        self.checkpoint = None
        if args.checkpoints > 0:
            self.checkpoint = 0
        self.df_metrics = pd.DataFrame(columns=['wx', 'wy', 'wz', 'pred_x', 'pred_y', 'pred_z', 'error'])
        self.bboxes_centers = []
        self.map_id_semantic = map_id_semantic
        self.reset()

    def get_plan(self):
        """
        Returns the plan
        """
        # Detect init state
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        # Filter and keep only the predicates that are in planning_predicates[args.env] and are True and map them to the PDDL format
        init_predicates = {predicate: True for predicate in state.keys() if state[predicate] and predicate.split("(")[0] in planning_predicates[self.args.env]}
        print("Initial predicates: ", init_predicates)
        # Usage

        add_predicates_to_pddl(pddl_path, init_predicates)
        # Generate a plan
        self.plan, _ = call_planner(pddl_path, mode=planning_mode[self.args.env])
        print("Task demonstrated: ", self.plan)

    def get_task(self):
        """
        Returns the task
        """
        return self.plan[self.operator_step]

    def operator_to_function(self, operation):
        """
        A function that maps the operation to the corresponding function
        returns the function, the semantic description of the operation and the goal
        """
        map_color = {"cube1": "blue", "cube2": "red", "cube3": "green",}
        operation = operation.lower().split(' ')
        print("Operation: ", operation[0])
        if not(args.vla):
            if len(operation) == 3:
                self.env.set_task((operation[1], operation[2]))
            elif 'turn' in operation[0]:
                self.env.set_task((None, None))
        if 'pick' in operation[0]:
            if self.args.env == "Hanoi":
                return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
            else:
                return self.pick, f'pick {operation[1]} from {operation[2]}', operation[1]
        elif 'place' in operation[0]:
            if self.args.env == "Hanoi":
                return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
            else:
                return self.place, f'place {operation[1]} on {operation[2]}', (operation[1], operation[2])
        elif 'turn-on' in operation[0]:
            return self.turn_on_button, f'switch on button', "button"
        elif 'turn-off' in operation[0]:
            return self.turn_off_button, f'switch off button', "button"
        else:
            return None

    def pixel_to_world_dual(self, cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        # Load linear Regression models for cube positions
        models_dual = self.regressor_model
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        features = np.array([[cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z]], dtype=np.float64)
        x = reg_x_dual.predict(features)[0]
        y = reg_y_dual.predict(features)[0]
        z = reg_z_dual.predict(features)[0]
        return x*1000., y*1000., z*1000.

    def yolo_estimate(self, obs_step):
        # Resize the image to fit YOLO input requirements

        cubes_predicted_xyz = {}
        confidence_agent = {}
        if "agentview" not in obs_step or "wrist_image" not in obs_step:
            print("No images provided for YOLO estimation.")
            return cubes_predicted_xyz
        agentview = obs_step["agentview"]
        wrist_image = obs_step["wrist_image"]
        # Get the predictions
        agentview_results = self.yolo_model(agentview, verbose=False)[0]
        wrist_results = self.yolo_model(wrist_image, verbose=False)[0]
        #print(len(agentview_results.boxes), "objects detected in agentview")
        #print(len(wrist_results.boxes), "objects detected in wristview")
        # print confidence and class name for each detected object
        # for pred in agentview_results.boxes:
        #     cls_id = int(pred.cls)
        #     cls = self.yolo_model.names[cls_id]
        #     conf = pred.conf
            # print(f"Agentview detected: {cls} with confidence {float(conf):.2f}")
            # # display the image with bounding boxes
            # displayed_image = agentview_results.plot()
            # cv2.imshow('agentview', cv2.cvtColor(displayed_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
        # for pred in wrist_results.boxes:
        #     cls_id = int(pred.cls)
        #     cls = self.yolo_model.names[cls_id]
        #     conf = pred.conf
            # print(f"Wristview detected: {cls} with confidence {float(conf):.2f}")
            # # display the image with bounding boxes
            # displayed_image = wrist_results.plot()
            # cv2.imshow('wristview', cv2.cvtColor(displayed_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
        #print(obs_step.keys())
        objects_info = obs_step["objects_pos"]  # dict with object names as keys and positions as values
        for pred in agentview_results.boxes:
            confidence_wrist = {}
            cls_id = int(pred.cls)
            cls = self.yolo_model.names[cls_id]
            #print("1: ", cls)
            x, y, w, h = pred.xywhn.tolist()[0]
            conf = pred.conf
            # Convert normalized coordinates to pixel coordinates
            x = int(x * agentview.shape[1])
            y = int(y * agentview.shape[0])
            w = int(w * agentview.shape[1])
            h = int(h * agentview.shape[0])

            # Convert to pixel coordinates
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            if cls_id not in confidence_agent:
                confidence_agent[cls_id] = conf
            else:
                if conf > confidence_agent[cls_id]:
                    confidence_agent[cls_id] = conf
                else:
                    continue

            # Get the ground truth position of the object
            if cls in objects_info:
                ground_truth_xyz = objects_info[cls]
                ee_pos = objects_info["gripper"]

                found_match = False
                for pred in wrist_results.boxes:
                    cls_id2 = int(pred.cls)
                    cls2 = self.yolo_model.names[cls_id2]
                    if cls_id2 not in confidence_wrist:
                        confidence_wrist[cls_id2] = pred.conf
                    else:
                        if pred.conf > confidence_wrist[cls_id2]:
                            confidence_wrist[cls_id2] = pred.conf
                        else:
                            continue
                    #print("2: ", cls2)
                    if cls_id2 == cls_id:
                            found_match = True
                            x_cam2, y_cam2, w_cam2, h_cam2 = pred.xywhn.tolist()[0]
                            conf_cam2 = pred.conf
                            # Convert normalized coordinates to pixel coordinates
                            x_cam2 = int(x_cam2 * wrist_image.shape[1])
                            y_cam2 = int(y_cam2 * wrist_image.shape[0])
                            w_cam2 = int(w_cam2 * wrist_image.shape[1])
                            h_cam2 = int(h_cam2 * wrist_image.shape[0])

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
                cubes_predicted_xyz.update({cls: predicted_xyz})
            else:
                # Print error message, no second image
                print("No second image provided for dual camera regression.")
            # Print the predicted and ground truth positions
            #print(f"Predicted: {predicted_xyz}, Ground Truth: {ground_truth_xyz}", "Error: ", np.linalg.norm(np.array(predicted_xyz) - np.array(ground_truth_xyz)))

        return cubes_predicted_xyz

    def reset(self, seed=None):
        """
        The reset function that resets the environment
        """
        if self.args.checkpoints > 0:
            if self.recorded_eps // self.args.checkpoints > self.checkpoint:
                print("\nSAVING CHECKPOINT: ", self.checkpoint)
                self.data_buffer = dict()
                self.action_steps = []
                self.checkpoint += 1
        self.operator_step = 0
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.task_buffer = list()
        try:
            obs = self.env.reset()
        except AttributeError:
            obs = self.env.env.reset()
        except AttributeError:
            obs = self.env.reset()
        self.env.sim.forward()
        self.get_plan()
        return obs

    def run_trajectory(self, obs):
        """
        Runs the trajectory
        """
        done = False
        for operation in self.plan:
            try:
                function, self.task, goal = self.operator_to_function(operation)
                print(f'Performing task: {self.task}, with goal: {goal}')
                done, obs = function(obs, goal)
            except:
                print(f'Performing task: {self.task}, with goal: {goal}')
                done = True
            if not(done):
                print("Failed to perform task")
                return False
            print("Successful task")
        print("Successful episode?: ", done)
        return done

    def save_trajectory(self):
        """
        Saves the trajectory to the buffer
        """
        if self.args.vla:
            episode = []
            for step in self.action_steps:
                if step in self.episode_buffer.keys():
                    for i in range(0,len(self.episode_buffer[step]),2): 
                        # Convert action from dx,dy,dz,gripper to dx,dy,dz,d_roll,d_pitch,d_yaw,gripper
                        action_4dim = self.episode_buffer[step][i]
                        action = np.concatenate((action_4dim[:3], np.zeros(3), action_4dim[3:]))

                        raw_image = self.episode_buffer[step][i+1]
                        image = np.array(raw_image).reshape((256, 256, 3))
                        
                        episode.append({
                            'image': image,
                            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                            # 'state': np.asarray(np.random.rand(10), dtype=np.float32),
                            'action': action,
                            'language_instruction': step,
                        })
            np.save(f'data/kitchen_env/episode_{self.recorded_eps}.npy', episode)
        
        else:
            for step in self.action_steps:
                print(step)
                print("\n\n\n")
                if step in self.episode_buffer.keys():
                    if step not in self.data_buffer.keys():
                        self.data_buffer[step] = [(self.episode_buffer[step], self.task_buffer)]
                    else:
                        self.data_buffer[step].append((self.episode_buffer[step], self.task_buffer))
            self.zip_buffer(self.args.traces)
        self.recorded_eps += 1
        self.save_csv_yolo(output_path=os.path.join(self.args.yolo_data, "yolo_data_{}.csv".format(self.recorded_eps)))
        obs = self.reset()
        return obs

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not self.bboxes_centers:
            print("No bounding boxes data to save.")
            return
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        print(f"YOLO data saved at {output_path}")

    def zip_buffer(self, dir_path):
        # Decompose the data buffer into action steps
        for step in self.action_steps:
            if step not in self.data_buffer.keys():
                continue
            # Convert the data buffer to bytes
            data_bytes = pickle.dumps(self.data_buffer[step])
            if self.args.checkpoints > 0:
                file_path = dir_path + str(self.checkpoint) + '/' + step + '.zip'
                os.makedirs(dir_path + str(self.checkpoint), exist_ok=True)
            else:
                file_path = dir_path + step + '.zip'
            # Write the bytes to a zip file
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)

    def filter_obs(self, obs, action_step="main"):
        if action_step == "main":
            return obs
        elif action_step == "pick":
            return np.concatenate([obs[3:6], [obs[-1]]])
        elif action_step == "place":
            return obs[-4:]
        elif action_step == "turn_on":
            return obs[:3]
        elif action_step == "turn_off":
            return obs[:3]
        elif action_step == "reach_pick":
            return obs[3:6]
        elif action_step == "reach_place":
            return obs[-4:-1]


    def record_step(self, obs, action, next_obs, state_memory, new_state, sym_action="MOVE", action_step="main", reward=-1.0, done=False, info=None, goal=None):
        """
        Records the step
        """
        if self.args.render and (self.vision_based):
            # display the image (obs)
            print(obs.shape)
            print(type(obs))
            cv2.imshow('image', obs)
            cv2.waitKey(1)
        if self.vision_based or self.args.vla:
            # convert obs type to np.uint8
            obs = np.array(obs, dtype=np.uint8)
            next_obs = np.array(next_obs, dtype=np.uint8)
        if self.args.train_yolo:
            # get all objects positions and the end effector position, store all obs within a dict
            if self.detector is None:
                print("Detector is None")
                return None
            objects_pos = self.detector.get_all_objects_pos()
            num_images = len(obs) // (self.args.size * self.args.size * 3)
            next_obs = {}
            for i in range(num_images):
                next_obs[f'image_{i}'] = np.array(obs[i * self.args.size * self.args.size * 3:(i + 1) * self.args.size * self.args.size * 3].reshape((self.args.size, self.args.size, 3)), dtype=np.uint8)
            #next_obs['objects_pos'] = np.asarray(objects_pos.copy(), dtype=np.float32)
            next_obs['objects_pos'] = next_obs["objects_pos"] = {k: np.asarray(v, dtype=np.float32).copy()
                           for k, v in objects_pos.items()}
            #print("pot pos from sim: ", next_obs['objects_pos']["pot"])
        if self.args.use_yolo:
            if len(obs) == 2:
                obs = obs[0]
            # get all objects positions and the end effector position, store all obs within a dict
            if self.detector is None:
                print("Detector is None")
                return None
            objects_pos = self.detector.get_all_objects_pos()
            #print("EE pos from sim: ", objects_pos["gripper"])
            #print("pot pos from sim: ", objects_pos.get("pot", None))
            num_images = len(obs) // (self.args.size * self.args.size * 3)
            obs_dict_temp = {}
            obs_dict = {}
            for i in range(num_images):
                obs_dict_temp[f'image_{i}'] = np.array(obs[i * self.args.size * self.args.size * 3:(i + 1) * self.args.size * self.args.size * 3].reshape((self.args.size, self.args.size, 3)), dtype=np.uint8)
            obs_dict["agentview"] = cv2.cvtColor(cv2.flip(obs_dict_temp["image_0"].reshape(256, 256, 3), 0), cv2.COLOR_RGB2BGR)  # Assuming the first image is from the agentview camera
            obs_dict["wrist_image"] = cv2.cvtColor(cv2.flip(obs_dict_temp["image_1"].reshape(256, 256, 3), 0), cv2.COLOR_RGB2BGR)  # Assuming the second image is from the wrist camera
            obs_dict['objects_pos'] = objects_pos
            cubes_predicted_xyz = self.yolo_estimate(obs_dict)
            #print(self.obj_to_pick, self.place_to_drop)
            #print("Predicted positions: ", cubes_predicted_xyz)
            if self.obj_to_pick in cubes_predicted_xyz.keys():
                predicted_pos_to_pick = cubes_predicted_xyz[self.obj_to_pick]
                world_pos_to_pick = copy.deepcopy(obs[7:10])
                obs[7:10] = np.array(predicted_pos_to_pick)
                self.df_metrics = pd.concat(
                    [
                        self.df_metrics,
                        pd.DataFrame([{
                            'wx': world_pos_to_pick[0],
                            'wy': world_pos_to_pick[1],
                            'wz': world_pos_to_pick[2],
                            'pred_x': predicted_pos_to_pick[0],
                            'pred_y': predicted_pos_to_pick[1],
                            'pred_z': predicted_pos_to_pick[2],
                            'error': np.linalg.norm(np.array(predicted_pos_to_pick) - np.array(world_pos_to_pick))
                        }])
                    ],
                    ignore_index=True
                )
                self.df_metrics["diff_x"] = self.df_metrics["wx"] - self.df_metrics["pred_x"]
                self.df_metrics["diff_y"] = self.df_metrics["wy"] - self.df_metrics["pred_y"]
                self.df_metrics["diff_z"] = self.df_metrics["wz"] - self.df_metrics["pred_z"]
            if self.place_to_drop in cubes_predicted_xyz.keys():
                predicted_pos_to_drop = cubes_predicted_xyz[self.place_to_drop]
                world_pos_to_drop = copy.deepcopy(obs[4:7])
                obs[4:7] = np.array(predicted_pos_to_drop)
                # update self.df_metrics
                self.df_metrics = pd.concat(
                    [
                        self.df_metrics,
                        pd.DataFrame([{
                            'wx': world_pos_to_drop[0],
                            'wy': world_pos_to_drop[1],
                            'wz': world_pos_to_drop[2],
                            'px': predicted_pos_to_drop[0],
                            'py': predicted_pos_to_drop[1],
                            'pz': predicted_pos_to_drop[2],
                            'error': np.linalg.norm(np.array(predicted_pos_to_drop) - np.array(world_pos_to_drop))
                        }])
                    ],
                    ignore_index=True
                )
                self.df_metrics["diff_x"] = self.df_metrics["wx"] - self.df_metrics["pred_x"]
                self.df_metrics["diff_y"] = self.df_metrics["wy"] - self.df_metrics["pred_y"]
                self.df_metrics["diff_z"] = self.df_metrics["wz"] - self.df_metrics["pred_z"]
        # else:
        #     if obs.shape != next_obs.shape:
        #         return state_memory
        if self.args.vla:
            action_step = self.task
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if self.args.filter_obs:
            obs = self.filter_obs(obs, action_step)
            next_obs = self.filter_obs(next_obs, action_step)
        if action_step not in self.episode_buffer.keys():
            # self.episode_buffer[action_step] = [obs, action, next_obs] # Why 3 things?
            self.episode_buffer[action_step] = [action, next_obs]
        else:
            self.episode_buffer[action_step] += [action, next_obs]
        return state_memory
        # else:
        #     keypoint = self.relative_obs_mapping(goal)
        #     transition = (obs, action, next_obs, keypoint, reward, done)
        #     if action_step not in self.action_steps:
        #         self.action_steps.append(action_step)
        #     if action_step not in self.episode_buffer.keys():
        #         self.episode_buffer[action_step] = [transition]
        #     else:
        #         self.episode_buffer[action_step].append(transition)
        #     self.task_buffer.append(self.task)
        #     return state_memory

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
        if not self.args.ee:
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
        while z_pos < ref_z:
            z_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])[2]
            dist_z = abs(ref_z - z_pos)
            dist_z = self.cap([dist_z])
            action = 5*np.concatenate([[0, 0], dist_z, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 100:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 400:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(goal_str)]:
            action = np.asarray([0,0,0,1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="pick", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(goal_str)]:
            action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.1, 3), [0]])
            action = 5*self.cap(action)
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
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

        print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(goal_str)]:
            #distance = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True)['over(gripper,stove)']
            #print("Distance: ", distance)
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
            elif "serving" in goal_str:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0.002, -0.05, 0])
            # elif self.args.env == "NutAssembly" and goal_str == "roundpeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, 0.05, 0])
            # elif self.args.env == "NutAssembly" and goal_str == "squarepeg":
            #     object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+ np.array([-0.01, 0.025, 0])
            #elif "stove" in goal_str:
            #    object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0.01, 0, 0])
            #elif "pot" in goal_str:
            #    object_pos = np.asarray(self.env.sim.data.body_xpos[goal])+np.array([0, -0.01, 0])
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2] #+ 0.05
            dist_xy_plan = self.cap(dist_xy_plan)
            action = 5*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 5*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="reach_place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 500:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("Moving down picked object on place to drop...")
        while not state['on({},{})'.format(pick_str, goal_str)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            #object_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            place_pos = np.asarray(self.env.sim.data.body_xpos[goal])
            dist_z_axis = [- (gripper_pos[2] - place_pos[2])]
            dist_z_axis = self.cap(dist_z_axis)
            #print("Distance to place: ", dist_z_axis, dist_xy_plan)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, self.noise_std_factor*np.linalg.norm(dist_z_axis), 1), [0]])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 200:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        print("dropping object...")
        while not(state['open_gripper(gripper)']):#state['grasped({})'.format(goal_str)]:
            action = np.asarray([0,0,0,-1])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            reset_step_count += 1
            if reset_step_count > 30:
                return False, obs
        reset_step_count = 0
        self.env.time_step = 0

        #print("Resetting gripper")
        # First move up
        for _ in range(15):
            action = np.array([0, 0, 1, 0])
            action = self.to_osc_pose(action)
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="place", goal=goal_str, info=info)
            if self.state_memory is None:
                return False, obs
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_on", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
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
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_step(obs, action, next_obs, self.state_memory, next_state, action_step="turn_off", goal=goal_str, info=info)
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
    parser.add_argument('--episodes', type=int, default=int(200), help='Number of trajectories to record')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the recording')
    parser.add_argument('--vision', action='store_true', help='Use vision based observation')
    parser.add_argument('--train_yolo', action='store_true', help='Store observation to train train_yolo+regressor format')
    parser.add_argument('--use_yolo', action='store_true', help='Use yolo+regressor for the object pose estimation as observation')
    parser.add_argument('--relative_obs', action='store_true', help='Use relative gripper-goal observation')
    parser.add_argument('--vla', action='store_true', help='Store the data in VLA friendly format')
    parser.add_argument('--size', type=int, default=256, help='Size of the observation')
    parser.add_argument('--checkpoints', type=int, default=0, help='Saves the data every n episodes, and resets the buffer')
    parser.add_argument('--ee', action='store_true', help='Use end effector observation, without rotations')
    parser.add_argument('--filter_obs', action='store_true', help='Filter the observations relevant to the operators')
    parser.add_argument('--rnd_reset', action='store_true', help='Random reset for Hanoi env')

    args = parser.parse_args()
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
    args.yolo_data = args.env_dir + '/yolo_data/'
    os.makedirs(args.env_dir, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)
    os.makedirs(args.yolo_data, exist_ok=True)

    # Load the controller config
    if not args.ee:
        controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    else:
        controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
    # Create the environment
    if args.env == 'Hanoi':
        env = suite.make(
            args.env,
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=args.vision or args.train_yolo or args.use_yolo,
            use_object_obs=not(args.vision) and not(args.train_yolo) and not(args.use_yolo),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.size,
            camera_widths=args.size,
            random_reset = args.rnd_reset
        )
    else:
        env = suite.make(
            args.env,
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=100000000,
            use_camera_obs=args.vision or args.train_yolo or args.use_yolo,
            use_object_obs=not(args.vision) and not(args.train_yolo) and not(args.use_yolo),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=args.size,
            camera_widths=args.size,
        )


    env.reset()

    # Wrap the environment
    yolo_model = None
    regressor_model = None
    if args.env == 'Hanoi':
        detector = HanoiDetector(env)
        pddl_path = './planning/PDDL/hanoi/'
        yolo_model = YOLO("PDDL/yolo_dual_cam.pt")
        regressor_model = joblib.load("data/hanoi_dual_cam_calibration_models.pkl")
    elif args.env == 'KitchenEnv':
        detector = KitchenDetector(env)
        pddl_path = './planning/PDDL/kitchen/'
        yolo_model = YOLO("PDDL/yolo_kitchen.pt")
        regressor_model = joblib.load("data/kitchen_dual_cam_calibration_models.pkl")
    elif args.env == 'NutAssembly':
        detector = NutAssemblyDetector(env)
        pddl_path = './planning/PDDL/nut_assembly/'
        yolo_model = YOLO("PDDL/yolo_nutassembly.pt")
        regressor_model = joblib.load("data/nutassembly_dual_cam_calibration_models.pkl")
    env = GymWrapper(env, proprio_obs=not(args.vision) and not(args.train_yolo) and not(args.use_yolo))
    if args.vla:
        print("Using VLA friendly format")
        env = VisualizationWrapper(env)
    elif not args.vision and not args.train_yolo and not args.use_yolo:
        print("Using object based observation")
        env = object_state_wrapper[args.env](env)
    else:
        print("Using vision based observation")
        env = VisualizationWrapper(env)
        env = vision_wrapper[args.env](env, patch_task=args.vision, image_size=args.size)

    env = RecordDemos(env, args.vision, detector, pddl_path, args, yolo_model=yolo_model, regressor_model=regressor_model, render=args.render, randomize=True)

    os.makedirs(f'data/kitchen_env', exist_ok=True)

    # Run the recording of the demonstrations
    episode = 1
    while env.recorded_eps < args.episodes: 
        obs = env.reset()
        print("Episode: {}".format(episode))
        keys = list(env.data_buffer.keys())
        done = env.run_trajectory(obs)
        if done:
            obs = env.save_trajectory()
        episode += 1
        print("Number of recorded episodes: {}".format(env.recorded_eps))
        if args.use_yolo:
            print("Metric df: \n", env.df_metrics.describe())
            # Compute the error metrics
            mean_error_x = env.df_metrics["diff_x"].mean()
            mean_error_y = env.df_metrics["diff_y"].mean()
            mean_error_z = env.df_metrics["diff_z"].mean()
            std_error_x = env.df_metrics["diff_x"].std()
            std_error_y = env.df_metrics["diff_y"].std()
            std_error_z = env.df_metrics["diff_z"].std()
            print(f"Mean Error X: {mean_error_x}, Std Error X: {std_error_x}")
            print(f"Mean Error Y: {mean_error_y}, Std Error Y: {std_error_y}")
            print(f"Mean Error Z: {mean_error_z}, Std Error Z: {std_error_z}")
            print(f"Overall Mean Error: {(mean_error_x**2 + mean_error_y**2 + mean_error_z**2)**0.5}")
            print("\n\n")
        print("\n\n")