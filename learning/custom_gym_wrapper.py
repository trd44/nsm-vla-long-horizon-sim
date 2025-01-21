import copy
import time
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import csv
from tarski import fstrips as fs
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from utils import *
from VLM.LlmApi import chat_completion

class OperatorWrapper(gym.Wrapper):
    def __init__(self, env:MujocoEnv, grounded_operator:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict, domain:str, rl_algo:str, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env)
        self.detector = load_detector(config=config, domain=domain, env=env)
        self.grounded_operator = grounded_operator
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = executed_operators
        self.config = config
        self.domain = domain
        self.rl_algo_name = rl_algo
        self.curr_subgoal = curr_subgoal
        self.last_subgoal_successes, self.subgoal_reward_shaping_fn_mapping = self._init_subgoal_dicts()
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        self.time_step = 0
        self.episode = -1
        self.record_rollouts = record_rollouts
        self.reward_range = (-float('inf'), 0)

        if self.record_rollouts:
            self.rollout_save_dir = f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning'][self.rl_algo_name]['seed']}"
            _, largest_file_number = find_file_with_largest_number(self.rollout_save_dir, 'rollout')
            if largest_file_number is None:
                largest_file_number = 0
            self.rollout_save_path = f"{self.rollout_save_dir}/rollout_{largest_file_number+1}.csv"
            self.csv_file = open(self.rollout_save_path, 'w')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['gripper_x', 'gripper_y', 'gripper_z', 'closest_x', 'closest_y', 'closest_z', 'collision_penalty', 'total_reward', 'num_achieved_subgoals', 'done', 'timestep', 'episode'])
        
    def _init_subgoal_dicts(self) -> Tuple[OrderedDict[str, bool], OrderedDict[str, Callable]]:
        """initialize the subgoal dictionaries for recording the subgoal successes of the last step and the mapping between subgoals and their reward shaping functions
        Returns:
            Tuple[Dict[str, bool], Dict[str, Callable]]: the subgoal success dictionary and the subgoal reward shaping function mapping
        """
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        subgoal_success_dict = OrderedDict()
        subgoal_reward_shaping_fn_mapping = OrderedDict()
        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            subgoal_success_dict[effect.pddl_repr()] = False
            subgoal_reward_shaping_fn_mapping[effect.pddl_repr()] = None
        return subgoal_success_dict, subgoal_reward_shaping_fn_mapping
    
    def set_subgoal_reward_shaping_fn(self, effect:Union[str, fs.SingleEffect], fn:Callable):
        """Set the reward shaping function for the subgoal

        Args:
            effect (str): the subgoal
            fn (Callable): the reward shaping function
        """
        if isinstance(effect, fs.SingleEffect):
            effect = effect.pddl_repr()
        self.subgoal_reward_shaping_fn_mapping[effect] = fn


    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """step function that steps the environment and computes the reward based on the observation with semantics

        Args:
            action (array): 7 dimensional array representing the action to take

        Returns:
            Tuple[np.array, float, bool, bool, dict]: the observation, reward, done, truncated, and info
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        action = self._discretize_gripper_action(action)
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env.unerapped)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        penalties, collision_points = self.collision_penalty(obs_with_semantics)
        # save reward and penalties separately
        self.episode_r_shaping += reward
        self.episode_collision_penalty += sum(penalties)
        self.episode_num_collisions += len(collision_points)
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = self.last_subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        for subgoal, success in self.last_subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(self.last_subgoal_successes.values())
        # episode is done also if the current subgoal we are focusing on is achieved
        done = self.last_subgoal_successes[self.curr_subgoal.pddl_repr()] or done

        # save info to the csv file, one row per each collision point
        # save every 10 episodes every 10 timesteps
        if self.record_rollouts and self.time_step % 100 == 0:
            for collision_point, penalty in zip(collision_points, penalties):
                g_pos = obs_with_semantics['gripper1_pos']
                self.csv_writer.writerow([g_pos[0], g_pos[1], g_pos[2], collision_point[0], collision_point[1], collision_point[2], penalty, reward, sum(list(self.last_subgoal_successes.values())), reward + sum(penalties) == 0, self.time_step, self.episode])
            self.csv_file.flush()
        
        self.time_step += 1
        return obs, reward + sum(penalties), done, truncated, info

    def reset(self, **kwargs):
        reset_success = False
        while not reset_success:
            # first, reset the environment to the very beginning
            try: # kwargs include the seed
                obs, info = self.env.reset(**kwargs)
            except:
                obs = self.env.reset(**kwargs)
                info = {}
            # second, execute the executors that should be executed before the operator to learn
            reset_success = True
            for op, ex in self.executed_operators.items():
                ex_success = ex.execute(self.detector, op)
                if not ex_success:
                    reset_success = False
                    break
        # reset the episode rewards and penalties
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        # reset the success of the subgoals
        self.last_subgoal_successes, _ = self._init_subgoal_dicts()
        self.detector.update_obs()
        self.episode += 1
        # reset the infos
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = self.last_subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        for subgoal, success in self.last_subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(self.last_subgoal_successes.values())
        info['is_success'] = info['goal_success'] # for compatibility with the stable_baselines3's update_info_buffer
        info['episode'] = { # for compatibility with the stable_baselines3's update_info_buffer
            'ep_cumu_r_shaping': self.episode_r_shaping,
            'ep_cumu_col_penalty': self.episode_collision_penalty,
            'ep_cumu_collisions': self.episode_num_collisions,
            'subgoal_success': self.last_subgoal_successes[self.curr_subgoal.pddl_repr()],
        }

        return obs, info

    def check_effect_satisfied(self, effect:fs.SingleEffect, binary_obs:dict) -> bool:
        """check if the effect is satisfied in the observation

        Args:
            effect (fs.SingleEffect): the effect to check
            binary_obs (dict): the binary observation with semantics

        Returns:
            bool: True if the effect is satisfied, False otherwise
        """
        effect_name = effect.atom.pddl_repr()# e.g. `free gripper1``
        # check if effect is negated
        if isinstance(effect, fs.DelEffect): # effect is negated e.g., `not (free gripper1)`
            return not binary_obs[effect_name]
        return binary_obs[effect_name]
    
    def check_duplicate_grasp_effects(self) -> bool:
        """check if the grounded operator has both `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` effects. If so, they should count as one effect

        Returns:
            bool: True if both effects are present, False otherwise
        """
        effects:List[fs.SingleEffect] = self.grounded_operator.effects
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        not_free_gripper_effect_present = False
        exclusively_occupying_gripper_effect_present = False
        for effect in effects:
            if effect.atom.pddl_repr() == 'free gripper1' and isinstance(effect, fs.DelEffect): # `not (free gripper1)` is present
                not_free_gripper_effect_present = True
            elif effect.atom.predicate.name == 'exclusively-occupying-gripper':
                exclusively_occupying_gripper_effect_present = True
        return not_free_gripper_effect_present and exclusively_occupying_gripper_effect_present
    
    def reward(self, reward:float):
        """reward the agent with a reward. Overwrites the reward function in the environment. Duplicates part of the logic in :meth:`step` to compute the reward based on the observation with semantics in case the reward function is called outside of the step function by the learning algorithm

        Args:
            reward (float): the reward to give to the agent
        """
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        penalties, _= self.collision_penalty(obs_with_semantics)
        return reward + sum(penalties)

    def compute_reward(self, numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            numeric_obs_with_semantics: the observation in which the keys have semantics and the values are arrays of numeric values
            binary_obs: the binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward between -1, 0
        """
        
        # there is a step cost of -1 regardless
        step_cost = -1
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect

        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        num_effects = len(effects) if not duplicate_grasp_effects else len(effects) - 1
         
        sub_goal_reward = 0
        num_subgoals_achieved = 0
        last_useful_reward_shaping_fn = None
        # reset subgoal successes
        for effect in effects:
            self.last_subgoal_successes[effect.pddl_repr()] = False

        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if self.check_effect_satisfied(effect, binary_obs):
                self.last_subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                num_subgoals_achieved += 1
                sub_goal_reward += 1/num_effects
                last_useful_reward_shaping_fn = self.subgoal_reward_shaping_fn_mapping.get(effect.pddl_repr())
            else:
                self.last_subgoal_successes[effect.pddl_repr()] = False
                llm_reward_shaping_fn = self.subgoal_reward_shaping_fn_mapping.get(effect.pddl_repr())
                if llm_reward_shaping_fn is None:
                    # reward shaping for this subgoal has not been specifically set yet. Use the previously useful reward shaping function that helped the robot achieve the last subgoal
                    llm_reward_shaping_fn = last_useful_reward_shaping_fn
                    self.subgoal_reward_shaping_fn_mapping[effect.pddl_repr()] = llm_reward_shaping_fn
                try: # the llm reward shaping function may not be error-free. In that case, raise an error
                    sub_goal_reward += llm_reward_shaping_fn(numeric_obs_with_semantics, f"({effect.pddl_repr()})") * 1/num_effects
                except Exception as e:
                    raise Exception(f"Error in the LLM reward shaping function for the effect {effect.pddl_repr()}: {e}")
                break # return the reward as soon as one effect is not satisfied. Assume later effects are at 0% progress therefore would get a shaping reward of 0 anyway.
        
        return step_cost + sub_goal_reward
    
    def collision_penalty(self, numeric_obs_with_semantics:dict) -> Tuple[float, List[np.array]]:
        """Penalize the robot for getting too close to objects it is not supposed to collide with

        Args:
            numeric_obs_with_semantics: the observation in which the keys have semantics and the values are arrays of numeric values

        Returns:
            penalties: a list of penalties for getting too close to objects
            collision_points: a list of 3D collision points
        """
        # find objects that the robot is allowed to collide with. These are objects that the robot grasps either in the precondition or the effects of the grounded operator
        allowed_objects = []
        collision_threshold = 0.01 # getting closer than this distance will incur a penalty
        for effect in self.grounded_operator.effects:
            if effect.atom.predicate.name == 'exclusively-occupying-gripper':
                for arg in effect.atom.subterms:
                    # find the parameter that's not the gripper
                    if arg.name != 'gripper1':
                        allowed_objects.append(arg.name)
        for condition in self.grounded_operator.precondition.subformulas:
            if not hasattr(condition, 'connective') and condition.predicate.name == 'exclusively-occupying-gripper':
                for arg in condition.subterms:
                    # find the parameter that's not the gripper
                    if arg.name != 'gripper1':
                        allowed_objects.append(arg.name)
        # find the objects that the robot is close to
        penalties = []
        collision_points = []
        for key, obs in numeric_obs_with_semantics.items():
            if 'collision_dist' in key and not any(obj in key for obj in allowed_objects):
                if obs[0] <= collision_threshold: # the first element is the collision distance.
                    penalties.append(-1/(obs[0]+0.001)) # the closer the robot gets to the object, the higher the penalty. Add a small value to avoid division by zero
                    collision_points.append(obs[1:]) # the rest of the elements are the 3D collision point coordinates
        return penalties, collision_points
    
    def close(self):
        self.csv_file.close()
        return super().close()
    
    def _discretize_gripper_action(self, action:np.array) -> np.array:
        """discretize the gripper opening action into 3 discrete actions: open, close, and do nothing

        Args:
            action (np.array): the action to discretize

        Returns:
            np.array: the discretized action
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        gripper_opening_min = self.env.action_space.low[-1]
        gripper_opening_max = self.env.action_space.high[-1]
        gripper_opening_range = gripper_opening_max - gripper_opening_min
        gripper_close_threshold = gripper_opening_min + gripper_opening_range/3
        gripper_open_threshold = gripper_opening_max - gripper_opening_range/3
        if action[-1] < gripper_close_threshold: # close the gripper
            action[-1] = gripper_opening_min
        elif action[-1] > gripper_open_threshold: # open the gripper
            action[-1] = gripper_opening_max
        else: # do nothing
            action[-1] = 0
        return action
    

class LLMAblatedOperatorWrapper(OperatorWrapper):
    def __init__(self, env:MujocoEnv, grounded_operator:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict, domain:str, rl_algo:str, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env, grounded_operator, executed_operators, config, domain, rl_algo, curr_subgoal, record_rollouts)


    def compute_reward(self, numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            numeric_obs_with_semantics: the observation in which the keys have semantics and the values are arrays of numeric values
            binary_obs: the binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward between -1, 0
        """
        
        # there is a step cost of -1 regardless
        step_cost = -1
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect

        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        num_effects = len(effects) if not duplicate_grasp_effects else len(effects) - 1
         
        sub_goal_reward = 0
        num_subgoals_achieved = 0
        
        # reset subgoal successes
        for effect in effects:
            self.last_subgoal_successes[effect.pddl_repr()] = False

        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if self.check_effect_satisfied(effect, binary_obs):
                self.last_subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                num_subgoals_achieved += 1
                sub_goal_reward += 1/num_effects
            else:
                self.last_subgoal_successes[effect.pddl_repr()] = False
                break # return the reward as soon as one effect is not satisfied. Assume later effects are not satisfied.
        
        return step_cost + sub_goal_reward
    

class CollisionAblatedOperatorWrapper(OperatorWrapper):
    def __init__(self, env:MujocoEnv, grounded_operator:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict, domain:str, rl_algo:str, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env, grounded_operator, executed_operators, config, domain, rl_algo, curr_subgoal, record_rollouts)
        self.reward_range = (-1, 0)

    def reward(self, reward:float):
        """reward the agent with a reward. Overwrites the reward function in the environment. Duplicates part of the logic in :meth:`step` to compute the reward based on the observation with semantics in case the reward function is called outside of the step function by the learning algorithm

        Args:
            reward (float): the reward to give to the agent
        """
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        return reward
    
    def step(self, action) -> Tuple[np.array, float, bool, bool, dict]:
        """step function that steps the environment and computes the reward based on the observation with semantics

        Args:
            action (array): 7 dimensional array representing the action to take

        Returns:
            Tuple[np.array, float, bool, bool, dict]: the observation, reward, done, truncated, and info
        """
        # discretize the gripper opening action into 3 discrete actions: open, close, and do nothing
        action = self._discretize_gripper_action(action)
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        # compute the reward based on the observation with semantics
        obs_with_semantics:dict = self.detector.get_obs()
        binary_obs:dict = self.detector.detect_binary_states(self.env)
        reward = self.compute_reward(obs_with_semantics, binary_obs)
        # save reward and penalties separately
        self.episode_r_shaping += reward
        self.episode_collision_penalty += 0 # assume no collision
        self.episode_num_collisions += 0 # assume no collision
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = self.last_subgoal_successes[self.curr_subgoal.pddl_repr()]
        # save additional subgoal success info for each subgoal
        for subgoal, success in self.last_subgoal_successes.items():
            info[f'{subgoal}_subgoal'] = success
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = all(self.last_subgoal_successes.values())
        # episode is done also if the current subgoal we are focusing on is achieved
        done = self.last_subgoal_successes[self.curr_subgoal.pddl_repr()] or done

        self.time_step += 1
        return obs, reward, done, truncated, info

class CollisionLLMAblatedOperatorWrapper(CollisionAblatedOperatorWrapper):
    def __init__(self, env, grounded_operator, executed_operators, config, domain, rl_algo, curr_subgoal, record_rollouts = False):
        super().__init__(env, grounded_operator, executed_operators, config, domain, rl_algo, curr_subgoal, record_rollouts)
        self.reward_range = (-1, 0)

    def compute_reward(self, numeric_obs_with_semantics:dict, binary_obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            numeric_obs_with_semantics: the observation in which the keys have semantics and the values are arrays of numeric values
            binary_obs: the binary observation whose keys are predicates and values are True/False

        Returns:
            float: the reward between -1, 0
        """
        
        # there is a step cost of -1 regardless
        step_cost = -1
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect

        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        num_effects = len(effects) if not duplicate_grasp_effects else len(effects) - 1
         
        sub_goal_reward = 0
        num_subgoals_achieved = 0
        
        # reset subgoal successes
        for effect in effects:
            self.last_subgoal_successes[effect.pddl_repr()] = False

        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if self.check_effect_satisfied(effect, binary_obs):
                self.last_subgoal_successes[effect.pddl_repr()] = True # record the subgoal success
                num_subgoals_achieved += 1
                sub_goal_reward += 1/num_effects
            else:
                self.last_subgoal_successes[effect.pddl_repr()] = False
                break # return the reward as soon as one effect is not satisfied. Assume later effects are not satisfied.
        
        return step_cost + sub_goal_reward