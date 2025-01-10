import copy
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
import csv
import stable_baselines3
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
    def __init__(self, env:MujocoEnv, grounded_operator:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict, curr_subgoal:fs.SingleEffect, record_rollouts:bool=False):
        super().__init__(env)
        self.detector = load_detector(config=config, env=env)
        self.grounded_operator = grounded_operator
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = executed_operators
        self.config = config
        self.domain = self.config['planning']['domain']
        self.curr_subgoal = curr_subgoal
        self.last_subgoal_successes, self.subgoal_reward_shaping_fn_mapping = self._init_subgoal_dicts()
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        self.time_step = 0
        self.episode = -1
        self.record_rollouts = record_rollouts

        if self.record_rollouts:
            self.rollout_save_dir = f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}"
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
        binary_obs:dict = self.detector.detect_binary_states(self.env)
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
                sub_goal_reward += llm_reward_shaping_fn(numeric_obs_with_semantics, f"({effect.pddl_repr()})") * 1/num_effects
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
    

    
        
class Learner:
    def __init__(self, env:MujocoEnv, domain:str, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict):
        self.config = config
        self.domain = domain
        # from stable_baselines3 import the learning algorithm
        self.RL_algorithm = importlib.import_module(f"stable_baselines3.{self.config['learning']['algorithm'].lower()}").__dict__[self.config['learning']['algorithm']]
        self.executed_operators = executed_operators
        self.grounded_operator = grounded_operator_to_learn
        self.unwrapped_env = env
        self.llm_reward_candidates = self._load_llm_reward_fn_candidates()
        np.random.seed(self.config['learning']['model']['seed'])
    
    
    def _load_llm_reward_fn_candidates(self) -> List[Callable]:
        """Load the LLM generated reward shaping function candidates for the grounded operator

        Returns:
            List[Callable]: a list of reward shaping functions
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        reward_fn_candidates = []
        for i in range(self.config['learning']['reward_shaping_fn']['num_candidates']):
            try:
                llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.config['planning']['domain']}.{op_name}_{i}")
                llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
                reward_fn_candidates.append(llm_reward_shaping_func)
            except:
                func = self.prompt_llm_for_reward_shaping_fn_candidate()
                # save the reward shaping function candidates to a file
                # save the output python function to a file in the reward_functions directory
                # create the directory if it does not exist
                if not os.path.exists(f"learning{os.sep}reward_functions{os.sep}{self.domain}"):
                    os.makedirs(f"learning{os.sep}reward_functions{os.sep}{self.domain}")
                # create a file with the operator's name and save the function in it
                with open(f"learning{os.sep}reward_functions{os.sep}{self.domain}{os.sep}{op_name}_{i}.py", 'w') as f:
                    f.write(func)
                
        return reward_fn_candidates
    
    def learn_operator(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the grounded operator

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        save_path = f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}seed_{self.config['learning']['model']['seed']}"
        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        model_data = None
        for effect in self.grounded_operator.effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects:
                continue
            model_data = self.learn_subgoal(effect, save_path, prev_subgoal_model_data=model_data)
        # load the model to re-save it in the model save_path
        model_path, _, _ = model_data
        model = self.RL_algorithm.load(path=model_path)
        model.save(path=f"{save_path}{os.sep}final_model")
        # create an Executor_RL object associated with the newly learned policy.
        executor = execution.executor.Executor_RL(
            operator_name=op_name, alg=self.config['learning']['algorithm'],
            policy=model_path, 
        )
        # Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
        with open(f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}seed_{self.config['learning']['model']['seed']}{os.sep}executor.pkl", 'wb') as f:
            dill.dump(executor, f)
        return executor

    def learn_subgoal(self, subgoal:fs.SingleEffect, save_path:str, prev_subgoal_model_data:Union[str, os.PathLike]=None) -> Tuple[str, Monitor, Any]:
        """Train an RL agent to learn a subgoal/effect of the operator.
        Args:
            subgoal (fs.SingleEffect): the subgoal to learn
            save_path (str): the path to save the model
            prev_subgoal_model_path (str|os.PathLike): the path to the previous subgoal model
        Returns:
            Tuple[str, Monitor, CustomEvalCallback]: the path to the model, the environment, and the evaluation callback
        """
        subgoal_name:str = subgoal.pddl_repr().replace(' ', '_')
        subgoal_save_path = f"{save_path}{os.sep}{subgoal_name}"
        if prev_subgoal_model_data is not None:
            prev_model_path, prev_env, prev_eval_callback = prev_subgoal_model_data
        active_model_data = []

        for i, reward_fn in enumerate(self.llm_reward_candidates):
            if prev_subgoal_model_data is not None:
                reward_fn_save_path = f"{prev_model_path}{i}"
            else:
                reward_fn_save_path = f"{subgoal_save_path}{os.sep}reward_fn_{i}"

            # initialize the model environment
            env:Monitor = self._wrap_env(self.unwrapped_env, subgoal=subgoal, save_path=reward_fn_save_path)  

            # initialize the evaluation environment
            eval_env:Monitor = self._wrap_env(deepcopy_env(self.unwrapped_env, self.config['eval_simulation']), subgoal=subgoal, save_path=f"{reward_fn_save_path}_eval", record_rollouts=False)

            if prev_subgoal_model_data is not None:
                # make a copy of the model
                env.env.subgoal_reward_shaping_fn_mapping = copy.deepcopy(prev_env.env.subgoal_reward_shaping_fn_mapping)
                model = self.RL_algorithm.load(path=prev_model_path, env=env)
                eval_env.env.subgoal_reward_shaping_fn_mapping = copy.deepcopy(prev_env.env.subgoal_reward_shaping_fn_mapping)
                eval_callback = CustomEvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=f"{reward_fn_save_path}{os.sep}best_model",
                    log_path=f"{reward_fn_save_path}{os.sep}eval_logs",
                    **self.config['learning']['eval']
                )
            else:
                model_kwargs = dict(self.config['learning']['model'])  # copy so we can modify
                noise_type = model_kwargs.pop("action_noise", None)
                noise_kwargs = model_kwargs.pop("action_noise_kwargs", None)
                
                action_noise = None
                if noise_type == "OrnsteinUhlenbeckActionNoise" and noise_kwargs is not None:
                    n_actions = env.action_space.shape[-1]
                    action_noise = OrnsteinUhlenbeckActionNoise(
                        mean=np.full(n_actions, noise_kwargs["mean"]),
                        sigma=noise_kwargs["sigma"] * np.ones(n_actions),
                        theta=noise_kwargs["theta"],
                        dt=noise_kwargs["dt"],
                )
                if action_noise == None:
                    model = self.RL_algorithm(
                        "MlpPolicy",    
                        env = env,
                        tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                        **model_kwargs
                    )
                else:
                    model = self.RL_algorithm(
                        "MlpPolicy",    
                        env = env,
                        tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                        action_noise=action_noise,
                        **model_kwargs
                    )
                eval_callback = CustomEvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=f"{reward_fn_save_path}{os.sep}best_model",
                    log_path=f"{reward_fn_save_path}{os.sep}eval_logs",
                    **self.config['learning']['eval']
                )
                # model = self.RL_algorithm(
                #     "MlpPolicy",    
                #     env = env,
                #     tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                #     **self.config['learning']['model']
                # )
                # eval_callback = CustomEvalCallback(
                #     eval_env=eval_env,
                #     best_model_save_path=f"{reward_fn_save_path}{os.sep}best_model",
                #     log_path=f"{reward_fn_save_path}{os.sep}eval_logs",
                #     **self.config['learning']['eval']
                # )
            model_save_path = f"{reward_fn_save_path}{os.sep}model"
            model.save(
                path = model_save_path
            )

            # set the reward shaping function for the subgoal
            env.env.set_subgoal_reward_shaping_fn(subgoal, reward_fn)
            eval_env.env.set_subgoal_reward_shaping_fn(subgoal, reward_fn)
            
            active_model_data.append((model_save_path, env, eval_callback))
        
        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        if duplicate_grasp_effects:
            num_subgoals = len(self.grounded_operator.effects) - 1
        else:
            num_subgoals = len(self.grounded_operator.effects)
        subgoal_total_timesteps = self.config['learning']['learn_operator']['total_timesteps'] // num_subgoals
        subgoal_timesteps_so_far = 0
        while subgoal_timesteps_so_far < subgoal_total_timesteps: # train each reward function candidate until the total timesteps are reached
            for active_model_path, env, eval_callback in active_model_data:
                model = self.RL_algorithm.load(path=active_model_path, env=env)
                model.learn(
                    total_timesteps=self.config['learning']['learn_subgoal']['timesteps_per_iter'],
                    callback=eval_callback
                )
                model.save(
                    path = f"{reward_fn_save_path}{os.sep}model"
                )
            subgoal_timesteps_so_far += self.config['learning']['learn_subgoal']['timesteps_per_iter']
            # terminate the worst performing model
            subgoal_success_rates = [eval_callback.get_recent_subgoal_success_rate() for _, _, eval_callback in active_model_data]
            worst_performing_model_idx = np.argmin(subgoal_success_rates)
            best_performing_model_idx = np.argmax(subgoal_success_rates)
            if best_performing_model_idx != worst_performing_model_idx and subgoal_success_rates[best_performing_model_idx] > 0.5: # start dropping the worst performing model if at least one model has a success rate of over 50%
                active_model_data.pop(worst_performing_model_idx)
        # return the best performing model
        return active_model_data[np.argmax(subgoal_success_rates)]



    def prompt_llm_for_reward_shaping_fn_candidate(self) -> str:
        """Prompt the LLM to generate reward shaping functions candidates for the grounded operator
        Returns
            str: the reward shaping function candidate
        """
        grounded_op = self._grounded_operator_repr()
        dummy_detector = load_detector(self.config, self.unwrapped_env)
        observation_with_semantics = dummy_detector.get_obs()
        # keep only the keys that include the parameters of the grounded operator
        op_name, grounded_params = extract_name_params_from_grounded(self.grounded_operator.ident())
        observation_with_semantics = {k:v for k,v in observation_with_semantics.items() if any(param in k for param in grounded_params)}

        prompt = reward_shaping_prompt.format(grounded_operator=grounded_op, observation_with_semantics=observation_with_semantics)
        
        out = chat_completion(prompt)
        #parse the output to get the reward shaping function
        fn_start = out.find('# llm generated reward shaping function')
        fn_end = out.find('```', fn_start)
        fn = out[fn_start:fn_end]
        return fn
    
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

    def _wrap_env(self, env:MujocoEnv, subgoal:fs.SingleEffect, save_path:str, record_rollouts=False) -> gym.Wrapper:
        """Wrap the environment in multiple wrappers.

        Args:
            env (gym environment): the environment to wrap
            subgoal (fs.SingleEffect): the subgoal to learn
            save_path (str): the path to save the monitor logs
            record_rollouts (bool): whether to record rollouts

        Returns:
            gym.Wrapper: the wrapped environment
        """
        env = GymWrapper(env)
        env = OperatorWrapper(env, self.grounded_operator, self.executed_operators, self.config, curr_subgoal=subgoal, record_rollouts=record_rollouts)
        subgoals = []
        for eff in self.grounded_operator.effects:
            if eff.pddl_repr() == 'not (free gripper1)' and self.check_duplicate_grasp_effects():
                continue
            else:
                subgoals.append(f'{eff.pddl_repr()}_subgoal')
        env = Monitor(
            env=env, 
            filename=f"{save_path}{os.sep}monitor_logs",
            allow_early_resets=True,
            info_keywords=('subgoal_success', 'goal_success', 'ep_cumu_r_shaping', 'ep_cumu_col_penalty', 'ep_cumu_collisions') + tuple(subgoals)
        )
        return env
    
    def _grounded_operator_repr(self) -> str:
        """Return a string representation of the grounded operator

        Returns:
            str: the string representation of the grounded operator
        """
        effects:list = [eff.pddl_repr() for eff in self.grounded_operator.effects]
        if self.check_duplicate_grasp_effects():
            effects.remove('not (free gripper1)')
        effects_str:str = ' '.join(f'({eff})' for eff in effects)
        return f"{self.grounded_operator.name}\nprecondition: {self.grounded_operator.precondition.pddl_repr()}\neffects: and {effects_str}"
    
    def _load_llm_subgoal_reward_shaping_fn(self, i) -> Callable:
        """Load the ith LLM generated subgoal reward shaping function for the grounded operator

        Returns:
            Callable: the sub-goal reward shaping function. Creates the function if it does not exist
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        # if the file exists, import the function and return it. Otherwise, prompt the LLM to write the function
        try:
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.config['planning']['domain']}.{op_name}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
        except:
            self.prompt_llm_for_reward_shaping_fn_candidates()
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.config['planning']['domain']}.{op_name}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
        return llm_reward_shaping_func



class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, render_mode='human', verbose=1):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.eval_env.render_mode = render_mode
        self._subgoal_successes_buffer: List[bool] = [] # stores the per episode subgoal successes
        self.evaluations_subgoal_successes: List[List[bool]] = [] # stores the subgoal successes for each round of evaluation i.e a few episodes per evaluation
        self._ep_r_shaping_buffer: List[float] = [] # stores the per episode reward shaping values
        self.evaluations_r_shaping: List[List[float]] = [] # stores the reward shaping values for each round of evaluation i.e a few episodes per evaluation
        self._ep_col_penalty_buffer: List[float] = [] # stores the per episode collision penalties
        self.evaluations_col_penalties: List[List[float]] = [] # stores the collision penalties for each round of evaluation i.e a few episodes per evaluation
        self._ep_num_collisions_buffer: List[int] = [] # stores the per episode number of collisions
        self.evaluations_num_collisions: List[List[int]] = [] # stores the number of collisions for each round of evaluation i.e a few episodes per evaluation

    def get_recent_subgoal_success_rate(self):
        """Return the recent subgoal success rate
        """
        if len(self.evaluations_subgoal_successes) == 0:
            return 0
        return np.mean(self.evaluations_subgoal_successes[-1])
        
    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset buffers
            self._is_success_buffer = []
            self._subgoal_successes_buffer = []
            self._ep_r_shaping_buffer = []
            self._ep_col_penalty_buffer = []
            self._ep_num_collisions_buffer = []

            episode_rewards, episode_lengths = evaluate_policy( # this evaluates the model for a few policies
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_subgoal_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_r_shaping.append(self._ep_r_shaping_buffer)
                self.evaluations_col_penalties.append(self._ep_col_penalty_buffer)
                self.evaluations_num_collisions.append(self._ep_num_collisions_buffer)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)
                if len(self._subgoal_successes_buffer) > 0:
                    self.evaluations_subgoal_successes.append(self._subgoal_successes_buffer)
                    kwargs['subgoal_successes'] = self.evaluations_subgoal_successes
                # save the per episode reward shaping values
                if len(self._ep_r_shaping_buffer) > 0:
                    self.evaluations_r_shaping.append(self._ep_r_shaping_buffer)
                    kwargs['r_shaping'] = self.evaluations_r_shaping
                # save the per episode collision penalties
                if len(self._ep_col_penalty_buffer) > 0:
                    self.evaluations_col_penalties.append(self._ep_col_penalty_buffer)
                    kwargs['col_penalties'] = self.evaluations_col_penalties
                # save the per episode number of collisions
                if len(self._ep_num_collisions_buffer) > 0:
                    self.evaluations_num_collisions.append(self._ep_num_collisions_buffer)
                    kwargs['num_collisions'] = self.evaluations_num_collisions
 

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            # Log the evaluation values
            success_rate = 0
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Mean Success rate per episode: {100 * success_rate:.2f}%")
            self.logger.record("eval/mean_goal_success_rate_per_ep", success_rate)
            
            subgoal_success_rate = 0
            if len(self._subgoal_successes_buffer) > 0:
                subgoal_success_rate = np.mean(self._subgoal_successes_buffer)
                if self.verbose > 0:
                    print(f"Mean subgoals success rate per episode: {100 * subgoal_success_rate:.2f}%")
            self.logger.record("eval/mean_ep_subgoal_success_rate", subgoal_success_rate)
            
            mean_ep_r_shaping = None
            if len(self._ep_r_shaping_buffer) > 0:
                mean_ep_r_shaping = np.mean(self._ep_r_shaping_buffer)
                if self.verbose > 0:
                    print(f"Mean episode reward shaping per episode: {mean_ep_r_shaping:.2f}")
            self.logger.record("eval/mean_ep_r_shaping", mean_ep_r_shaping)

            mean_col_penalty = None
            if len(self._ep_col_penalty_buffer) > 0:
                mean_col_penalty = np.mean(self._ep_col_penalty_buffer)
                if self.verbose > 0:
                    print(f"Mean episode collision penalty per episode: {mean_col_penalty:.2f}")
            self.logger.record("eval/mean_ep_col_penalty", mean_col_penalty)

            mean_num_collisions = None
            if len(self._ep_num_collisions_buffer) > 0:
                mean_num_collisions = np.mean(self._ep_num_collisions_buffer)
                if self.verbose > 0:
                    print(f"Mean episode number of collisions per episode: {mean_num_collisions:.2f}")
            self.logger.record("eval/mean_ep_num_collisions", mean_num_collisions)

            # Dump log so the evaluation results are printed with the correct timestep
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"mean episode reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # get the min and max episode reward too
                print(f"Min episode reward: {np.min(episode_rewards)}, " f"Max episode reward: {np.max(episode_rewards)}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(f"Min episode length: {np.min(episode_lengths)}, " f"Max episode length: {np.max(episode_lengths)}")
            # Add to current Logger
            self.logger.record("eval/mean_ep_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)           
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            
            # Save the results in a csv file located in the second to last directory of log_path
            # Split the log_path to get the second to last directory
            csv_path = os.path.split(self.log_path)[0]
            with open(os.path.join(csv_path, 'results_eval.csv'), 'a') as f:
                f.write("{},{},{},{},{},{},{},{}\n".format(
                    self.num_timesteps,  
                    subgoal_success_rate,
                    success_rate, 
                    mean_ep_length,
                    mean_reward,  
                    mean_ep_r_shaping, 
                    mean_num_collisions,
                    mean_col_penalty,  
                ))
                f.close()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
    
    def _log_subgoal_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the subgoal success rate during evaluation.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if locals_["done"]:
            # store the episode reward shaping values and collision penalties
            self._ep_r_shaping_buffer.append(info.get('ep_cumu_r_shaping'))
            self._ep_col_penalty_buffer.append(info.get('ep_cumu_col_penalty'))
            self._ep_num_collisions_buffer.append(info.get('ep_cumu_collisions'))
            maybe_is_success = info.get("goal_success")
            subgoal_success = info.get("subgoal_success")
            #subgoals = [key for key in info.keys() if '_subgoal' in key]
            self._subgoal_successes_buffer.append(subgoal_success)
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            
