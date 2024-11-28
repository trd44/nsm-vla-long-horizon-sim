import copy
import dill
import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
import numpy as np
from tarski import fstrips as fs
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from utils import *
from VLM.LlmApi import chat_completion

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, render_mode='human', verbose=1):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.eval_env.render_mode = render_mode

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

                # Reset success rate buffer
                self._is_success_buffer = []

                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                    kwargs = {}
                    # Save success log if present
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes.append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes)

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

                if self.verbose > 0:
                    print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                # Add to current Logger
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                success_rate = 0
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose > 0:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
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
                    f.write("{},{},{},{}\n".format(self.num_timesteps, success_rate, mean_reward, mean_ep_length))
                    f.close()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()

            return continue_training

class OperatorWrapper(gym.Wrapper):
    def __init__(self, env:MujocoEnv, grounded_operator:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict):
        super().__init__(env)
        self.detector = load_detector(config=config, env=env)
        self.grounded_operator = grounded_operator
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = executed_operators
        self.config = config
        self.domain = self.config['planning']['domain']
        self.llm_reward_shaping_fn:Callable = self._load_llm_sub_goal_reward_shaping_fn()

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
        done = done or reward == 0
        
        return obs, reward, done, truncated, info

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

        self.detector.update_obs()
        #obs = self.detector.get_obs()
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
        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if self.check_effect_satisfied(effect, binary_obs):
                sub_goal_reward += 1/num_effects
            else:
                sub_goal_reward += self.llm_reward_shaping_fn(numeric_obs_with_semantics, effect.pddl_repr()) * 1/num_effects
                return step_cost + sub_goal_reward # return the reward as soon as one effect is not satisfied. Assume later effects are at 0% progress therefore would get a shaping reward of 0 anyway.
        
        return step_cost + sub_goal_reward
    
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
    
    def _grounded_operator_repr(self) -> str:
        """Return a string representation of the grounded operator

        Returns:
            str: the string representation of the grounded operator
        """
        effects:str = '\n'.join([eff.pddl_repr() for eff in self.grounded_operator.effects])
        return f"{self.grounded_operator.name}\nprecondition: {self.grounded_operator.precondition.pddl_repr()}\neffects:\n{effects}\n"

    
    def _load_llm_sub_goal_reward_shaping_fn(self) -> Callable:
        """Load the LLM generated sub-goal reward shaping function for the grounded operator

        Returns:
            Callable: the sub-goal reward shaping function. Creates the function if it does not exist
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        # if the file exists, import the function and return it. Otherwise, prompt the LLM to write the function
        try:
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.config['planning']['domain']}.{op_name}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
        except:
            self._llm_sub_goal_reward_shaping()
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.config['planning']['domain']}.{op_name}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
        return llm_reward_shaping_func
    
    def _llm_sub_goal_reward_shaping(self):
        """Prompt the LLM to write a sub-goal reward shaping function that takes in an effect (sub-goal) and the observation with semantics and returns a reward depending on the progress towards achieving the effect.
        """
        # dynamically fill in the prompt with operator specific information such as the operator's name and effects
        grounded_op = self._grounded_operator_repr()
        observation_with_semantics = self.detector.get_obs()
        # keep only the keys that include the parameters of the grounded operator
        op_name, grounded_params = extract_name_params_from_grounded(self.grounded_operator.ident())
        observation_with_semantics = {k:v for k,v in observation_with_semantics.items() if any(param in k for param in grounded_params)}

        prompt = reward_shaping_prompt.format(grounded_operator=grounded_op, observation_with_semantics=observation_with_semantics)
        out = chat_completion(prompt)
        #parse the output to get the reward shaping function
        fn_start = out.find('# llm generated reward shaping function')
        fn_end = out.find('```', fn_start)
        fn = out[fn_start:fn_end]
        # save the output python function to a file in the reward_functions directory
        # create the directory if it does not exist
        if not os.path.exists(f"learning{os.sep}reward_functions{os.sep}{self.domain}"):
            os.makedirs(f"learning{os.sep}reward_functions{os.sep}{self.domain}")
        # create a file with the operator's name and save the function in it
        with open(f"learning{os.sep}reward_functions{os.sep}{self.domain}{os.sep}{op_name}.py", 'w') as f:
            f.write(fn)

        
class Learner:
    def __init__(self, env:MujocoEnv, domain:str, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict):
        self.config = config
        self.domain = domain
        self.executed_operators = executed_operators
        self.grounded_operator = grounded_operator_to_learn
        self.env = self._wrap_env(env)
        self.eval_env = self._wrap_env(deepcopy_env(env, config['eval_simulation']))

    def learn(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the operator.

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator and executes the policy for the operator when called
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())

        eval_callback = CustomEvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/best_model",
            log_path=f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/eval_logs",
            **self.config['learning']['eval']
        )
        model = SAC(
            "MlpPolicy",
            env = self.env,
            tensorboard_log=f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/tensorboard_logs",
            **self.config['learning']['model']
        )
        model.learn(
            **self.config['learning']['learn'],
            callback=eval_callback
        )
        model.save(
            path = f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/model"
        )
        # create an Executor_RL object associated with the newly learned policy.
        executor = execution.executor.Executor_RL(
            operator_name=op_name, alg='SAC',
            policy=f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/model", 
        )
        # Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
        with open(f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/executor.pkl", 'wb') as f:
            dill.dump(executor, f)
        return executor
    

    def _wrap_env(self, env) -> gym.Wrapper:
        """Wrap the environment in multiple wrappers.

        Args:
            env (gym environment): the environment to wrap

        Returns:
            gym.Wrapper: the wrapped environment
        """
        op_name, grounded_params = extract_name_params_from_grounded(self.grounded_operator.ident())
        # obs_with_semantics = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations() # hacky way to get the observations with semantics
        # obs_keys_to_inlcude = [k for k in obs_with_semantics.keys() if any(param in k for param in grounded_params)] # find the keys that include the parameters of the grounded operator

        env = GymWrapper(env)
        env = OperatorWrapper(env, self.grounded_operator, self.executed_operators, self.config)
        env = Monitor(
            env=env, 
            filename=f"learning/policies/{self.domain}/{op_name}/seed_{self.config['learning']['model']['seed']}/monitor_logs",
            allow_early_resets=True
        )
        return env



