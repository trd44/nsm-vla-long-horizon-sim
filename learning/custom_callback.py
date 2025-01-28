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
import stable_baselines3
import logging
from tarski import fstrips as fs
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from typing import *
from utils import *

class RenderCallback(EventCallback):
    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
    
    def _on_step(self):
        self.training_env.render_mode = self.render_mode
        self.training_env.render()
        return True

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, logger, best_model_save_path, log_path, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, render_mode='human', verbose=1):
        super().__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.eval_env.render_mode = render_mode
        self.custom_logger = logger
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
                    self.custom_logger.info(f"Mean Success rate per episode: {100 * success_rate:.2f}%")
            self.logger.record("eval/mean_goal_success_rate_per_ep", success_rate)
            
            subgoal_success_rate = 0
            if len(self._subgoal_successes_buffer) > 0:
                subgoal_success_rate = np.mean(self._subgoal_successes_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean subgoals success rate per episode: {100 * subgoal_success_rate:.2f}%")
            self.logger.record("eval/mean_ep_subgoal_success_rate", subgoal_success_rate)
            
            mean_ep_r_shaping = None
            if len(self._ep_r_shaping_buffer) > 0:
                mean_ep_r_shaping = np.mean(self._ep_r_shaping_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode reward shaping per episode: {mean_ep_r_shaping:.2f}")
            self.logger.record("eval/mean_ep_r_shaping", mean_ep_r_shaping)

            mean_col_penalty = None
            if len(self._ep_col_penalty_buffer) > 0:
                mean_col_penalty = np.mean(self._ep_col_penalty_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode collision penalty per episode: {mean_col_penalty:.2f}")
            self.logger.record("eval/mean_ep_col_penalty", mean_col_penalty)

            mean_num_collisions = None
            if len(self._ep_num_collisions_buffer) > 0:
                mean_num_collisions = np.mean(self._ep_num_collisions_buffer)
                if self.verbose > 0:
                    self.custom_logger.info(f"Mean episode number of collisions per episode: {mean_num_collisions:.2f}")
            self.logger.record("eval/mean_ep_num_collisions", mean_num_collisions)

            # Dump log so the evaluation results are printed with the correct timestep
            if self.verbose > 0:
                self.custom_logger.info(f"Eval num_timesteps={self.num_timesteps}, " f"mean episode reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # get the min and max episode reward too
                self.custom_logger.info(f"Min episode reward: {np.min(episode_rewards)}, " f"Max episode reward: {np.max(episode_rewards)}")
                self.custom_logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                self.custom_logger.info(f"Min episode length: {np.min(episode_lengths)}, " f"Max episode length: {np.max(episode_lengths)}")
            # Add to current Logger
            self.logger.record("eval/mean_ep_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)           
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    self.custom_logger.info("New best mean reward!")
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