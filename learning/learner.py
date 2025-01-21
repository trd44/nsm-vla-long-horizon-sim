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
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from learning.custom_eval_callback import CustomEvalCallback
from learning.custom_gym_wrapper import *
from utils import *
from VLM.LlmApi import chat_completion


class BaseLearner:
    """Learner class that learns the grounded operator using an RL algorithm without the LLM. One learner is associated with one grounded operator.
    """
    def __init__(self, env:MujocoEnv, domain:str, rl_algo:str, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict):
        self.config = config
        self.domain = domain
        self.rl_algo_name = rl_algo
        # from stable_baselines3 import the learning algorithm
        self.rl_algo = importlib.import_module(f"stable_baselines3.{self.rl_algo_name.lower()}").__dict__[self.rl_algo_name]
        self.executed_operators = executed_operators
        self.grounded_operator = grounded_operator_to_learn
        self.unwrapped_env = env
        np.random.seed(self.config['learning'][self.rl_algo_name]['seed'])

        # create the logger
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        save_path = f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}{self.rl_algo_name}{os.sep}seed_{self.config['learning'][self.rl_algo_name]['seed']}"
        
        # create the logger
        # set up a logger here to log the terminal printouts for the training of each subgoal
        self.logger = logging.getLogger('learning')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): # Remove duplicate handlers
            self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(f"{save_path}{os.sep}learner_train_logs.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def learn_operator(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the grounded operator

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        save_path = f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}{self.rl_algo_name}{os.sep}seed_{self.config['learning'][self.rl_algo_name]['seed']}"
    

        # configure the model
        model_kwargs = dict(self.config['learning'][self.rl_algo_name])  # copy so we can modify
        noise_type = model_kwargs.pop("action_noise", None)
        noise_kwargs = model_kwargs.pop("action_noise_kwargs", None)

        # create the eval env
        last_subgoal = self.grounded_operator.effects[-1]
        env = self._wrap_env(deepcopy_env(self.unwrapped_env, self.config['learn_simulation']), subgoal=last_subgoal, save_path=save_path, record_rollouts=False)
        eval_env:Monitor = self._wrap_env(deepcopy_env(self.unwrapped_env, self.config['eval_simulation']), subgoal=last_subgoal, save_path=f"{save_path}_eval", record_rollouts=False)
        eval_callback = CustomEvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"{save_path}{os.sep}best_model",
            log_path=f"{save_path}_eval{os.sep}eval_logs",
            **self.config['learning']['eval'],
            logger=self.logger
            )
        
        action_noise = None
        if noise_type == "OrnsteinUhlenbeckActionNoise" and noise_kwargs is not None:
            n_actions = self.unwrapped_env.action_space.shape[-1]
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.full(n_actions, noise_kwargs["mean"]),
                sigma=noise_kwargs["sigma"] * np.ones(n_actions),
                theta=noise_kwargs["theta"],
                dt=noise_kwargs["dt"],
        )
        if action_noise == None:
            model = self.rl_algo(
                "MlpPolicy",    
                env = env,
                tensorboard_log=f"{save_path}{os.sep}tensorboard_logs",
                **model_kwargs
            )
        else:
            model = self.rl_algo(
                "MlpPolicy",    
                env = env,
                tensorboard_log=f"{save_path}{os.sep}tensorboard_logs",
                action_noise=action_noise,
                **model_kwargs
            )
        
        # train the model
        model.learn(
            total_timesteps=self.config['learning']['learn_operator']['total_timesteps'],
            callback=eval_callback
        )
        # save the model
        model_path = f"{save_path}{os.sep}final_model"
        model.save(path=model_path)
        # create an Executor_RL object associated with the newly learned policy.
        executor = execution.executor.Executor_RL(
            operator_name=op_name,
            alg=self.rl_algo_name,
            policy=model_path
        )
        # Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
        with open(f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}seed_{self.config['learning'][self.rl_algo_name]['seed']}{os.sep}executor.pkl", 'wb') as f:
            dill.dump(executor, f)
        return executor

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
        env = LLMAblatedOperatorWrapper(
            env=env,
            rl_algo=self.rl_algo_name,
            domain=self.domain,
            grounded_operator=self.grounded_operator, 
            executed_operators=self.executed_operators, 
            config=self.config, 
            curr_subgoal=subgoal, 
            record_rollouts=record_rollouts,
        )
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

class LLMLearner(BaseLearner):
    """prompts the LLM for reward shaping function candidates for the grounded operator and learns the grounded operator using the reward shaping function candidates
    """
    def __init__(self, env:MujocoEnv, domain:str, rl_algo:str, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action, execution.executor.Executor], config:dict):
        super().__init__(env, domain, rl_algo, grounded_operator_to_learn, executed_operators, config)
        self.llm_reward_candidates = self._load_llm_reward_fn_candidates()
    
    
    def _load_llm_reward_fn_candidates(self) -> List[Callable]:
        """Load the LLM generated reward shaping function candidates for the grounded operator

        Returns:
            List[Callable]: a list of reward shaping functions
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        reward_fn_candidates = []
        for i in range(self.config['learning']['reward_shaping_fn']['num_candidates']):
            try:
                llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.domain}.{op_name}_{i}")
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
                # wait 5 seconds for the file to be saved
                time.sleep(5)
                llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{self.domain}.{op_name}_{i}")
                llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
                reward_fn_candidates.append(llm_reward_shaping_func)
        return reward_fn_candidates
    
    def learn_operator(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the grounded operator

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator
        """
        op_name, _ = extract_name_params_from_grounded(self.grounded_operator.ident())
        save_path = f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}{self.rl_algo_name}{os.sep}seed_{self.config['learning'][self.rl_algo_name]['seed']}"
        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        model_data = None
        for effect in self.grounded_operator.effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects:
                continue
            model_data = self.learn_subgoal(effect, save_path, prev_subgoal_model_data=model_data)
        # load the model to re-save it in the model save_path
        model_path, _, _ = model_data
        model = self.rl_algo.load(path=model_path)
        model.save(path=f"{save_path}{os.sep}final_model")
        # create an Executor_RL object associated with the newly learned policy.
        executor = execution.executor.Executor_RL(
            operator_name=op_name,
            alg=self.rl_algo_name,
            policy=model_path
        )
        # Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
        with open(f"learning{os.sep}policies{os.sep}{self.domain}{os.sep}{op_name}{os.sep}seed_{self.config['learning'][self.rl_algo_name]['seed']}{os.sep}executor.pkl", 'wb') as f:
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
        if not os.path.exists(subgoal_save_path):
            os.makedirs(subgoal_save_path)
        # Remove duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(f"{subgoal_save_path}{os.sep}rw_fn_candidates_train_logs.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if prev_subgoal_model_data is not None:
            prev_model_path, prev_env, prev_eval_callback = prev_subgoal_model_data
        model_data = []

        for i, reward_fn in enumerate(self.llm_reward_candidates):
            if prev_subgoal_model_data is not None:
                # drop the '{os.sep}model' from the path and add '{i}' to the end
                reward_fn_save_path = prev_model_path[:prev_model_path.rfind(f'{os.sep}model')] + f"{os.sep}{i}"
            else:
                reward_fn_save_path = f"{subgoal_save_path}{os.sep}reward_fn_{i}"

            # initialize the model environment
            env:Monitor = self._wrap_env(self.unwrapped_env, subgoal=subgoal, save_path=reward_fn_save_path)  

            # initialize the evaluation environment
            eval_env:Monitor = self._wrap_env(deepcopy_env(self.unwrapped_env, self.config['eval_simulation']), subgoal=subgoal, save_path=f"{reward_fn_save_path}_eval", record_rollouts=False)

            if prev_subgoal_model_data is not None:
                # make a copy of the model
                env.env.subgoal_reward_shaping_fn_mapping = copy.deepcopy(prev_env.env.subgoal_reward_shaping_fn_mapping)
                model = self.rl_algo.load(path=prev_model_path, env=env)
                eval_env.env.subgoal_reward_shaping_fn_mapping = copy.deepcopy(prev_env.env.subgoal_reward_shaping_fn_mapping)
            else:
                model_kwargs = dict(self.config['learning'][self.rl_algo_name])  # copy so we can modify
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
                    model = self.rl_algo(
                        "MlpPolicy",    
                        env = env,
                        tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                        **model_kwargs
                    )
                else:
                    model = self.rl_algo(
                        "MlpPolicy",    
                        env = env,
                        tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                        action_noise=action_noise,
                        **model_kwargs
                    )
            eval_callback = CustomEvalCallback(
                eval_env=eval_env,
                best_model_save_path=f"{reward_fn_save_path}{os.sep}best_model",
                log_path=f"{reward_fn_save_path}_eval{os.sep}eval_logs",
                **self.config['learning']['eval'],
                logger=self.logger
                )
                # model = self.RL_algorithm(
                #     "MlpPolicy",    
                #     env = env,
                #     tensorboard_log=f"{reward_fn_save_path}{os.sep}tensorboard_logs",
                #     **self.config['learning'][self.rl_algo_name]['model']
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
            
            model_data.append((model_save_path, env, eval_callback))
        
        subgoal_total_timesteps = self.config['learning']['learn_subgoal']['total_timesteps']
        subgoal_timesteps_so_far = 0
        while subgoal_timesteps_so_far < subgoal_total_timesteps: # train each reward function candidate until the total timesteps are reached
            model_indices_to_train = list(range(len(model_data)))
            model_indices_to_exclude = []
            
            for i in model_indices_to_train:
                active_model_path, env, eval_callback = model_data[i]
                model = self.rl_algo.load(path=active_model_path, env=env)
                self.logger.info(f"\n{'='*40}\nTraining model {active_model_path} for {subgoal.pddl_repr()} for {self.config['learning']['learn_subgoal']['timesteps_per_iter']} timesteps, already trained for {subgoal_timesteps_so_far} timesteps\n{'='*40}\n")
                try: # try to train the model with llm reward shaping function
                    model.learn(
                        total_timesteps=self.config['learning']['learn_subgoal']['timesteps_per_iter'],
                        callback=eval_callback,
                        reset_num_timesteps=False
                    )
                except Exception as e: # if the llm reward shaping function is not error-free, catch the error, log it, and eliminate this model from the active models
                    self.logger.error(e)
                    model_indices_to_exclude.append(i)
                # save the model after however much training has been done
                model.save(path = active_model_path)
            subgoal_timesteps_so_far += self.config['learning']['learn_subgoal']['timesteps_per_iter']

            # find the worst and best performing model
            subgoal_success_rates = {}
            worst_performance = 1
            worst_performing_model_idx = None
            best_performance = 0
            best_performing_model_idx = None
            for i in model_indices_to_train:
                eval_callback = model_data[i][2]
                subgoal_success_rate = eval_callback.get_recent_subgoal_success_rate()
                if subgoal_success_rate < worst_performance:
                    worst_performance = subgoal_success_rate
                    worst_performing_model_idx = i
                if subgoal_success_rate > best_performance:
                    best_performance = subgoal_success_rate
                    best_performing_model_idx = i
                subgoal_success_rates[i] =  subgoal_success_rate

            if best_performing_model_idx != worst_performing_model_idx and subgoal_success_rates[best_performing_model_idx] > 0.5: # start dropping the worst performing model if at least one model has a success rate of over 50%
                self.logger.info(f"Terminating the worst performing model {model_data[worst_performing_model_idx][0]}")
                model_indices_to_exclude.append(worst_performing_model_idx)
            # remove the excluded models from the active models
            for i in model_indices_to_exclude:
                model_indices_to_train.remove(i)
        # return the best performing model
        return model_data[np.argmax(subgoal_success_rates)]


    def prompt_llm_for_reward_shaping_fn_candidate(self) -> str:
        """Prompt the LLM to generate reward shaping functions candidates for the grounded operator
        Returns
            str: the reward shaping function candidate
        """
        grounded_op = self._grounded_operator_repr()
        dummy_detector = load_detector(self.config, self.domain, self.unwrapped_env)
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
            env = OperatorWrapper(
                env=env,
                rl_algo=self.rl_algo_name,
                domain=self.domain,
                grounded_operator=self.grounded_operator, 
                executed_operators=self.executed_operators, 
                config=self.config, 
                curr_subgoal=subgoal, 
                record_rollouts=record_rollouts,
            )
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
