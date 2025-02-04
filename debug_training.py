import os
import gymnasium as gym
import importlib
import numpy as np
import logging
import argparse
import json
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import SAC, PPO, DDPG
from typing import *
from learning.reward_functions.rewardFunctionPrompts import *
from learning.learning_utils import *
from learning.custom_gym_wrapper import *
from learning.custom_callback import *
from utils import *

# set up a logger here to log the terminal printouts for the training of each subgoal
logger = logging.getLogger('learning')
logger.setLevel(logging.INFO)

class MinimalWrapper(gym.Wrapper):
    def __init__(self, env, config, domain):
        super().__init__(env)
        self.env = env
        self.config = config
        self.domain = domain
        self.detector = load_detector(config=config, domain=domain, env=env)
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = self.action(action)
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        
        obs_from_detector = self.detector.get_obs()
        binary_obs = self.detector.detect_binary_states(self.env)
        dist = np.linalg.norm(obs_from_detector['gripper1_pos'] - obs_from_detector['mug1_pos'])
        normalized_progress = 1 - np.clip(dist / 1.0, 0, 1) # assume a normalization factor of 1.0
        grasp = binary_obs['exclusively-occupying-gripper mug1 gripper1']
        if grasp:
            reward = 1
        else:
            reward = normalized_progress * 0.99 # without fully grasping the mug, the reward is capped at 0.99
        self.episode_r_shaping += reward
        done = done or grasp
        info['ep_cumu_r_shaping'] = self.episode_r_shaping
        info['ep_cumu_col_penalty'] = self.episode_collision_penalty
        info['ep_cumu_collisions'] = self.episode_num_collisions
        
        # save subgoal successes
        info['subgoal_success'] = grasp
        
        # overall goal success is if all subgoals are achieved
        info['goal_success'] = grasp
        step_cost = - 4
        return obs, reward + step_cost, done, truncated, info
    
    
    def action(self, action):
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
    
    def reset(self, **kwargs):
        self.episode_r_shaping = 0
        self.episode_collision_penalty = 0
        self.episode_num_collisions = 0
        return self.env.reset(**kwargs)

if __name__ == '__main__':
    config:dict = load_config("config.yaml")
    algo = 'SAC'

    # parse model commandline args
    parser = argparse.ArgumentParser()
    # add default args according to config
    for arg, val in config['learning'][algo].items():
        parser.add_argument(f'--{arg}', type=type(val), default=val)
        if type(val) == list:
            parser.add_argument(f'--{arg}', type=json.loads, default=val)
    for arg, val in config['learning']['eval'].items():
        if arg in config['learning'][algo]:
            continue
        parser.add_argument(f'--{arg}', type=type(val), default=val)
        if type(val) == list:
            parser.add_argument(f'--{arg}', type=json.loads, default=val)
    parser.add_argument('--total_timesteps', type=int, default=config['learning']['learn_subgoal']['total_timesteps'])
    parser.add_argument('--ep_len', type=int, default=100)
    parser.add_argument('--render_training', type=bool, default=False)
    parser.add_argument('--domain', type=str, default='cleanup')
    parser.add_argument('--net_arch', type=json.loads, default=config['learning'][algo]['policy_kwargs']['net_arch'])
    parser.add_argument('--lr_schedule', action='store_true')
    parser.set_defaults(lr_schedule=False)  # Default value
    parser.add_argument('--op_wrap', action='store_true', help='Enable operator wrapper')
    parser.set_defaults(op_wrap=False)  # Default value
    parser.add_argument('--rw_shaping', type=int, default=0, help='the reward shaping function candidate to use. Usually from 0 - 2.')
    args = parser.parse_args()

    domain = args.domain
    # print the non-default commandline args and construct kwargs for model and eval
    model_kwargs = {}
    eval_kwargs = {}
    save_path = f"{algo}_approach_{domain}"
    # print the algorithm name and save path
    print(f"#######Training Algorithm: {algo} at {save_path}#######")
    for arg, val in vars(args).items():
        if val != parser.get_default(arg):
            print(f"{arg}: {val}")
            save_path += f"_{arg}_{val}"
        if arg == 'net_arch': # special case for net_arch
            model_kwargs['policy_kwargs'] = {'net_arch': val}
        if arg in config['learning'][algo]:
            model_kwargs[arg] = val
        elif arg in config['learning']['eval']:
            eval_kwargs[arg] = val
    if args.lr_schedule:
        model_kwargs['learning_rate'] = linear_schedule(args.learning_rate)
    
    plan = load_plan(config['planning'][domain]) # load the plan in case the operator wrapper is used
    grounded_op = plan[0] # assuming the first operator is the one we need to learn. It's not necessarily the case
    curr_subgoal = grounded_op.effects[0] if 'not (free gripper1)' not in grounded_op.effects[0].pddl_repr() else grounded_op.effects[1] # assuming the first effect is the subgoal. It's not necessarily the case
    # load LLM generated reward function in case the operator wrapper is used
    op_name, _ = extract_name_params_from_grounded(grounded_op.ident())
    reward_fn_candidates = []
    for i in range(config['learning']['reward_shaping_fn']['num_candidates']):
        try:
            llm_reward_func_module = importlib.import_module(f"learning.reward_functions.{domain}.{op_name}_{i}")
            llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping_fn')
            reward_fn_candidates.append(llm_reward_shaping_func)
        except:
            raise Exception(f"Reward function {op_name}_{i} not found")
    
    # set seed for env too
    np.random.seed(model_kwargs['seed'])
    robosuite_env = load_env(domain, config['simulation'])
    visual_env = VisualizationWrapper(robosuite_env, indicator_configs=None)
    gym_env = GymWrapper(robosuite_env, keys=['gripper1_pos', 'mug1_pos'])
    time_limit_env = TimeLimit(gym_env, max_episode_steps=args.ep_len)

    if args.op_wrap:
        wrapped_env = CollisionAblatedOperatorWrapper(time_limit_env, grounded_operator=grounded_op, executed_operators={}, config=config, domain=domain, rl_algo=algo, curr_subgoal=curr_subgoal)
        wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
    else:
        wrapped_env = MinimalWrapper(time_limit_env, config, domain)
    env = Monitor(wrapped_env, filename=f'{save_path}{os.sep}approach_monitor', allow_early_resets=True)

    eval_robosuite_env = load_env(domain, config['simulation'])
    eval_gym_env = GymWrapper(eval_robosuite_env, keys=['gripper1_pos', 'mug1_pos'])
    eval_time_limit_env = TimeLimit(eval_gym_env, max_episode_steps=args.ep_len)
    if args.op_wrap:
        eval_wrapped_env = CollisionAblatedOperatorWrapper(eval_time_limit_env, grounded_operator=grounded_op, executed_operators={}, config=config, domain=domain, rl_algo=algo, curr_subgoal=curr_subgoal)
        eval_wrapped_env.set_subgoal_reward_shaping_fn(curr_subgoal, reward_fn_candidates[args.rw_shaping])
    else:
        eval_wrapped_env = MinimalWrapper(eval_time_limit_env, config, domain)
    eval_env = Monitor(eval_wrapped_env, filename=f'{save_path}{os.sep}approach_eval_monitor', allow_early_resets=True)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    # load the algo from stable_baselines3
    rl_algo = importlib.import_module(f"stable_baselines3.{algo.lower()}").__dict__[algo.upper()]
    # load model if it exists
    
    if os.path.exists(f"{save_path}{os.sep}best_model{os.sep}best_model.zip"):
        model = rl_algo.load(f"{save_path}{os.sep}best_model{os.sep}best_model.zip", env)
    else:
        model = rl_algo("MlpPolicy", env, **model_kwargs, tensorboard_log=f"{save_path}{os.sep}tensorboard/")

    # create the logger
    # set up a logger here to log the terminal printouts for the training of each subgoal
    logger = logging.getLogger('learning')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): # Remove duplicate handlers
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{save_path}{os.sep}learner_train_logs.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

   
    eval_callback = CustomEvalCallback(eval_env=eval_env, logger=logger, best_model_save_path=save_path, log_path=f"{save_path}{os.sep}logs/", recent_model_save_path=save_path, **eval_kwargs)
    callbacks = CallbackList([eval_callback])

    if args.render_training:
        train_callback = RenderCallback()
        callbacks = CallbackList([train_callback, eval_callback])

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=False)
    model.save(f"{save_path}{os.sep}model")
