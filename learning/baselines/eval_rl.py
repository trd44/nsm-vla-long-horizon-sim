import argparse
import numpy as np
import os
import robosuite as suite
import time
import warnings
import torch
from datetime import datetime
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from custom_rl_callback import CustomEvalCallback
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

# Import environments
from robosuite.wrappers.nutassembly.assemble_pick import AssemblePickWrapper
from robosuite.wrappers.nutassembly.assemble_place import AssemblePlaceWrapper
from robosuite.wrappers.nutassembly.vision import AssembleVisionWrapper
from robosuite.wrappers.nutassembly.object_state import AssembleStateWrapper
from robosuite.wrappers.kitchen.kitchen_pick import KitchenPickWrapper
from robosuite.wrappers.kitchen.kitchen_place import KitchenPlaceWrapper
from robosuite.wrappers.kitchen.turn_on_stove import TurnOnStoveWrapper
from robosuite.wrappers.kitchen.turn_off_stove import TurnOffStoveWrapper
from robosuite.wrappers.kitchen.vision import KitchenVisionWrapper
from robosuite.wrappers.kitchen.object_state import KitchenStateWrapper
from robosuite.wrappers.hanoi.hanoi_pick import HanoiPickWrapper
from robosuite.wrappers.hanoi.hanoi_place import HanoiPlaceWrapper
from robosuite.wrappers.hanoi.vision import HanoiVisionWrapper
from robosuite.wrappers.hanoi.object_state import HanoiStateWrapper

warnings.filterwarnings("ignore")

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

op_to_wrapper = {"Hanoi":
	{   "pick": HanoiPickWrapper,
        "place": HanoiPlaceWrapper,
    },
	"KitchenEnv":
    {   "pick": KitchenPickWrapper,
        "place": KitchenPlaceWrapper,
        "turnon": TurnOnStoveWrapper,
        "turnoff": TurnOffStoveWrapper,
    },
    "NutAssembly":
    {   "pick": AssemblePickWrapper,
        "place": AssemblePlaceWrapper,
    }
    }

vision_wrapper = {"Hanoi": HanoiVisionWrapper, "KitchenEnv": KitchenVisionWrapper, "NutAssembly": AssembleVisionWrapper}
object_state_wrapper = {"Hanoi": HanoiStateWrapper, "KitchenEnv": KitchenStateWrapper, "NutAssembly": AssembleStateWrapper}

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def learn_policy(args, env, eval_env, name):
    # Add noise to SAC actions
    #action_dim = env.action_space.shape[0]
    #action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.2 * np.ones(action_dim))
    # Define the model
    policy_mode = 'MlpPolicy' if not args.vision else 'CnnPolicy'
    model = SAC(
        policy_mode,
        env,
        batch_size=512,
        #action_noise=action_noise,
        #learning_rate=args.lr,
        #buffer_size=int(1e6),
        #device=device,
        #learning_starts=100,#10000,
        #tau=0.005,
        #gamma=0.99,
        #policy_kwargs=dict(net_arch=[1024, 512, 256]),
        policy_kwargs=dict(net_arch=[256, 512, 128]),
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed
    )
    print("Saving the model in: {}, as best_model.zip and final model {}".format(args.modeldir, os.path.join(args.bufferdir, 'task' + '_sac')))
    # Define all callbacks
    #callbacks = []

    #callbacks.append(eval_callback)
    # Add a stop callback on success rate of 100%
    #stop_training_callback = EvalCallback(
    #callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=0.95, verbose=1))
    # Add a stop callback on no improvement
    # no_improvement_callback = EvalCallback(eval_env=eval_env, 
    #                                        eval_freq=args.eval_freq, 
    #                                        n_eval_episodes=args.n_eval_episodes, 
    #                                        deterministic=True, 
    #                                        render=False, 
    #                                        verbose=1, 
    #                                        callback_after_eval=StopTrainingOnNoModelImprovement(max_no_improvement_evals=1_000_000, verbose=1),
    #                                        log_path=args.logdir)
    #callbacks.append(StopTrainingOnNoModelImprovement(max_no_improvement_evals=1_000_000, verbose=1))
    #callbacks = CallbackList(callbacks)

    # Define the evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=args.modeldir,
        log_path=args.logdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        stop_training_threshold=0.95,
        #callback_after_eval=StopTrainingOnRewardThreshold(reward_threshold=0.95, verbose=1),
    )

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback
    )
    # Save the model
    model.save(os.path.join(args.modeldir, name + '_sac'))
    return model

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly'], help='Name of the environment to run the experiment in')
    parser.add_argument('--op', type=str, default='pick', choices=['pick', 'place', 'turnon', 'turnoff'], help='Name of the operator to train the policy for')
    parser.add_argument('--dir', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--timesteps', type=int, default=int(1e7), help='Number of timesteps to train for')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') # 0.00005 0.00001
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--vision', action='store_true', help='Use vision based learning')
    parser.add_argument('--logs', type=str, default=None)
    parser.add_argument('--size', type=int, default=128, help='size of the image observation (square)')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)

    # Define the evaluation frequency
    args.eval_freq = min(args.eval_freq, args.timesteps)

    # Define the directories
    data_folder = args.dir
    experiment_name = f"{args.env}_op_{args.op}_{args.seed}"
    experiment_id = f"{to_datestring(time.time())}"#self.hashid
    if args.name is not None:
        experiment_id = args.name
    args.env_dir = os.path.join(data_folder, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))
    print("Saving the experiment in: {}".format(args.env_dir))
    print("Arguments: {}".format(args))

    # Create the directories
    os.makedirs(args.env_dir, exist_ok=True)

    # Save args in a txt file
    with open(os.path.join(args.env_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
        f.close()
    args.logdir = os.path.join(args.env_dir, 'logs')
    args.modeldir = os.path.join(args.env_dir, 'models')
    args.bufferdir = os.path.join(args.env_dir, 'buffers')
    if args.logs == None:
        args.logs = args.logdir
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)
    os.makedirs(args.bufferdir, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    # Save PID to a file
    with open(args.logs+"/pid.txt", "w") as f:
        f.write(str(os.getpid()))

    # Create the environment
    env = suite.make(
        args.env,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=1000,
        use_camera_obs=args.vision,
        use_object_obs=not(args.vision),
        camera_heights=args.size,
        camera_widths=args.size,
    )

    eval_env = suite.make(
        args.env,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=1000,
        use_camera_obs=args.vision,
        use_object_obs=not(args.vision),
        camera_heights=args.size,
        camera_widths=args.size,
    )

    if args.env == "Hanoi":
        env.random_reset = True
        eval_env.random_reset = True

    # Wrap the environment
    if args.vision:
        env = VisualizationWrapper(env)
        eval_env = VisualizationWrapper(eval_env)
    env = GymWrapper(env, proprio_obs=False)
    eval_env = GymWrapper(eval_env, proprio_obs=False)
    env = op_to_wrapper[args.env][args.op](env, horizon=100, image_obs=args.vision)
    eval_env = op_to_wrapper[args.env][args.op](eval_env, horizon=100, image_obs=args.vision)
    if args.vision:
        print("Using vision wrapper")
        env = vision_wrapper[args.env](env)
        eval_env = vision_wrapper[args.env](eval_env)
    else:
        print("Using object state wrapper")
        env = object_state_wrapper[args.env](env)
        eval_env = object_state_wrapper[args.env](eval_env)
        env.relative_obs = True

    check_env(env)
    #env = Monitor(env, filename=None, allow_early_resets=True)
    #eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

    # Trains the policy
    model = learn_policy(args, env, eval_env, args.op)
    # Deletes the model
    del model
