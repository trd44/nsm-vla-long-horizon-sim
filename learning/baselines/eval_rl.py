import argparse
import numpy as np
import os
import robosuite as suite
import time
import warnings
from datetime import datetime
from robosuite.wrappers.gym_wrapper import GymWrapper
from custom_rl_callback import CustomEvalCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
warnings.filterwarnings("ignore")

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

def learn_policy(args, env, eval_env, name):
    # Define the model
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.lr,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed
    )
    print("Saving the model in: {}, as best_model.zip and final model {}".format(args.modeldir, os.path.join(args.bufferdir, 'task' + '_sac')))
    # Define all callbacks
    callbacks = []
    # Define the evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=args.modeldir,
        log_path=args.logdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    # Add a stop callback on success rate of 100%
    callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=0.99, verbose=1))
    # Add a stop callback on success rate of 100%
    callbacks.append(StopTrainingOnNoModelImprovement(check_freq=1000, max_no_improvement=1_000_000, verbose=1))

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks
    )
    # Save the model
    model.save(os.path.join(args.modeldir, name + '_sac'))
    return model

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='her_symbolic_augmented', choices=['her', 'sac'], 
                        help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--timesteps', type=int, default=int(5e5), help='Number of timesteps to train for')
    parser.add_argument('--eval_freq', type=int, default=20000, help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--no_transfer', action='store_true', help='No transfer learning')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate') # 0.00005 0.00001
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dense', action='store_true', help='Use dense reward')
    parser.add_argument('--init_policy', type=str, default=None, help='Path to initial policy')
    parser.add_argument('--novelty', type=str, default=None, help='Novelty to learn')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)

    # Define the evaluation frequency
    args.eval_freq = min(args.eval_freq, args.timesteps)
    n_eval_between_novelty = 20 if args.n_eval_episodes > 1 else 1

    # Define the directories
    data_folder = args.data_folder
    experiment_name = args.experiment + '_dense_' + str(args.dense) + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid
    if args.name is not None:
        experiment_id = args.name
    args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

    # Create the directories
    os.makedirs(args.experiment_dir, exist_ok=True)

    # Save args in a txt file
    with open(os.path.join(args.experiment_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
        f.close()


    for i in range(len(list_of_novelties)):
        # Create the directories
        args.logdir = os.path.join(args.experiment_dir, list_of_novelties[i], 'logs')
        args.modeldir = os.path.join(args.experiment_dir, list_of_novelties[i], 'models')
        args.bufferdir = os.path.join(args.experiment_dir, list_of_novelties[i], 'buffers')
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.modeldir, exist_ok=True)
        os.makedirs(args.bufferdir, exist_ok=True)

        # Test if the file 'best_model.zip' already exists in the folder './models/'+ args.experiment + '/'
        if list_of_novelties[i] == 'PickPlaceCan' and (os.path.isfile('./models/'+ args.experiment + '/best_model.zip') or (args.no_transfer)):
            continue
        print("\nNovelty: {}".format(list_of_novelties[i]))
        # Create the environment
        env = suite.make(
            list_of_novelties[i],
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=1000,
            render_camera="agentview",
        )
        eval_env = suite.make(
            list_of_novelties[i],
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=1000,
            render_camera="agentview",
        )


        #check_env(env)
        env = Monitor(env, filename=None, allow_early_resets=True)
        eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

        # Trains the policy
        model = learn_policy(args, env, eval_env, list_of_novelties[i])
        # Deletes the model
        del model