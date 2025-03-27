import argparse
import numpy as np
import robosuite as suite
import time
import warnings
import torch
from datetime import datetime
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper
from stable_baselines3 import SAC, PPO

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


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hanoi', choices=['Hanoi', 'KitchenEnv', 'NutAssembly'], help='Name of the environment to run the experiment in')
    parser.add_argument('--op', type=str, default='pick', choices=['pick', 'place', 'turnon', 'turnoff'], help='Name of the operator to train the policy for')
    parser.add_argument('--path', type=str, default='./data/', help='Path to the policy to test')
    parser.add_argument('--vision', action='store_true', help='Use vision based learning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)

    # Create the environment
    env = suite.make(
        args.env,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        horizon=1000,
        use_camera_obs=args.vision,
        use_object_obs=not(args.vision),
        camera_heights=256,
        camera_widths=256,
    )

    if args.env == "Hanoi":
        env.random_reset = False

    # Wrap the environment
    if args.vision:
        env = VisualizationWrapper(env)
    env = GymWrapper(env, proprio_obs=False)
    env = op_to_wrapper[args.env][args.op](env, horizon=100, image_obs=args.vision)
    if args.vision:
        print("Using vision wrapper")
        env = vision_wrapper[args.env](env)
    else:
        print("Using object state wrapper")
        env = object_state_wrapper[args.env](env)

    # Load the trained policy
    model = SAC.load(args.path)

    # Evaluate the policy
    for i in range(5):
        obs, _ = env.reset()
        done = False
        print("Obj to pick: ", env.obj_to_pick, "Obj to place: ", env.place_to_drop)
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()
            time.sleep(0.01)

    env.close()
