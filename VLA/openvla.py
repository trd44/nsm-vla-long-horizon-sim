from robosuite.utils.observables import Observable
import torch
import mimicgen
import numpy as np
import gymnasium as gym
from PIL import Image
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import *
from mimicgen.envs.robosuite.coffee import *
import os

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModel

from robosuite.models.arenas import TableArena

# torch.cuda.empty_cache()
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load Processor & VLA
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # [Optional] Requires `flash_attn`
    # attn_implementation="flash_attention_2",
    # torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda")

# vla = AutoModel.from_pretrained(
#     "Stanford-ILIAD/minivla-vq-bridge-prismatic"
#     ).to("cuda")

# Load the OSC_POSE controller configuration
controller_config = load_controller_config(default_controller="OSC_POSE")

env = suite.make(
    env_name='MugCleanup',
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="third_person",
    camera_names=["third_person"],
    camera_heights=480,
    camera_widths=640,
    controller_configs=controller_config,
)

observation = env.reset()
done = False
step = 0
print(observation)
image = Image.fromarray(np.uint8(observation['third_person_image']))
image.show()
while not done:
    print(step)
    step += 1
    env.render()
    image = Image.fromarray(np.uint8(observation['third_person_image']))
    image.show()
    INSTRUCTION = "Open the drawer."
    prompt = "In: What action should the robot take to {INSTRUCTION}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    # This action is del_x, del_ygit , del_z, del_roll, del_pitch, del_yaw, gripper
    # Need to convert to the joint angle and gripper action space the robot is expecting.
    action = vla.predict_action(
        **inputs, unnorm_key="bridge_orig", do_sample=False)
    print(action)
    # action = np.append(action, 0.0)

    observation, reward, done, info = env.step(action)
