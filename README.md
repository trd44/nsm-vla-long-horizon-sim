# Finetuning Pi0

## Create training config
Record your demo and put it in `.datasets/{task_name}`
Create training config in `openpi/src/openpi/training/config.py`
Add a train config, make necessary modifications. Pay special attention to the `extra_

```bash
    TrainConfig(
        name="pi0_{task_name}",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="{username}/{task_name}",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=16,  # Reduced from default 32 to save memory
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        instruction_override="Assemble the block tower.",
        # num_workers=0,
    ),
```

## Calculate Norm Stats
Calculate the norm stats of your data
```bash
uv run scripts/compute_norm_stats.py --config-name {TrainConfig}
```
The norm stats will show up in `openpi/assets/{TrainConfig}/`

## Finetune
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py {TrainConfig} --exp-name=my_experiment --overwrite
```
## Run Inferece on Finetuned Model
First modify the `openpi/examples/robosuite/compose.yml` to point to the name of the finetuned checkpoint. This is the same as the TrainConfig above.
```
environment:
    - SERVER_ARGS=policy:checkpoint --policy.config {TrainConfig} --policy.dir /app/checkpoints/{TrainConfig}/{TrainConfig}/29999
```
if the policy is plan guided, set it to true in `openpi/examples/robosuite/args.py`
```
planner_guided: bool =True
```
then docker compose
```
docker compose -f examples/robosuite/compose.yml up
```
## Dependencies

Clone MimicGen in /OpLearn directory
```bash
git clone https://github.com/helenlu66/mimicgen
```

Clone RoboSuite in /OpLearn directory
```bash
git clone https://github.com/helenlu66/robosuite.git
```

Clone tarski in /OpLearn directory. This is a symbolic planning parsing library.
```bash
git clone https://github.com/helenlu66/tarski.git
```

Install required libraries
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Docker Container (Optional)
There is an optional docker container for you to use. The docker-compose is setup to use Nvidia GPUs. You will need to install the Nvidia Container Toolkit to use this. 

### Installing Nvidia Container Toolkit
Instructions for installing the Nvidia Container Toolkit can be found here:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 

### Building and running the Docker container

Building
```bash
docker compose build
```

Running
```bash
docker compose up
```

Closing runaway containers. If you get an error on a subsequent docker build. It's likely because part of the container is still running in the background. You can use this command to stop it.

```bash
docker compose down -v
```

### Add your OpenAI API key to the container as an environment variable
Replace ```YOUR_API_KEY``` with the API key you get from OpenAI
```bash
echo 'export OPENAI_API_KEY="YOUR_API_KEY"' >> ~/.bashrc
```

Source your .bashrc file for the changes to take effect
```bash
source ~/.bashrc
```

### Connecting to VS Code (Optional)
Install the dev containers extension in VS Code

While the container is up, press F1 in VS Code and type in the command pallete "Dev Containers: Attach to running container..."

Select the running container. In my case it says "/oplearn_oplearn_1"

Open the /home/user/oplearn directory to start working on the files from within the container.

## Test Installation
You can test your mujoco installation first with 
```bash
python test_mujoco.py
```

You can test your Robosuite installation with 
```bash
python robosuite/robosuite/demos/demo_random_action.py
```

You can test your MimicGen installation with 
```bash
python mimicgen/mimicgen/scripts/demo_random_action.py
```

## Testing Environments
You can test our environments and control the manipulator with your keyboard with:
```bash
python keboard_control_envs.py
```

Follow the instructios to chose the environment and robot you want. The keyboard controls will be displayed after running this script also.

## Running the Hybrid Planning and Learning Agent
```bash
python hybrid_planning_learning_agent.py
```

Use config.yaml to edit the configuration of the agent.

Use visualize_policy.py to view the policy the agent learned.
