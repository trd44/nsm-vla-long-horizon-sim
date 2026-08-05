# Finetuning Pi0

## Create training config
Record your demo and put it in `.datasets/{task_name}`
Create training config in `openpi/src/openpi/training/config.py`
Add a train config, make necessary modifications. Pay special attention to the `extra_delta_transform` boolean flag in the `DataConfig`

```bash
    TrainConfig(
        name="pi0_{task_name}",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="{username}/{task_name}",
            assets=AssetsConfig(
                asset_id="{username}/{task_name}",  # This should match the assets directory structure
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                offline_mode=True,  # Use only locally cached datasets, don't download from HuggingFace
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

Robosuite, robosuite-task-zoo and tarski are **git submodules** of this repo — do
not clone them separately. The old instructions pointed at `helenlu66` forks;
`.gitmodules` is authoritative.

```bash
git submodule update --init --recursive
```

### Policy checkpoints (Git LFS)

`policies/**/*.ckpt` are LFS-tracked. A plain clone leaves them as ~134-byte
pointer files and every run fails at `torch.load`. Fetch the real weights:

```bash
git lfs install --local && git lfs pull
```

Each checkpoint should be ~143 MB. If `git lfs` is unavailable (common on
clusters), grab a static binary from
https://github.com/git-lfs/git-lfs/releases and put it on your `PATH`.

### Python environment

Use Python 3.9 from a **managed** interpreter — the system Python on most
clusters ships no `Python.h`, and `evdev` in `requirements.txt` is source-only:

```bash
uv python install 3.9
uv venv --python-preference only-managed --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e tarski --no-deps
uv pip install -e robosuite --no-deps
uv pip install -e robosuite-task-zoo --no-deps
```

`requirements.txt` no longer contains `mujoco-py` (legacy — robosuite 1.4.1 uses
the `mujoco` bindings), `open3d` (no cp39 wheel, unused), `pybullet-svl` (build
failure; only needed for IK controllers, and we use `OSC_POSITION`) or the two
`-e git+` URLs (`mimicgen` is unused by the neuro-symbolic driver; `tarski` is a
submodule).

### diffusion_policy

`planning/executor.py` imports `diffusion_policy`, which is **not** vendored or
submoduled here. Clone it and put it on `PYTHONPATH`:

```bash
git clone https://github.com/real-stanford/diffusion_policy.git ../diffusion_policy
git -C ../diffusion_policy checkout 5ba07ac6661db573af695b419a7947ecb704690f
export PYTHONPATH="$(realpath ../diffusion_policy):$PWD"
```

### Metric-FF

`call_planner` shells out to `./Metric-FF-v2.1/ff` relative to the repo root, so
it must be built there:

```bash
cd Metric-FF-v2.1 && make && cd ..   # needs bison and flex
```

## Running the neuro-symbolic experiments

Headless rendering needs an offscreen GL backend (`egl` on a GPU node,
`osmesa` on CPU):

```bash
export MUJOCO_GL=egl
python -u analysis/experiments_neurosymbolic.py --env Hanoi --episodes 50
```

Perception is not switchable here: this driver always loads YOLO plus the
bbox→3D regressor and passes them to every executor. (The price-is-not-right
driver has a `--use_yolo` flag that gates this; this one does not.)
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
