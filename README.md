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

`planning/executor.py` imports `diffusion_policy`, a submodule pinned to
upstream commit `5ba07ac`. Fetch it with the other submodules and put it on
`PYTHONPATH`:

```bash
git submodule update --init diffusion_policy
export PYTHONPATH="$PWD/diffusion_policy:$PWD"
```

### Metric-FF

`call_planner` shells out to `./Metric-FF-v2.1/ff` relative to the repo root, so
it must be built there:

```bash
cd Metric-FF-v2.1 && make && cd ..   # needs bison and flex
```

## Running the neuro-symbolic experiments

Set `MUJOCO_GL` to the offscreen GL backend for your machine: `egl` on a GPU node,
`osmesa` on CPU, `cgl` on macOS. On a machine that can't run the tracked
`Metric-FF-v2.1/ff` binary, point `NSM_FF_BIN` at a native build.

```bash
export PYTHONPATH="$PWD/robosuite/src:$PWD:$PWD/diffusion_policy"
export MUJOCO_GL=egl OMP_NUM_THREADS=1
python -u analysis/experiments_neurosymbolic.py --task hanoi_4 --perception vision \
    --episodes 50 --seed 1000 --results_dir results/nsm
```

- `--task`: `sort_2color`, `tower_3_set`, `hanoi_3`, `sort_2color_swapped`,
  `sort_3color`, `tower_3_random`, `tower_4_set` or `hanoi_4`.
- `--perception vision` uses YOLO and `models/regressors/nsm_regressor_rollout.pkl`, and
  never reads an object position from the simulator (there is no fallback).
  `--perception gt` is the only way to feed simulator poses to the policies; it isolates
  policy behaviour and is not a vision result.
- `--policy_set original` (default) uses the checkpoints in `policies/hanoi_original/`.
- Each run writes per-episode JSONL and a summary with Wilson intervals against the
  paper's number. Logs print a `[PERCEPTION]` line saying which source was used.

Seed ranges: 0-9 were used for development, 1000+ are held out for evaluation, and
2000+ are for regressor training. `perception/collect_rollouts.py` refuses seeds below 2000.

In vision mode, positions come only from vision:
- Cubes come from YOLO plus the regressor; a held cube is at the gripper (proprioception).
- Pegs, bins and the platform are fixed fixtures with no YOLO class. Their positions come
  from each task's workcell map in `planning/task_configs.py`, which
  `tests/test_task_configs.py` checks against the environments. `--env_kwargs` that move
  the fixtures are refused in vision mode.
- Which cube is which comes from colour, where vision last saw each cube, and the
  symbolic state, never from simulator positions. When several cubes share a colour
  (sorting), symbolically identical cubes are interchangeable: if the robot reaches one
  other than the planned cube, the two names are swapped for the rest of the plan
  (`[IDENTITY]` in the log), as the relational grounding in paper §3.4.3 allows.

The symbolic state (`on`, `clear`, `grasped`, `type_match`) is given, as in the paper:
the planner's state, skill termination, goal checks and "which cube is held" read the
simulator's predicate detector, not vision.

OWLv2 distillation (paper §3.4.2) is available but not the default.
`perception/owl_label.py` labels frames with OWLv2 (boxes from OWLv2, class from an
added HSV colour step, since OWLv2 prompts label the yellow cube "green"), and
`perception/train_yolo_distill.py` trains and evaluates `models/yolo/nsm_yolo_owl.pt`
(use with `--yolo models/yolo/nsm_yolo_owl.pt --regressor models/regressors/nsm_regressor_owl.pkl`).
It is distilled offline, not during episodes as in the paper. It finds more cubes in
bins, in the gripper and in the wrist view, but in 8-episode dev pilots it scored
below `hanoi_yolo.pt` (hanoi_4 0/8 vs 3/8). `--owl_fallback` queries OWLv2 at runtime
for a cube YOLO has lost; it is experimental and off by default.

Object tracking: a cube is never mapped to a track of another colour, one track maps
to at most one cube, each cube's last vision position is shared across the four skills
(each skill still builds its own cube-to-track mapping), a held cube is placed at the
gripper, and a cube that is not held keeps its last vision position instead of timing
out. YOLO runs on MPS/CUDA when available (`NSM_YOLO_DEVICE` overrides); the diffusion
policy runs on CPU.

Rebuild the perception model with `perception/collect_data.py`,
`perception/train_regressor.py`, `perception/collect_rollouts.py` and
`perception/refit_rollouts.py`. Run the wiring tests with
`pytest tests/test_task_configs.py`.

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
