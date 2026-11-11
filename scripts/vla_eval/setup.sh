#!/usr/bin/env bash
# One-time setup on a Linux machine with an NVIDIA GPU (>= 16 GB, e.g. RTX 4090 / A10 / L4 / A100).
# Run from the repository root:  bash scripts/vla_eval/setup.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# System packages: EGL for headless MuJoCo rendering, bison/flex in case Metric-FF must be rebuilt.
if command -v apt-get >/dev/null; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git git-lfs build-essential bison flex libegl1 libgl1 libglib2.0-0 ffmpeg >/dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

git submodule update --init robosuite tarski openpi

# 1) Policy server environment (openpi, Python 3.11, JAX with CUDA).
(cd openpi && git submodule update --init && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .)

# 2) Simulation client environment (same Python 3.9 environment as the NSM evaluation).
uv python install 3.9
[ -d .venv ] || uv venv --python-preference only-managed --python 3.9 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e tarski --no-deps
uv pip install --python .venv/bin/python -e robosuite --no-deps
uv pip install --python .venv/bin/python "msgpack>=1.0.5" "websockets>=11.0" "dm-tree>=0.1.8" huggingface_hub

# 3) Planner + task configs. The tracked Metric-FF binary is a Linux x86-64 build; it is
#    rebuilt only if the planner tests fail with it.
export PYTHONPATH="$PWD/robosuite/src:$PWD:$PWD/openpi/packages/openpi-client/src" MUJOCO_GL=egl
uv pip install --python .venv/bin/python -q pytest
if ! .venv/bin/python -m pytest -q tests/test_task_configs.py -x -k initial_plan >/dev/null 2>&1; then
  echo "Rebuilding Metric-FF..."
  (cd Metric-FF-v2.1 && (make clean >/dev/null 2>&1 || true) && make)
fi
.venv/bin/python -m pytest -q tests/test_task_configs.py -x

# 4) The simulator renders headless and the policy server sees the GPU.
.venv/bin/python - <<'EOF'
import robosuite as suite
from planning.task_configs import PAPER_TASKS, make_env
env = make_env(PAPER_TASKS["hanoi_3"], has_offscreen_renderer=True, use_camera_obs=True,
               camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
               controller_configs=suite.load_controller_config(default_controller="OSC_POSE"))
obs = env.reset()
print("render ok:", obs["agentview_image"].shape, "action_dim:", env.action_dim)
EOF
(cd openpi && uv run python -c "import jax; print('jax devices:', jax.devices())")
echo "Setup done. Next: bash scripts/vla_eval/run_all.sh"
