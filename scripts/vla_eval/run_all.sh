#!/usr/bin/env bash
# Run the paper's 12 VLA conditions under the NSM evaluation protocol (seeds 5000-5049).
# For each fine-tuned checkpoint: download params + norm stats from Hugging Face, serve it
# with openpi, run every condition that uses it, stop the server, delete the weights.
# Finished runs (a complete .summary.json with all episodes) are skipped, so the script
# can simply be restarted after an interruption.
#
#   bash scripts/vla_eval/run_all.sh                 # everything
#   EPISODES=2 bash scripts/vla_eval/run_all.sh hanoi_3   # quick check on one checkpoint
#
# Env vars: EPISODES (50), SEED (5000), RESULTS (results/vla_5000), PARALLEL (2 sim clients
# per server), PORT (8000), CKPT_DIR (checkpoints), KEEP_WEIGHTS=1 to keep downloads.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EPISODES=${EPISODES:-50}; SEED=${SEED:-5000}; RESULTS=${RESULTS:-results/vla_5000}
PARALLEL=${PARALLEL:-2}; PORT=${PORT:-8000}; CKPT_DIR=${CKPT_DIR:-checkpoints}
export PYTHONPATH="$PWD/robosuite/src:$PWD:$PWD/openpi/packages/openpi-client/src" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$RESULTS/logs" "$CKPT_DIR"

# checkpoint key -> tasks run with it (first = fine-tuned condition, rest = zero-shot)
declare -A RUNS=(
  [sort_2color]="sort_2color sort_2color_swapped sort_3color"
  [tower_3_set]="tower_3_set tower_3_random tower_4_set"
  [hanoi_3]="hanoi_3 hanoi_4"
  [sort_3color]="sort_3color"
  [tower_3_random]="tower_3_random"
  [tower_4_set]="tower_4_set"
  [hanoi_4]="hanoi_4"
)
ORDER=(${@:-hanoi_3 sort_2color tower_3_set sort_3color tower_3_random tower_4_set hanoi_4})

info() { .venv/bin/python -c "from analysis.experiments_vla import CHECKPOINTS as C; print(C['$1'][$2])"; }
done_run() {  # 0 if the run's summary already has all episodes
  local s="$RESULTS/${1}_vla_${2}_${3}_seed${SEED}.summary.json"
  [ -f "$s" ] && .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$s'))['episodes']>=$EPISODES else 1)"
}

for key in "${ORDER[@]}"; do
  repo=$(info "$key" 0); config=$(info "$key" 1)
  todo=()
  for task in ${RUNS[$key]}; do
    cond=$([ "$task" = "$key" ] && echo finetuned || echo zeroshot)
    done_run "$task" "$cond" "$key" || todo+=("$task")
  done
  [ ${#todo[@]} -eq 0 ] && { echo "== $key: all runs done"; continue; }
  echo "== $key ($repo, $config): ${todo[*]}"

  dir="$CKPT_DIR/$key"
  .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('$repo', local_dir='$dir', allow_patterns=['params/*','assets/*','_CHECKPOINT_METADATA'])"

  (cd openpi && exec uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
      --policy.config "$config" --policy.dir "$(cd .. && realpath "$dir")") > "$RESULTS/logs/server_${key}.log" 2>&1 &
  server=$!
  trap 'kill $server 2>/dev/null || true' EXIT
  echo "   waiting for policy server (loading ~6 GB of weights)..."
  until .venv/bin/python -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 1)" 2>/dev/null; do
    kill -0 $server 2>/dev/null || { echo "server exited; see $RESULTS/logs/server_${key}.log"; tail -30 "$RESULTS/logs/server_${key}.log"; exit 1; }
    sleep 5
  done

  # Split each run into PARALLEL seed chunks; all chunks share the one server.
  per=$(( (EPISODES + PARALLEL - 1) / PARALLEL ))
  for task in "${todo[@]}"; do
    pids=()
    for ((c = 0; c < PARALLEL; c++)); do
      start=$(( SEED + c * per )); n=$(( EPISODES - c * per )); [ $n -gt $per ] && n=$per; [ $n -le 0 ] && continue
      .venv/bin/python -u analysis/experiments_vla.py --task "$task" --checkpoint "$key" --episodes $n \
          --seed $start --port "$PORT" --results_dir "$RESULTS/chunks" --overwrite \
          > "$RESULTS/logs/${task}_${key}_${start}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || echo "   chunk failed: see $RESULTS/logs/ (${task})"; done
    .venv/bin/python scripts/vla_eval/merge.py --results "$RESULTS" --task "$task" --checkpoint "$key" \
        --seed "$SEED" --episodes "$EPISODES"
  done

  kill $server 2>/dev/null || true; wait $server 2>/dev/null || true; trap - EXIT
  [ "${KEEP_WEIGHTS:-0}" = 1 ] || rm -rf "$dir"
done
.venv/bin/python scripts/vla_eval/merge.py --results "$RESULTS" --table
