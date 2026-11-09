# NSM final evaluation: seeds 5000-5049

This is the final evaluation of the rebuilt NSM pipeline (branch `nsm-rebuild`).

- **Setup:** Kinova3, original diffusion checkpoints, 50 episodes per task, seeds 5000-5049. These seeds were never run before this evaluation, and nothing was tuned on them.
- **How the runs were split:** each task ran as five 10-episode chunks, `*_seed5000` to `*_seed5040`. Episode *i* of a chunk uses seed `chunk + i`, so the chunks together cover exactly seeds 5000-5049.
- **Files:**
  - `vision/` and `gt/` hold one `.jsonl` row per episode, plus a `.summary.json` per chunk.
  - `logs/` holds the full run logs.

## Results

Success rates are in %, with Wilson 95% intervals in brackets; "adv." is advancement in %.

| Task | Paper success / adv. | Vision success | Vision adv. | GT success | GT adv. |
|---|---|---|---|---|---|
| `sort_2color` | 100 / 100 | 98 (90-100) | 99.5 | 92 (81-97) | 98.5 |
| `tower_3_set` | 100 / 100 | 92 (81-97) | 96.0 | 100 (93-100) | 100.0 |
| `hanoi_3` | 100 / 100 | 100 (93-100) | 100.0 | 100 (93-100) | 100.0 |
| `sort_2color_swapped` | 100 / 100 | 96 (87-99) | 99.0 | 100 (93-100) | 100.0 |
| `sort_3color` | 86 / 96.5 | 100 (93-100) | 100.0 | 100 (93-100) | 100.0 |
| `tower_3_random` | 100 / 100 | 94 (84-98) | 98.0 | 98 (90-100) | 99.3 |
| `tower_4_set` | 97 / 99.5 | 78 (65-87) | 82.5 | 78 (65-87) | 84.5 |
| `hanoi_4` | 78 / 84.4 | 84 (71-92) | 91.1 | 98 (90-100) | 98.1 |
| Basic-task average | 100 / 100 | 96.7 | 98.5 | 97.3 | 99.5 |
| Complex-task average | 92.2 / 96.1 | 90.4 | 94.1 | 94.8 | 96.4 |

## What "vision" means here

- **No simulator positions:** `vision` reads no object position from the simulator, and there is no fallback.
- **Cubes:** cube positions come from YOLO (`hanoi_yolo.pt`) plus `nsm_regressor_rollout.pkl`. A held cube is placed at the gripper.
- **Fixtures:** pegs, platforms and bins come from the task's workcell map in `planning/task_configs.py`. That map is tested equal to the environments.
- **Cube identity:** identity comes from colour, vision memory and the symbolic state.
- **Symbolic state:** `on`, `clear`, `grasped` and `type_match` come from the simulator's predicate detector, as in the paper.
- **`gt`:** feeds simulator poses to the policies. It is not a vision result.

## Code versions

The vision runs used two commits.

| Runs | Commit | Notes |
|---|---|---|
| `hanoi_4`, `tower_3_set`, `tower_3_random`, `tower_4_set` (vision) | `27d71b1` | Fixtures were read from the simulator, but their values are bit-identical to the workcell map. Cube identity was already decided by colour, with the simulator-based matches discarded by the colour lock. |
| Sort tasks, `hanoi_3` (vision); all GT runs | `42e922e` | |

For the `27d71b1` runs:
- Their logs print a `[GT-FALLBACK]` summary. It lists the fixtures only; the manipulable-object line reads `NONE`.
- Rerunning 5 of their episodes at `42e922e` reproduced the records exactly: `hanoi_4` seeds 5000-5001 and `tower_3_set` seeds 5000-5002.

Sort vision runs made at `27d71b1` were discarded rather than reported. At that commit, same-coloured cubes were matched against simulator positions.

## Failures

- **`tower_4_set`:** vision and GT fail on largely the same seeds: 5003, 5004, 5012, 5013, 5015, 5016, 5021, 5024, 5042 and 5047. Most of these failures are a Pick that makes no progress. This is the cube-spawn issue described in `results/nsm_final/README.md`: uniform ±2.5 cm noise sometimes leaves a neighbouring cube under a fingertip. It was left unfixed by decision.
- **`hanoi_4` vision:** 8 failures, spread over ReachDrop misplacements, preconditions found unmet after a misplaced cube, and two ReachPick timeouts. GT fails once.
- **`tower_3_set` vision:** 4 ReachDrop failures (2 misplacements, 2 preconditions found unmet). GT is 50/50.
- **Sort tasks:** failures are cubes placed outside a bin. Same-colour identity swaps (`[IDENTITY]` in the logs) did not cause any failure.
