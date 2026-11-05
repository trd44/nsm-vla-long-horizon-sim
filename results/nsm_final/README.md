# NSM evaluation on held-out seeds

Code at commit `8296be8`, Kinova3, 50 episodes per task, seeds 1000-1049, none used for
tuning or training. Each `.jsonl` holds one row per episode; each `.summary.json` holds totals.

| Task | Paper success / adv. | Vision success (Wilson 95% CI) | Vision adv. | GT success | GT adv. |
|---|---|---|---|---|---|
| `sort_2color` | 100 / 100 | 86 (74-93) | 95.0 | 96 | 99.0 |
| `tower_3_set` | 100 / 100 | 98 (90-100) | 99.3 | 98 | 99.3 |
| `hanoi_3` | 100 / 100 | 100 (93-100) | 100.0 | 100 | 100.0 |
| `sort_2color_swapped` | 100 / 100 | 92 (81-97) | 96.0 | 98 | 100.0 |
| `sort_3color` | 86 / 96.5 | 100 (93-100) | 100.0 | 100 | 100.0 |
| `tower_3_random` | 100 / 100 | 92 (81-97) | 97.3 | 96 | 99.3 |
| `tower_4_set` | 97 / 99.5 | 74 (60-84) | 77.5 | 78 | 80.5 |
| `hanoi_4` | 78 / 84.4 | 62 (48-74) | 82.7 | 98 | 98.1 |
| Complex-task average | 92.2 / 96.1 | 84.0 | 90.7 | 94.0 | 95.6 |

`vision` is YOLO plus `nsm_regressor_rollout.pkl`, with no simulator fallback for cube
positions in any episode (every `[GT-FALLBACK]` summary read `NONE`). Simulator state still
supplies destination positions, the planner's initial state, termination and goal checks,
the grasp override, and object identity when colours repeat. `gt` feeds simulator cube
poses to the policies and is not a vision result.

## Known causes

- `tower_4_set`: the first failed skill was Pick in 11 of 13 vision failures and all 11 GT
  failures, on the same 11 seeds. All 10 layouts with neighbouring cubes under 90 mm apart
  failed in GT, against 1 of 40 otherwise. In seed 1001, a contact log showed a fingertip
  on the neighbouring cube for 352 of 605 steps. The threshold was found after seeing these
  results, and the contact log covers that one failing seed plus one passing seed.
- `hanoi_4`: 18 seeds failed only in vision and none failed only in GT, so this gap is perception.
- `sort_3color`: colours are fixed as r, g, b, r, which may be easier than the paper's setup.
