# noise_reachability step 0: actions executed per inference (Kaggle)

Step 0 of `noise_reachability/README.md`. `lerobot/smolvla_libero@a24f1f3804` with `n_action_steps` = 10 and
50 against the `n_action_steps=1` run (`week6/smolvla_libero_lerobot_eval_kaggle/`, 171/300), everything
else unchanged: init states 0-9 x seeds 0-2, 520 steps, 256x256, the camera rename map. Kaggle 2 x T4,
`baseline_performance/kaggle_baseline_lerobot_eval.ipynb` at commit `d164aba` with a clean tree
(`provenance.jsonl` in each folder).

Stated before the runs: on tasks 0, 4, 6, 8, `n_action_steps=10` raises success from 27/120 to at least
39/120; a setting qualifies if ahead with exact McNemar p < 0.025, and the winner replaces the baseline unless
it is behind 171/300 on all 10 tasks with p < 0.05.

## Result: `n_action_steps=10` is the new baseline, 216/300 (72.0%)

Weak tasks (`step0_decision_weak_tasks.txt`), out of 30 per task:

| Task | 1 | 10 | 50 |
|---|---|---|---|
| 0 | 1 | 7 | 5 |
| 4 | 10 | 14 | 11 |
| 6 | 14 | 20 | 12 |
| 8 | 2 | 16 | 9 |
| Total of 120 | 27 (22.5%) | 57 (47.5%) | 37 (30.8%) |
| Seeds 0 / 1 / 2 | | 20 / 19 / 18 | 11 / 13 / 13 |

- `n_action_steps=10`: 40 episodes succeed only at 10, 10 only at 1; exact McNemar p < 0.0001; paired
  difference +25.0 points, exact 95% interval +13.6 to +33.3. Qualifies; the hypothesis is supported.
- `n_action_steps=50`: 24 against 14, p = 0.1433; +8.3 points, interval -2.5 to +17.9. Does not qualify.
- 10 against 50 directly (`weak_tasks_nas10_vs_nas50.txt`): 33 against 13, p = 0.0045.

All 10 tasks at `n_action_steps=10` (`all_tasks_nas10_vs_nas1.txt`):

| | `n_action_steps=1` | `n_action_steps=10` |
|---|---|---|
| Total | 171/300 (57.0%) | 216/300 (72.0%) |
| Seeds 0 / 1 / 2 | 58 / 58 / 55 | 73 / 71 / 72 |
| Mean ± std over seeds | 57.0 ± 1.7% | 72.0 ± 1.0% |

- Paired: 62 episodes succeed only at 10, 17 only at 1; exact McNemar p < 0.0001; +15.0 points, exact 95%
  interval +9.4 to +19.5. Not behind 171/300, so `n_action_steps=10` is the baseline for steps 1-4.
- Per task (of 30), 1 -> 10: 0: 1 -> 7, 1: 23 -> 28, 2: 29 -> 30, 3: 27 -> 29, 4: 10 -> 14, 5: 26 -> 28,
  6: 14 -> 20, 7: 17 -> 18, 8: 2 -> 15, 9: 22 -> 27. No task is lower.
- The all-task file's last line is the script's generic decision text; the all-task run is that check.

Repeat (`repeat_nas10_weak_tasks.txt`): the all-task run repeats the weak-task runs at `n_action_steps=10`
with the same settings on the same machine type (`nas10_all_tasks/per_episode_tasks_0_4_6_8.csv` against
`nas10_weak_tasks/per_episode.csv`). 57/120 and 56/120; 115 of 120 episodes had the same outcome (3 and 2
changed), and no init state was solved in one run and never in the other.

Time (`evaluate_all.runner.log`): the 12 weak-task runs finished after 32.2 minutes at 10 and 20.8 at 50.
All 10 tasks took 64.7 minutes at 10 against 493.7 at 1 (`week6/smolvla_libero_lerobot_eval_kaggle/`).

## What it means

- The gap between our 57.0% and the paper's 71% on LIBERO-Long was the evaluation setting, not the checkpoint.
  `n_action_steps=1` came from LIBERO-CTRL's protocol, whose 56.8% has the same cause.
  `baseline_performance/README.md` is corrected.
- Earlier comparisons with `lerobot/smolvla_libero` (continued training, self-filtered BC, residual SAC pilots)
  were made at `n_action_steps=1`. Their verdicts hold for that setting; whether they hold at 10 was not tested.
- Re-planning less often helps up to a point: 50 actions per inference (the checkpoint's own default) is
  worse than 10.

## Files

- `nas10_weak_tasks/`, `nas50_weak_tasks/`, `nas10_all_tasks/`: the three Kaggle versions (results zips of
  the notebook): `per_episode.csv`, `consolidated_*`, `evaluate_all.runner.log`, `provenance.jsonl`,
  `task_<t>_seed_<s>/eval_info.json` and gzipped `eval.log`, `_timing/`, and the notebook's own step-9
  comparison (`comparison_with_n_action_steps_1.txt`).
- `step0_decision_weak_tasks.txt` (+ `.json`), `all_tasks_nas10_vs_nas1.txt` (+ `.json`):
  `python -S noise_reachability/compare_action_steps.py week6/smolvla_libero_lerobot_eval_kaggle/per_episode.csv
  10=<run>/per_episode.csv [50=<run>/per_episode.csv] --out <file>`.
- `weak_tasks_nas10_vs_nas50.txt`, `repeat_nas10_weak_tasks.txt`: `python -S
  baseline_performance/compare_per_state.py <a> <b> --out <file>`.
