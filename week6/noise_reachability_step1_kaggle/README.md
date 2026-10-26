# noise_reachability step 1: per-state reachability over noise seeds (Kaggle)

Step 1 of `noise_reachability/README.md`. `lerobot/smolvla_libero@a24f1f3804` at `n_action_steps=10` (the
step-0 baseline) on tasks 0, 4, 6, 7, 8 x init states 0-9 x noise seeds 0-15: 800 episodes, of which seeds
0-2 come from step 0 (`week6/noise_reachability_step0_kaggle/nas10_all_tasks/`) and seeds 3-15 from this run.
Everything else as step 0: 520 steps, 256x256, the camera rename map. Kaggle 2 x T4,
`baseline_performance/kaggle_baseline_lerobot_eval.ipynb` at commit `08d33bc` with a clean tree
(`provenance.jsonl`); `n_action_steps': 10` appears in every `eval.log`.

Stated before the run: on the five tasks pooled, headroom = (pass@16 - pass@1) / (1 - pass@1) at least 0.5,
and at least 5 of the 50 init states systematic (0 of 16 seeds). Below 0.5, steps 2-3 are reconsidered; fewer
than 5 systematic states, step 4 is dropped.

## Result: both parts hold (`step1_reachability.txt`)

| Task | Successes | pass@1 | pass@16 | Systematic init states (0 of 16) |
|---|---|---|---|---|
| 0 | 41/160 | 0.256 | 0.500 | 2, 5, 6, 7, 8 |
| 4 | 79/160 | 0.494 | 0.900 | 2 |
| 6 | 97/160 | 0.606 | 0.900 | 8 |
| 7 | 90/160 | 0.562 | 1.000 | none |
| 8 | 70/160 | 0.438 | 0.900 | 4 |
| Pooled | 377/800 | 0.471 | 0.840 | 8 of 50 |

- Pooled pass@k: 0.471 (k=1), 0.604 (2), 0.719 (4), 0.795 (8), 0.840 (16).
- Headroom 0.697 (hypothesis >= 0.5: supported). Steps 2-3 go ahead.
- 8 systematic states (hypothesis >= 5: supported). Step 4 stays in the plan.
- Per seed, successes out of 50: 23.6 +- 2.5 over the 16 seeds (19 to 28); seeds 0-2 give 26, 26, 22 and
  seeds 3-15 together 303/650 (46.6%), so the two runs agree.
- Successes per init state out of 16 (states): 0 (8), 1 (3), 2 (2), 3 (1), 4 (3), 5 (5), 6 (2), 7 (2),
  8 (4), 9 (2), 10 (1), 11 (2), 12 (1), 13 (3), 14 (2), 15 (1), 16 (8).

Repeat (`nas10_step1_repeat/`): seed 0 run again on the same 50 episodes. 0 of 50 changed outcome, against 12.4
expected under a fresh noise seed (sum of 2p(1-p), p each state's rate over 16 seeds). Together with step 0's
repeat (5 of 120 changed, none on seed 0), Kaggle runs of one seed are close to deterministic: differences
between seeds are effects of the noise seed, not run-to-run variation.

## What it means

- About 70% of the failures of a single sample are reachable by the frozen policy with another noise seed;
  with an oracle that picks the better of 2 seeds, success would rise from 0.47 to 0.60. This is a ceiling for
  episode-level selection with a perfect judge, not a prediction for step 2, which selects per chunk with an
  untrained score.
- The systematic states are concentrated: 5 of 8 are on task 0, none on task 7.
- 0 of 16 does not mean probability 0: the exact one-sided 95% upper bound for a state with 0 of 16 is 17%, and 3 states
  succeeded in exactly 1 of 16. Step 4 should treat the systematic states as hard, not impossible.
- Step-3 tasks, by the rule fixed in the plan: task 7 (most reachable states, 10) and task 0 (most systematic
  states, 5).

## Side record: a version with the wrong setting (`side_nas1_wrong_setting/`)

The first step-1 version changed only `RUN_TAG` and `SEEDS` in the notebook, so it ran at `n_action_steps=1` on
all 10 tasks, seeds 3-15, and stopped at the 9-hour deadline after 35 of 130 evaluations (350 episodes; commit
`9c78f17`). It is not part of step 1. The notebook now sets every step-1 setting from `RUN_TAG` (commit `08d33bc`).

With seeds 0-2 of the `n_action_steps=1` baseline (`week6/smolvla_libero_lerobot_eval_kaggle/`), seeds 0-5 are
complete for every task. On the five step-1 tasks (`side_nas1_reachability_seeds0-5.txt`), against the same six
seeds at `n_action_steps=10` (`nas10_reachability_seeds0-5.txt`):

| Over seeds 0-5 | `n_action_steps=1` | `n_action_steps=10` |
|---|---|---|
| pass@1 | 0.283 | 0.477 |
| pass@6 | 0.520 | 0.760 |
| Headroom | 0.330 | 0.541 |
| Systematic states (0 of 6) | 24 of 50 | 12 of 50 |

Executing 10 actions per inference halves the number of states that no seed solves. The `n_action_steps=1`
seeds come from two Kaggle versions at different commits (seeds 0-2 and 3-5).

## Files

- `nas10_step1/`, `nas10_step1_repeat/`, `side_nas1_wrong_setting/`: the Kaggle results zips:
  `per_episode.csv`, `consolidated_*`, `evaluate_all.runner.log`, `provenance.jsonl`,
  `task_<t>_seed_<s>/eval_info.json` and gzipped `eval.log`, `_timing/`. `nas10_step1/reachability.txt` is the
  notebook's own step-9 output (without the repeat); it matches `step1_reachability.txt`.
- `step1_reachability.txt` (+ `.json`): `python -S noise_reachability/reachability.py --tasks 0 4 6 7 8
  --seeds 0-15 week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv
  week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv --original
  week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv --repeat
  week6/noise_reachability_step1_kaggle/nas10_step1_repeat/per_episode.csv --repeat_seed 0 --out <file>`.
- `side_nas1_reachability_seeds0-5.txt` (+ `.json`): the same script with `--seeds 0-5` on
  `week6/smolvla_libero_lerobot_eval_kaggle/per_episode.csv` and `side_nas1_wrong_setting/per_episode.csv`.
- `nas10_reachability_seeds0-5.txt` (+ `.json`): `--seeds 0-5` on the two `n_action_steps=10` files above.
