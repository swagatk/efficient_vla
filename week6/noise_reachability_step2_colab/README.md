# noise_reachability step 2: training-free noise selection per chunk (Colab)

Step 2 of `noise_reachability/README.md`. `lerobot/smolvla_libero` at `n_action_steps=10`, 520 steps, the
camera rename map, run through `noise_reachability/eval_noise_selection.py` (`vla_core` harness, base and
selector paired on the same environment and torch seed per (seed, init state)). Colab A100-SXM4-40GB,
`noise_reachability/colab_step2_noise_selection.ipynb`, one process per task in parallel. Each stage ran from
one clean commit (`provenance.jsonl`): the select stage at `9b29d2a`, the report stage at `0972349`; the
`noise_reachability/` and `vla_core/` code is identical at both.

Stated before the run: the selector chosen on init states 10-19 raises success over the paired base on init
states 0-9 x seeds 0-2 by at least 10 points (exact McNemar p < 0.05), and solves at most 1 in 10 of the 24
systematic episodes of step 1.

Both folders pass `tools/validate_run.py` (one clean commit, `n_action_steps=10`, every (task, seed, init
state) exactly once per policy, no tracebacks).

## Selection stage (`step2_select.txt`): `accel_n8` chosen

Tasks 0, 4, 6, 7, 8 x init states 10-19 x seeds 0-1, 100 paired episodes per policy.

| Policy | Successes | Only base | Only policy | McNemar p |
|---|---|---|---|---|
| base | 52 | | | |
| first_n8 (no selection) | 54 | 10 | 12 | 0.83 |
| medoid_n8 | 53 | 11 | 12 | 1.00 |
| accel_n8 | 55 | 7 | 10 | 0.63 |
| medoid_n16 | 48 | 16 | 12 | 0.57 |
| accel_n16 | 55 | 9 | 12 | 0.66 |

By the rule fixed in the README, `accel_n8` (tied with `accel_n16`, ties to the smaller N) is ahead of base, so
the report stage ran. The margin is within noise: `first_n8`, which selects nothing, is +2.

## Report stage (`step2_report.txt`): hypothesis not supported

Tasks 0, 4, 6, 7, 8 x init states 0-9 x seeds 0-2, 150 paired episodes per policy.

| Task | base | accel_n8 |
|---|---|---|
| 0 | 5/30 | 7/30 |
| 4 | 16/30 | 16/30 |
| 6 | 16/30 | 17/30 |
| 7 | 18/30 | 19/30 |
| 8 | 9/30 | 12/30 |
| All | 64/150 | 71/150 |
| Step-1 systematic episodes | 0/24 | 0/24 |

- Paired difference +4.7 points, exact 95% interval -3.5 to +11.9; only base 13, only `accel_n8` 20; exact
  McNemar p = 0.30. The gain of at least 10 points with p < 0.05 is not supported.
- Systematic episodes solved at most 1 in 10: holds (0 of 24 for both), so the gain, such as it is, is on
  reachable states.
- Harness check: base 64/150 here against 74/150 through lerobot-eval on the same episodes (step 0, different
  noise draws, not paired); the gap of 10 is within the 17 allowed.
- Mean episode length: base 414 steps, `accel_n8` 404 (from `eval.log`).

## What it means

Choosing among 8 noise samples per chunk with an untrained score does not separate good chunks from bad ones
well enough to matter here: the effect is at most a few points (the interval excludes more than +12), similar
to executing an arbitrary one of 8 samples, and neither policy solved a systematic state. Step 1's headroom
(0.697) is real, but reaching it needs a judge that knows more than the flow's own curvature or the samples'
agreement. Step 3 (a noise policy trained on task success, tasks 7 and 0) goes ahead as planned; the base rows
here (task 7: 18/30, task 0: 5/30) are its paired baseline on init states 0-9 x seeds 0-2.

## Files

- `select/`, `report/`: the Colab result folders: `task_<t>/per_episode_<policy>.csv`, `eval.log` (one line
  per episode, with `inferences` and `chosen_per_sample` for selectors), `provenance.jsonl`; `runner.log`,
  `repo_commit.txt`. In both stages the last worker process started 82 (select) and 37 (report) minutes after
  the first (`provenance.jsonl` dates).
- `step2_select.txt` (+ `.json`): `python -S noise_reachability/analyze_step2.py --stage select --root
  week6/noise_reachability_step2_colab/select --policies
  base,first_n8,medoid_n8,accel_n8,medoid_n16,accel_n16 --out <file>`.
- `step2_report.txt` (+ `.json`): `python -S noise_reachability/analyze_step2.py --stage report --root
  week6/noise_reachability_step2_colab/report --policies base,accel_n8 --step1
  week6/noise_reachability_step1_kaggle/step1_reachability.txt.json --lerobot_eval
  week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv --out <file>`.
