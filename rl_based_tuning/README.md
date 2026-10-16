# Residual RL on a frozen SmolVLA (LIBERO-10)

A small residual head is trained with SAC on top of the frozen base policy. The head's correction is added
to the base action in env action units, capped at `--residual_scale` per dimension.

## Hypothesis

Interaction adds information that the demonstrations the base policy was trained on do not contain, so a
residual trained by RL on one task should raise that task's success on held-out init states above the base
policy, measured on the same episodes. For the run to count as a result, the paired difference on held-out
init states 0-9 x 3 evaluation seeds (30 paired episodes per training seed, 3 training seeds) has to be
positive across training seeds, with an exact McNemar test on the pooled pairs. No target size is claimed
before the first valid run; week 5's numbers are not a valid baseline for one (see below).

## Files

- `train_online_sac.py`: training and paired evaluation (thin CLI; `--help` lists every flag).
- `sac_residual.py`: residual head, replay buffer and SAC update (no env, base policy or checkpoint I/O).
- `run_sac_residual.sh`: the exact commands for reported runs (train per seed, evaluate base once, evaluate
  each residual paired with the base episodes).
- Shared code in `vla_core/`: `envs.py` (LeRobot's `LiberoEnv`, built as lerobot-eval builds it),
  `policy.py` (frozen base policy with lerobot-eval's per-step processing), `residual.py`, `evaluation.py`.
- `kaggle_sac_residual.ipynb`: the task-6 pilot on Kaggle (stages `verify`, `train`, `eval`; see its first cell).
- `train_online_ppo.py`: retired, refuses to run (reasons below).

## Protocol

- Training episodes start from benchmark init states 10-49 (`--train_init_states`), evaluation uses 0-9,
  the init states of `baseline_performance/results_300926`; the script refuses overlapping sets.
- Evaluation: hard resets, deterministic residual (tanh of the mean), and the same env seed and noise seed
  for base and residual on each (seed, init state). Results are `per_episode_<policy>.csv` in the format of
  `baseline_performance/per_episode.csv`; `comparison_base_vs_residual.txt` is the output of
  `baseline_performance/compare_per_state.py`.
- Training uses soft resets (`--train_hard_reset 0`): one simulator for the whole run, which avoids the
  resource build-up behind FAIL-004; evaluation runs at most 10 episodes per process.
- The base policy defaults to `lerobot/smolvla_libero` with `--n_action_steps 1` and the `camera1`/`camera2`
  rename map: 57.0% on LIBERO-10 through our evaluation path, matching LIBERO-CTRL
  (`week6/smolvla_libero_lerobot_eval_kaggle/`). `--policy_path HuggingFaceVLA/smolvla_libero
  --n_action_steps -1 --rename_map '{}'` selects the older checkpoint behind `results_300926`.
- Multi-session runs: `--time_limit_hours` stops before a new episode once the limit is passed (exit code 3);
  the replay buffer is saved next to the checkpoint (`--save_replay 1`, about 2 GB) and restored on
  `--resume_training`.

Memory: about 2 GB of RAM for the default 20,000-transition replay buffer at 128 px, plus the base policy.

## What was wrong before the October 2026 fix

Review of the scripts as they were at commit `85c02f8` (week 5):

**`train_online_sac.py` / `sac_residual_agent.py`**

1. The residual was added to the base policy's *normalised* action and the sum clamped to [-1, 1]. With
   mean/std normalisation that also clamps the base action to +-1 standard deviation of the training actions:
   translation commands were limited to about [-0.27, 0.40] (x), [-0.29, 0.47] (y), [-0.54, 0.35] (z)
   instead of [-1, 1]. With a zero residual the agent was not the base policy. The preflight check ran the
   base without the clamp, so it never saw this.
2. The residual was at most `residual_scale` (0.01-0.02) in normalised units, i.e. about 0.004 in env units
   for translation: too small to change an outcome.
3. Every episode did `env.reset()`, `set_init_state(...)`, then `env.reset()` again. LIBERO's second reset
   re-randomises the scene, so training and evaluation never used the benchmark init states.
4. Evaluation drew 10 random scenes per seed, ran the base in bfloat16 under autocast, and was compared with
   a `week1_baseline_failures.csv` that is not in the repo; the base was never measured in the same harness.

**`train_online_ppo.py` / `ppo_residual_agent.py`** bypassed LeRobot's pre/post-processors: the state was 7
joint angles + 1 finger (un-normalised) instead of eef position / axis-angle / gripper, the image was flipped
vertically only (mirrored), and normalised actions were executed as env actions.

The week-5 SAC results (`week5/`) therefore do not measure residual SAC. The algorithmic parts (twin Q,
target networks, tanh log-probability, terminal only on success, automatic entropy tuning) were correct and
are kept in `sac_residual.py`. `sac_residual_agent.py` and `ppo_residual_agent.py` stay at the repo root
only because `reproduce_task.py` imports them.

## Status

The fixed code has passed CPU tests (`tests/test_sac_residual.py`, `tests/test_vla_core_rl.py`) and a short
real rollout; no training run has been made with it yet.
