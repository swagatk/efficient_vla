---
name: residual-rl-debugging
description: Checklist for investigating why residual RL (SAC on top of frozen SmolVLA, rl_based_tuning/) is not improving success on LIBERO-10. Use when debugging residual RL training, rewards, rollouts or the residual head.
---

# Residual RL debugging

The week-5 PPO and SAC runs are invalid (base actions clamped, init states discarded; see `rl_based_tuning/README.md`
and `week5/README.md`). The current code is `rl_based_tuning/train_online_sac.py` with the agent in
`rl_based_tuning/sac_residual.py`, run via `rl_based_tuning/run_sac_residual.sh`. `rl_based_tuning/train_online_ppo.py`
is retired, and the root `sac_residual_agent.py` and `ppo_residual_agent.py` are pre-fix code kept only for
`reproduce_task.py`.

Inspect these areas in order before widening scope:

1. `rl_based_tuning/train_online_sac.py`: reward shaping, rollout and replay storage, update inputs, and success logging.
2. `rl_based_tuning/sac_residual.py`: what observations the residual head actually consumes and whether the update path
   matches rollout-time inputs.
3. `visual_prompt_wrapper.py`: whether RL training still uses week-2 or week-3 prompting even though the wrapper path
   was deprioritized.
4. Existing W&B logs under `wandb/` and checkpoints under `checkpoints_task_*/` for evidence of reward trends, entropy
   collapse, or resume behavior.

Verify these assumptions explicitly rather than assuming they are correct:

- `done` means true task success rather than time-limit termination.
- Updates replay the same observation modalities used during rollout.
- The reward scale is compatible with the BC anchor and the RL losses.
- The residual head receives the information needed to correct base-policy failures.
- Visual prompting is intentional in RL training rather than an inherited week-2 default.
- Residual evaluation keeps `--eval_replan_each_step 1` (FAIL-005).
