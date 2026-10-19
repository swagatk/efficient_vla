# Residual SAC pilot 2, task 6 (negative result)

Kaggle, 2 x T4, `rl_based_tuning/kaggle_sac_residual.ipynb` at commit `85c1e5e` (clean tree; see the
`provenance_*.jsonl` files). Base policy `lerobot/smolvla_libero@a24f1f3804`, `n_action_steps=1`. Follows
pilot 1 and its diagnostic (`week6/sac_residual_task6_pilot1/`), which showed that neither the training
environment nor the exploration noise lowers success, so pilot 1's drop came from the learned residual.

## Changes from pilot 1

- Residual cap per action dimension = 0.15 x that dimension's action std (`--residual_scale_std 0.15`):
  0.050, 0.057, 0.067 translation, 0.006, 0.009, 0.012 rotation, 0.150 gripper (pilot 1: 0.05 everywhere).
- Critic warm-up: the first 5 episodes run the base policy alone and update only the critic
  (`--warmup_episodes 5`).

Unchanged: `--alpha 0.001` (fixed), sparse success reward, soft resets, training init states 10-49, two
seeds, 100 training episodes each. Both seeds finished all 100 episodes in one session without errors.

Expected (stated before the run): training success after the warm-up stays near base + exploration noise
(13/40 in the diagnostic, about 30%) instead of collapsing as in pilot 1.

## Training: no collapse, but no improvement

| | Training success (all 100) | Warm-up (eps 1-5) | Eps 6-100 | Per 20 episodes |
|---|---|---|---|---|
| Pilot 1 seed 0 | 3/100 | | | |
| Pilot 1 seed 1 | 12/100 | | | |
| Pilot 2 seed 0 | 22/100 | 1/5 | 21/95 | 6, 3, 6, 3, 4 |
| Pilot 2 seed 1 | 29/100 | 3/5 | 26/95 | 12, 7, 5, 4, 1 |

- Against base + noise (13/40), episodes 6-100: Fisher p = 0.28 (seed 0), 0.54 (seed 1). The warm-up and
  per-dimension cap removed pilot 1's collapse.
- Seed 0 has no trend (Spearman over episodes 6-100, p = 0.49). Seed 1 declines (p = 1.4e-04); its last 50
  episodes score 7/50 against 13/40 (p = 0.044). The decline is not an init-state effect: on the init states
  the LIBERO-CTRL reference solves (`baseline_performance/reference_libero_ctrl/`), seed 1 succeeded 10/18
  in episodes 6-50 and 5/21 in episodes 51-100 (seed 0: 4/15 and 8/25).
- The mean |delta| per episode (fraction of the cap, exploration noise included) rose from about 0.35 to
  about 0.5; tanh-squashed noise alone at the initial log std (-1) gives about 0.27, so the actor's mean moved
  away from zero. It was the same size in successful and failed episodes (0.41 vs 0.41, Mann-Whitney
  p = 0.87).

## Held-out evaluation (stage `eval`): the residual makes the policy worse

Deterministic residual (tanh of the mean, no noise), task 6, init states 0-9 x eval seeds 0-2,
replanning every step (`n_action_steps=1`, the FAIL-005 requirement), paired with the base episodes from the harness check
(`week6/sac_residual_task6_pilot1/eval_base/`, 15/30).

| | Base | Residual | Exact McNemar |
|---|---|---|---|
| Training seed 0 | 15/30 | 3/30 | p = 0.0042 |
| Training seed 1 | 15/30 | 6/30 | p = 0.049 |
| Both seeds | 30/60 | 9/60 | p = 0.0003 |

Across both seeds, 27 paired episodes succeeded only with the base and 6 only with the residual
(`comparison_pooled.txt`, `seed_*/eval/`, `eval_train*_s*.log`).

## Interpretation

The changes fixed the instability, not the learning problem. With one sparse reward per episode, about 95
learning episodes per seed, image observations, and task-6 outcomes that flip under small perturbations
(pilot 1's diagnostic), the critic has too little signal, and the actor's mean drifts to a correction that
is harmful once the exploration noise is removed. The drift explanation is inferred from the logged |delta|;
the checkpoints were not inspected (they stay in the Kaggle output, not in this folder).

Further 100-episode sessions of this setup are not planned: seed 1 got worse with training, and each
session costs about 10 GPU hours. Next candidate: supervised learning on the base policy's own successful
noisy rollouts (success-filtered imitation), to be specified with a hypothesis before any code changes.
