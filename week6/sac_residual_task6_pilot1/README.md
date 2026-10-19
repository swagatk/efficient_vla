# Residual SAC pilot 1, task 6 (negative result)

Kaggle, 2 x T4, `rl_based_tuning/kaggle_sac_residual.ipynb` at commit `33f8740` (clean tree; see the
`provenance_*.jsonl` files). Base policy `lerobot/smolvla_libero@a24f1f3804`, `n_action_steps=1`.

## Harness check (stage `verify`): passed

The base policy through the RL evaluation path (`vla_core`) on task 6, init states 0-9 x seeds 0-2 scored
15/30, against 14/30 from lerobot-eval on the same init states
(`week6/smolvla_libero_lerobot_eval_kaggle/`): exact McNemar p = 1.00, no init state solved in only one run,
per-state permutation p = 0.92 (`harness_check.txt`, `eval_base/`).

## Training (stage `train`): the residual made the policy worse

Two seeds, 100 training episodes each on init states 10-49, all in one session, default settings of that
commit: `--residual_scale 0.05` for every action dimension, `--alpha 0.001` (fixed), no dense shaping, actor
updates from the first full minibatch, soft resets.

| | Training-episode success | First 20 | Last 20 |
|---|---|---|---|
| Seed 0 | 3/100 | 2 | 0 |
| Seed 1 | 12/100 | 3 | 1 |
| Base policy on init states 10-49 (LIBERO-CTRL, `baseline_performance/reference_libero_ctrl/`) | 16/40 | | |

Fisher exact test against 16/40: p = 7.8e-08 (seed 0), p = 3.8e-04 (seed 1). Training episodes include
exploration noise, but a drop from about 40% to 3-12% is far larger than noise around a near-zero residual
should cause. Both seeds succeeded on their first episode (residual still about zero) and rarely afterwards.
Losses stayed finite; the critic's mean Q rose while success stayed near zero (`train_seed*.log`, W&B
summary at the end). The held-out `eval` stage was not run.

## Suspected causes (as listed before the diagnostic below)

1. `--residual_scale 0.05` is the same for all dimensions: small for translation (action std about
   0.34-0.44) but about one standard deviation for rotation (std 0.04-0.08), so even the exploration noise is
   comparable to the base policy's rotation commands.
2. The actor follows an untrained critic from the start; with `alpha = 0.001` almost nothing restrains it.
3. Training uses soft resets, which LeRobot notes are not bit-identical to hard resets (only 3 base
   episodes were run in that environment, 1 success per seed).

Next: a diagnostic (base alone and base + exploration noise in the training environment), per-dimension
residual scales and a critic warm-up; see `rl_based_tuning/README.md`.

## Diagnostic (stage `diagnose`): neither the environment nor the noise explains the drop

Kaggle, same notebook at commit `85c1e5e` (the notebook clones `main`; diagnostic mode writes no
provenance file, so the commit is inferred from the clone, and the per-dimension caps printed in
`diag_noise_new.log` exist only from that commit on). Seed 0, the same 20 distinct training init states
(drawn from 10-49) and episode seeds in every condition, no learning. In the noise conditions the head is
untrained with a zero-initialised mean, so the only change to the base action is its exploration noise
(tanh of a Gaussian with log std -1, times the cap). Files in `diagnostic/`.

| Condition | Successes |
|---|---|
| LIBERO-CTRL reference on the same 20 init states | 7/20 |
| Base, soft reset (as in training) | 5/20 |
| Base, hard reset | 5/20 |
| Base + exploration noise, uniform cap 0.05 (pilot 1) | 7/20 |
| Base + exploration noise, per-dimension cap 0.15 std (new default) | 6/20 |

- The training environment is fine: base soft reset solves 4 of the reference's 7 init states plus one
  more (48); 5/20 against a 40% rate gives binomial p = 0.25.
- Soft and hard resets give identical outcomes and step counts on all 20 episodes.
- Exploration noise alone does not lower success at either size (base vs noise: exact McNemar p = 0.73
  for pilot 1's size, 1.00 for the new size; `summary.txt`).
- Pilot 1's training success is significantly below base + the same noise (7/20): Fisher p = 1.1e-04
  (seed 0, 3/100) and p = 0.018 (seed 1, 12/100). So the drop came from the learned residual, which fits
  cause 2 (actor following an untrained critic), not cause 1 or 3. The per-dimension cap is still the
  better-scaled setting, but this data does not show it was needed; the critic warm-up is the change that
  addresses the measured cause.
- Noise rarely changes the success rate but often changes which episodes succeed: only init state 40
  succeeds in all four conditions, and 30, 34 and 37 succeed with the base alone but fail under both noise
  sizes. Task-6 outcomes flip under small perturbations, so per-episode rewards are noisy and 20 paired
  episodes detect only large effects such as pilot 1's.

Next: pilot 2 with the warm-up and per-dimension cap; results in `week6/sac_residual_task6_pilot2/`.
