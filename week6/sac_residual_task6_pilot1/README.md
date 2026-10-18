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

## Suspected causes (not yet separated)

1. `--residual_scale 0.05` is the same for all dimensions: small for translation (action std about
   0.34-0.44) but about one standard deviation for rotation (std 0.04-0.08), so even the exploration noise is
   comparable to the base policy's rotation commands.
2. The actor follows an untrained critic from the start; with `alpha = 0.001` almost nothing restrains it.
3. Training uses soft resets, which LeRobot notes are not bit-identical to hard resets (only 3 base
   episodes were run in that environment, 1 success per seed).

Next: a diagnostic (base alone and base + exploration noise in the training environment), per-dimension
residual scales and a critic warm-up; see `rl_based_tuning/README.md`.
