# Hybrid diffusion: residual head on frozen SmolVLA

A small diffusion head, conditioned on frozen SmolVLA features and the base action chunk, predicts a
residual that is added to the base policy's action (`residual` policy mode, scaled by `alpha`).

- `residual_cache.py`: cache frozen features and base chunks for one task's demos, then train the head.
- `train_hybrid_diffusion.py --eval_only`: LIBERO rollouts for `base` and `residual` policy modes.
- `run_task3_residual.sh`: cache, train and evaluate one task for one seed (`TASK_ID`, `SEED`).
- `run_task3_residual_3seed.sh`: the 3-seed comparison below, followed by `aggregate_seeds.py`.

## Hypothesis under test (3-seed replication, task 3)

On LIBERO-10 task 3, the residual head (`h512_l6_n0.5_e300`, `alpha=1.0`) raises success over base
SmolVLA by about 10 points. The exploratory seed-0 run gave 43/50 base and 48/50 residual
(`outputs/task3_residual/eval_base/eval_summary.csv`,
`outputs/task3_residual/eval_residual_h512_l6_n0.5_e300_alpha1.0/eval_summary.csv`), which is not
significant on its own (1 base-only vs 6 residual-only episodes, exact McNemar p = 0.125).

Confirmation run: seeds 1, 2, 3, each with a head trained on that seed and both policies evaluated on
the same 50 init states with that sampling-noise seed (150 episodes per policy). Hyperparameters and
alpha are fixed in advance. The hypothesis is supported if residual is ahead on pooled success and the
pooled exact McNemar test gives p < 0.05; otherwise it is not.

```bash
bash hybrid_diffusion/run_task3_residual_3seed.sh
```

Results land in `outputs/task3_residual/` (`eval_base_s<seed>/`, `eval_residual_<run>_s<seed>_alpha1.0/`,
`seed_summary_s1-2-3_alpha1.0.json`).

## Seeds

- The episode `seed` in `eval_results.json` is the LIBERO init-state index.
- `--eval_noise_seed` seeds the policy's sampling noise independently of the init state. With the
  default 0 an evaluation is deterministic and re-running it reproduces the same episodes, so repeats
  need a different noise seed and a separate `--eval_output_dir`.

## Task 0

The same recipe on task 0 showed no difference: 16/50 base and 15/50 residual
(`outputs/task0_residual/eval_base/eval_summary.csv`,
`outputs/task0_residual/eval_residual_h512_l6_n0.5_e300_alpha1.0/eval_summary.csv`), single seed.
