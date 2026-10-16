# Week 5: residual SAC (invalid result)

The CSVs in this folder are the week-5 residual SAC runs on tasks 0 and 3, produced by
`week5_run_sac_3seed_task.sh` with `train_online_sac.py` as of commit `85c02f8`.

They do not measure residual SAC. A review in October 2026 found that the training script clamped the base
policy's normalised actions to +-1 standard deviation (so even a zero residual changed the base policy),
allowed a residual of only about 0.004 in env units, and discarded the benchmark init states with a second
`env.reset()`; the base policy was never evaluated in the same harness. Details and the fixed code:
[`rl_based_tuning/README.md`](../rl_based_tuning/README.md).

Kept as the record of what was run. Do not cite these numbers as evidence for or against residual RL.
