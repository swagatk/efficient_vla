# Hybrid diffusion: residual head on frozen SmolVLA

A small diffusion head, conditioned on frozen SmolVLA features and the base action chunk, predicts a
residual that is added to the base policy's action (`residual` policy mode, scaled by `alpha`).

- `residual_cache.py`: cache frozen features and base chunks for one task's demos, then train the head.
- `train_hybrid_diffusion.py --eval_only`: LIBERO rollouts for `base` and `residual` policy modes.
- `run_task3_residual.sh`: cache, train and evaluate one task for one seed (`TASK_ID`, `SEED`).
- `run_task3_residual_3seed.sh`: the 3-seed comparison below, followed by `aggregate_seeds.py`.
- `run_base_noise_seeds.sh`: base policy only, several noise seeds on one task, followed by
  `failure_overlap.py` (see "Are base failures random or systematic?").

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

`PARALLEL=1` trains the heads first and then runs the six evaluations as concurrent processes (about
2.7 GB of GPU memory each). `colab_task3_3seed.ipynb` does this on a Colab A100, with the environment
pinned by `requirements-colab.txt` and results on Google Drive. All six evaluations of one comparison
must come from the same machine.

Results land in `outputs/task3_residual/` (`eval_base_s<seed>/`, `eval_residual_<run>_s<seed>_alpha1.0/`,
`seed_summary_s1-2-3_alpha1.0.json`).

## Seeds

- The episode `seed` in `eval_results.json` is the LIBERO init-state index.
- `--eval_noise_seed` seeds the policy's sampling noise independently of the init state (default 0 is
  the original seeding). Each noise seed needs its own `--eval_output_dir`; mixing them is refused.
- Rollouts are not bit-reproducible: the same init state and noise seed gave 245 and then 229 steps on
  two consecutive runs of task 3 (base, init state 0) while another job shared the GPU. Outcomes on a
  single episode can therefore differ between runs with identical seeds.

## Result of the 3-seed replication (task 3)

Not supported. Base 133/150, residual 130/150, pooled exact McNemar p = 0.7428
(`week6/task3_residual_3seed_colab/seed_summary_s1-2-3_alpha1.0.json`, all episodes on a Colab A100).

## Are base failures random or systematic? (task 0)

Hypothesis: the base policy's failures on LIBERO-10 task 0 mostly move with the sampling noise instead
of repeating on the same init states. On task 0 the base solved 16/50 locally with noise seed 0
(`outputs/task0_residual/eval_base/eval_summary.csv`), so there is room to see either outcome.

Run: base policy, noise seeds 1, 2, 3, the same 50 init states each (150 episodes), no head involved.

```bash
bash hybrid_diffusion/run_base_noise_seeds.sh
```

`colab_task0_base_noise_seeds.ipynb` runs it on a Colab A100 with `PARALLEL=1`.

`failure_overlap.py` reports the success of a best-of-3 oracle (an init state counts as solved if any
seed solved it) and a randomness index: (oracle - single seed) / (oracle under independent failures -
single seed). It is about 1 when failures move with the seed and about 0 when the same init states
fail every time. Decision rule, fixed in advance:

- index >= 0.6: mostly random. Next step is selecting among or steering the base policy's own samples.
- index <= 0.3: mostly systematic. Next step needs new data from the base policy's own rollouts.
- in between: mixed; report both and do not pick a direction from this run alone.

Rollouts are not bit-reproducible (see "Seeds"), so part of any seed-to-seed variation is run-to-run
nondeterminism. The index measures how much outcomes vary on a fixed init state, whatever the source.

Results land in `outputs/task0_residual/` (`eval_base_s<seed>/`, `failure_overlap_s1-2-3.json`).

### Result (task 0, Colab A100)

Mixed. Base 13/50, 9/50 and 8/50 for seeds 1, 2, 3 (20.0 % pooled, 30/150); best-of-3 oracle 36.0 %
against 49.0 % if failures were independent; randomness index 0.55; 5 init states solved under all
three seeds and 32 under none; permutation test for repeated failures p = 0.0001
(`week6/task0_base_noise_seeds_colab/failure_overlap_s1-2-3.json`). Under the rule above this run does
not pick a direction.

The Colab base rate matches the reference baseline for task 0 (16.7 +- 5.8 %,
`baseline_performance/results_300926/consolidated_summary.txt`). The local seed-0 run was higher, 16/50
(`week6/task0_base_local_seed0/eval_results.json`, copied from `outputs/task0_residual/eval_base/`);
16/50 against 30/150 gives Fisher exact p = 0.085.

Unplanned comparison, not part of the decision rule: the local seed-0 run solved 8 of the 32 init states
that Colab never solved in 96 episodes (states 1, 2, 4, 15, 20, 33, 34, 37; 8/32 against 0/96, Fisher
exact p = 7.4e-6). With the same code, init states and pinned packages, which states succeed depends on
the machine, so the 32 states are not shown to be unsolvable. The local side is a single 50-episode run.
Planned check: base on the local GPU with noise seed 1, compared per init state with Colab seed 1.

## Task 0

The same recipe on task 0 showed no difference: 16/50 base and 15/50 residual
(`outputs/task0_residual/eval_base/eval_summary.csv`,
`outputs/task0_residual/eval_residual_h512_l6_n0.5_e300_alpha1.0/eval_summary.csv`), single seed.
