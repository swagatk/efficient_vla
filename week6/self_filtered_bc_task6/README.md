# Self-filtered behaviour cloning, task 6 (negative result)

Kaggle, 2 x T4, `self_filtered_bc/kaggle_self_filtered_bc.ipynb` at commit `c46108d` (clean tree in every
stage; see the `provenance.jsonl` files). Base policy `lerobot/smolvla_libero@a24f1f3804`, `n_action_steps=1`.
Method, hypothesis and control: `self_filtered_bc/README.md`. Follows the two residual-SAC pilots
(`week6/sac_residual_task6_pilot1/`, `week6/sac_residual_task6_pilot2/`).

Stated before the run: held-out success of at least 21/30 per training seed with a positive pooled paired
difference; failure if pooled success is at or below the base's 45/90.

## Harness check (stage `verify`)

The base policy through this eval path on init states 0-9, seed 0, gave the same outcome as pilot 1's base
evaluation in all 10 episodes (5/10 each; `harness_check/harness_check.txt`).

## Data (stage `collect`)

The frozen base on init states 10-49 x seeds 0-2: 36/120 successful episodes (`data/collection.csv`).
Training init states 10-44: 30/105 successes on 16 distinct init states. Validation init states 45-49:
6 successes.

## Training (stage `train`)

LoRA r16/alpha32 on the seven projections of the action expert (3,416,064 trainable parameters), 2000 steps,
batch 16, training seeds 0-2 (`run_logs/train_*.log`, `runs/*/train_history.json`).

- **Method (`success`):** the 30 successful episodes, 4194 frames, the same for every training seed.
- **Control (`all_matched`):** 30 episodes drawn without regard to outcome, a different draw per training
  seed: 6671, 6996 and 6814 frames, so about 1.6x the method's frames.
- The validation loss on the 6 held-out successful episodes was flat from step 500 to 2000 in all six runs
  (0.16-0.21) and did not separate method from control.

## Held-out evaluation (stage `eval`): no improvement

Init states 0-9 x eval seeds 0-2 per training seed, paired with the base episodes from the harness check
(`week6/sac_residual_task6_pilot1/eval_base/`, 15/30). Exact McNemar per run from
`runs/*/eval/comparison_base_vs_lora.txt`.

| | Training seed 0 | Training seed 1 | Training seed 2 | Pooled |
|---|---|---|---|---|
| Base | 15/30 | 15/30 | 15/30 | 45/90 |
| Method (`success`) | 19/30 (p = 0.39) | 13/30 (p = 0.79) | 14/30 (p = 1.0) | 46/90 |
| Control (`all_matched`) | 13/30 (p = 0.73) | 8/30 (p = 0.065) | 8/30 (p = 0.065) | 29/90 |

Mean ± std over training seeds: method 51.1 ± 10.7%, control 32.2 ± 9.6%, base 50%.

Pooled over training seeds (`pooled/`; episodes keyed by seed = 100 x training seed + eval seed):

| Comparison | Episodes succeeding only in A / only in B | Exact McNemar |
|---|---|---|
| Base vs method | 16 / 17 | p = 1.0 |
| Base vs control | 23 / 7 | p = 0.0052 |
| Control vs method | 10 / 27 | p = 0.0076 |

The pooled base column is the same 30 episodes counted three times, so the two comparisons against the base
overstate the evidence (their permutation tests too). Control vs method pairs independent LoRA episodes.

Successes per init state, out of 3 (base) and 9 (method, control):

| Init state | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 1 | 2 | 3 | 0 | 1 | 2 | 2 | 3 | 0 | 1 |
| Method | 3 | 6 | 6 | 3 | 4 | 8 | 8 | 5 | 0 | 3 |
| Control | 0 | 5 | 4 | 0 | 1 | 4 | 5 | 8 | 0 | 2 |

## Interpretation

- **Against the hypothesis:** no training seed reached 21/30, and pooled success (46/90) is one episode above
  the failure line. The method is indistinguishable from the base, so this is recorded as a negative result.
- **Filtering prevents harm; it does not add skill.** Cloning as many unfiltered episodes made the policy
  clearly worse (50% to 32%), and the method was better than the control (p = 0.0076). The control also had
  about 1.6x the frames, mostly from failed episodes, so it is not matched on data volume.
- **The seed spread is as large as the seed-0 gain.** Training seed 0 alone (19/30) looked like a +4 episode
  improvement; seeds 1 and 2 came in at or below the base.
- The successful episodes only contain behaviour the base already has. Init state 8, never solved by the
  base, stayed at 0/9.

Further variants of this setup (more steps, a frame-matched control) are not planned. A next idea needs a
source of behaviour the base does not already produce, with its own hypothesis.

## Files

- `harness_check/`: base vs pilot-1 base on init states 0-9, seed 0.
- `data/`: per-episode outcomes of the collection (episodes and adapter weights stay in the Kaggle output).
- `runs/{success,all_matched}_seed{0,1,2}/`: run config, provenance, training history, and per-eval-seed
  CSVs and comparisons. `runs/comparison_pooled_success_seeds1-2.txt` is the notebook's pooled report for the
  version that evaluated training seeds 1-2 only.
- `pooled/`: all three training seeds, built from the per-run CSVs and compared with
  `python -S baseline_performance/compare_per_state.py pooled/per_episode_<A>.csv pooled/per_episode_<B>.csv`.
- `run_logs/`: harness, collect, train and eval logs.
