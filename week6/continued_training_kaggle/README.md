# Continued training on LIBERO-10, Kaggle evaluation (negative result)

`lerobot/smolvla_libero` trained for 25,000 more steps on the LIBERO-10 demonstrations
(`continued_training/`, training record in `week6/continued_training_colab/`), evaluated with
`baseline_performance/kaggle_baseline_lerobot_eval.ipynb` (`CHECKPOINT = "continued"`) on 2 x T4, at commit
`643773a` with a clean tree (`provenance.jsonl`). Same settings as the base run in
`week6/smolvla_libero_lerobot_eval_kaggle/`: init states 0-9 x seeds 0-2 per task, `n_action_steps=1`, 520
steps, 256x256. 504 minutes (`evaluate_all.runner.log`).

Stated before the run (`continued_training/README.md`): at least 195/300; failure at 171/300 or below;
172-194 inconclusive.

## Result: no improvement

From `comparison_with_smolvla_libero_lerobot_eval.txt` and `per_episode.csv`:

| | Base | Continued |
|---|---|---|
| Total | 171/300 (57.0%) | 174/300 (58.0%) |
| Eval seeds 0 / 1 / 2 | 58 / 58 / 55 | 58 / 60 / 56 |
| Mean ± std over eval seeds | 57.0 ± 1.7% | 58.0 ± 2.0% |

- Paired by (task, seed, init state): 26 episodes succeed only with the base, 29 only with the continued
  model; exact McNemar p = 0.79.
- By the stated rule, 174 is in the inconclusive band. Added after the run: the exact 95% interval for the
  paired difference (Clopper-Pearson on the 29 of 55 discordant episodes) is -4.1 to +6.0 percentage
  points. It excludes the hypothesised +8 points, so this is recorded as a negative result for the
  hypothesis.
- Per task (out of 30): 0: 1 -> 0, 1: 23 -> 25, 2: 29 -> 30, 3: 27 -> 27, 4: 10 -> 11, 5: 26 -> 26,
  6: 14 -> 11, 7: 17 -> 20, 8: 2 -> 1, 9: 22 -> 23. The weak tasks (0, 4, 6, 8) stay weak.
- Which init states succeed did change: the summed per-state difference has permutation p = 0.034 (14 of
  100 init states solved in only one run, p = 0.091). Training altered the policy without a net gain.
- One training seed only; the three eval seeds measure eval variance, not training variance.

## Interpretation

7.9 more passes over the LIBERO-10 demonstrations lowered the training loss from 0.112 to 0.046 and did not
raise success. Together with self-filtered BC (`week6/self_filtered_bc_task6/`), more imitation of the same
data does not close the gap to the paper's 71%. The checkpoint's smaller training budget alone does not
explain that gap.

Later finding: the gap was the evaluation setting. With `n_action_steps=10` the base scores 216/300 (72.0%)
(`week6/noise_reachability_step0_kaggle/`). Both models here were evaluated at `n_action_steps=1`; the
continued model was not evaluated at 10.

## Files

- `per_episode.csv`, `consolidated_summary.txt`, `consolidated_report.csv`: 300 episodes.
- `comparison_with_smolvla_libero_lerobot_eval.txt`: paired comparison with the base run.
- `task_<t>_seed_<s>/`: `eval_info.json` and the gzipped `eval.log` of each run.
- `_timing/`: the notebook's timing probe; `evaluate_all.runner.log`, `provenance.jsonl`.
