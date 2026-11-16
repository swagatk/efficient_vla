# Step 14: is round 2's loss on 0/8 real, and is it dilution? (Kaggle 2xT4, 2026-11-15/16)

Protocol: `noise_reachability/README.md`, step 14, fixed 2026-11-15 before the run. Notebook
`noise_reachability/kaggle_step14_dilution.ipynb` at commit `0f152b8` (pinned by `faa1d0d`),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps. Inputs: step 10's and step 12's Kaggle
outputs (adapters N0_s and R0_s, their recordings). Two sessions. Final source: `step14_results.zip` of session 2,
sha256 `bbd8b0700c8e53f5f6aa8f06a52707effce7cbb68f173c3b27db002fa1ed2fda`, committed as `run/` without logs (as step 12).
Report: `run/step14_report.txt` (+ `.json`), written by the notebook and reproduced exactly by
`python3 -S noise_reachability/analyze_step14.py --root week6/noise_reachability_step14_kaggle/run`. Adapter weights of
L0_0-L0_2: the Kaggle output of the step-14 notebook. Power: `step14_power.txt`.

**Result under the rule fixed in advance: neither test passes.** Round 2's loss on 0/8 was not replicated at this
budget, and longer training was not shown to help. Per training seed, 0/8, seeds 80-175 (96 paired episodes), exact
one-sided McNemar:

| Seed | N | R | L | Test A (N > R): only N / only R, p | Test B (L > R): only L / only R, p |
|---|---|---|---|---|---|
| 0 | 83 | 73 | 84 | 15 / 5, 0.0207 | 19 / 8, 0.0261 |
| 1 | 86 | 82 | 69 | 12 / 8, 0.252 | 7 / 20, 0.997 |
| 2 | 89 | 84 | 78 | 10 / 5, 0.151 | 9 / 15, 0.924 |

Each test needs p < 0.05 in at least 2 of 3 seeds; each passes in seed 0 only. N is above R in every seed, as in
step 12, but by less (pooled 258 against 239 of 288; step 12 had 41 against 32 of 48).

Reported, not part of the rule (two-sided):

- L against N on 0/8: 84 / 83 (p 1), 69 / 86 (p 0.00232), 78 / 89 (p 0.00739). Training round 2's set for 2,671 steps
  did not bring 0/8 back to round 1; in seeds 1 and 2 it is lower than R as well, so 0/8 varies more across L's seeds.
- L against R on the searched states (0/2, 0/5, 0/6, 0/11, seeds 80-95, 64 per seed): 42 / 33 (p 0.049), 41 / 35
  (p 0.286), 41 / 29 (p 0.0169); pooled 124 against 97 of 192. Most of it is 0/5 (L 11, 12, 10 of 16 against R 2, 6,
  3). No test was fixed for this, so it is a lead, not a finding.
- R on task 0's searched states with fresh seeds: 97 of 192 pooled, against 106 of 192 in step 12.

Pooled counts carry no p-value: the seeds share their evaluation episodes.

**Checks.**

- Harness gate passed: 8 of 8 committed episodes reproduced in success and steps (N0 4/4, R0 4/4; at least 7 needed).
- `tools/validate_run.py --expect n_action_steps=10 --expect max_steps=520` passes on all 15 evaluation folders: every
  (seed, state) once per policy, 1,248 evaluation episodes, one clean commit in all 161 provenance records. Its
  `eval.log` check does not apply (the notebook writes stdout logs to `logs/`); none of the 83 logs has a traceback.
- Training: L0_0-L0_2 trained on exactly R0_s's 50 episodes (same sha256, checked by the analysis), 2,671 steps, LoRA
  r16/alpha32 on all seven action-expert projections, batch 16, lr 1e-4, warmup 100, `--selection success`, no
  validation states; the settings of the three runs differ only in the seed. Peak GPU memory 4.2 GB, peak RSS 6.0 GB.
- The report counts were recounted from the CSVs by a separate script (same pairs and p-values).

## Session 1 (2026-11-15 23:43 to 2026-11-16 08:35 UTC, host `0038e72418f5`)

`step14_results.zip` sha256 `80666b074b6dc1013a1e15d64cb516b734ce44f5a134a5ce19ea4ab7c1c3191f`. Harness, the N and R
evaluation, L0_0 and L0_1 trained and evaluated: 1,096 of 1,256 episodes. L0_2 did not start, since the session could
not finish it; the analysis refused the partial run. `validate_run.py` passed on every folder.

## Session 2 (2026-11-16 11:58 to 16:09 UTC, host `f3d2cd243ba1`)

Resumed from session 1's output: all of session 1's CSV files are byte-identical in session 2's zip, which adds only
L2's 10 evaluation files. L0_2 trained (4.46 s per step) and was evaluated (160 episodes); the notebook then ran the
analysis.
