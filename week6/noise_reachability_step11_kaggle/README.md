# Step 11: does fixed noise beat sampling on step 9's eight held-out states? (Kaggle 2xT4, 2026-11-08)

Protocol: `noise_reachability/README.md`, step 11, fixed 2026-11-04 before the run. Notebook
`noise_reachability/kaggle_step11_replication.ipynb` at commit `8980b37` (clean tree in all 60 provenance records),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, hard resets, family `sampled`, noise ids
6000-6049 on step 9's eight held-out states. One session on host `1d8039dc09c0`, first and last provenance records 13:18
and 16:14 UTC (2.9 h). The `repeat` arm is step 9's committed run (`f2a067e`); its files matched the pinned sha256 and
the package versions equal step 9's. Source: `step11_results.zip`, sha256
`369343f1243371bdb0c69c063fbdce9234e74c525ab02912585dac7102a31d8d`. Report: `run/step11_report.txt` (+ `.json`), reproduced
exactly by `noise_reachability/analyze_step11.py --root week6/noise_reachability_step11_kaggle/run`.

**Result under the rule fixed in advance: replicated.** Fixed noise beats sampling on eight new resistant states chosen
in advance, and also without task 7.

| Test (one-sided Fisher, Holm at 0.05) | `repeat` (step 9) | `sampled` (step 11) | p | |
|---|---|---|---|---|
| A: all eight states | 23/400 (3.7-8.5%) | 3/400 (0.2-2.2%) | 3.4e-5 | passes |
| B: tasks 0, 4, 6 (six states) | 10/300 (1.6-6.0%) | 2/300 (0.1-2.4%) | 0.018 | passes |

Paired McNemar cross-check (same noise id and scene per pair): 4.4e-5 and 0.019. Intervals are exact 95%.

| State | `repeat` | `sampled` |
|---|---|---|
| 0/14 | 4/50 | 2/50 |
| 0/27 | 1/50 | 0/50 |
| 4/18 | 2/50 | 0/50 |
| 4/26 | 3/50 | 0/50 |
| 6/14 | 0/50 | 0/50 |
| 6/20 | 0/50 | 0/50 |
| 7/38 | 7/50 | 1/50 |
| 7/47 | 6/50 | 0/50 |

Gates. Harness, base of 30 on init states 0-9 x seeds 0-2 against step 0: task 0 6 (7), task 4 15 (14), task 6 16
(20), task 7 14 (18), all within 6. All eight states passed the scene check. No traceback in any log.

**Read with care.** Test B rests on 10 against 2 successes, and task 6 contributes nothing to either arm, so it holds
on four states of tasks 0 and 4. The two arms ran in different sessions (same hardware type, same package versions,
commits `f2a067e` and `8980b37`, between which `deep_noise_search.py` only gained `--record_dir`, unused here). Step 8's
test 2 had most of its effect on 7/33; this run has no 7/33, and the effect holds on states selected before any of
these episodes.

Files: `run/` (episodes, provenance, report; logs left out), `step11_power.txt` (the design's power).
