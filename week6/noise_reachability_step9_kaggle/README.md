# Step 9: do hit noises carry over to other resistant states? (Kaggle 2xT4, 2026-11-03)

Protocol: `noise_reachability/README.md`, step 9, fixed before the run. Commit `f2a067e` (clean tree in all 76
provenance records), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, family `repeat`, tasks
0, 4, 6, 7. One session on host `76af95dd99ee`, one package set, first and last provenance records 12:25 and 16:17 UTC
(3.9 h). Source: `step9_results.zip`, sha256 `01949811af01cde99fa65a9c6978f4eae16f7cf5a908692ee08b9257b62fb9c4`.

**Result under the rule fixed in advance: not shown, no carry-over found.** The library's successes over the states
read: 10 of 142 against 23 of 400 for fresh ids, one-sided exact permutation p 0.408 (`run/step9_report.txt`). The
test had about 0.25-0.55 power for the hypothesised 2x gain, so this does not show that a hit noise is specific to its
state.

| State | Library | Random (ids 6000-6049) |
|---|---|---|
| 0/14 | 8/20 (19.1-63.9%) | 4/50 (2.2-19.2%) |
| 0/27 | 1/20 (0.1-24.9%) | 1/50 (0.1-10.6%) |
| 4/18 | 0/6 (0.0-45.9%) | 2/50 (0.5-13.7%) |
| 4/26 | 1/6 (0.4-64.1%) | 3/50 (1.3-16.5%) |
| 6/14 | 0/20 (0.0-16.8%) | 0/50 (0.0-7.1%) |
| 6/20 | 0/20 (0.0-16.8%) | 0/50 (0.0-7.1%) |
| 7/38 | 0/25 (0.0-13.7%) | 7/50 (5.8-26.7%) |
| 7/47 | 0/25 (0.0-13.7%) | 6/50 (4.5-24.3%) |
| Pooled | 10/142 (3.4-12.6%) | 23/400 (3.7-8.5%) |

Intervals are exact 95%. Library hits: task 0, ids 12, 15, 23, 35, 38, 44, 53, 57 on 0/14 (found by step 5a on 0/6, 0/7
and 0/8) and 1130 on 0/27 (step 8, 0/11); task 4, id 1276 on 4/26 (step 8, 4/11). No library id hit both states of its
task.

Gates. Harness, base of 30 on init states 0-9 x seeds 0-2 against step 0: task 0 6 (7), task 4 15 (14), task 6 16
(20), task 7 14 (18), all within 6. All eight held-out states passed the scene check, so all were read.

**Read with care (exploratory, not part of the rule; `step9_checks.txt`, `explore_step9.py`).** Episodes are treated as
independent in these p values.

- **The tasks point in opposite directions, and they cancel.** On 0/14 the library hit 8 of 20 against 4 of 50 (p
  0.0031; eight states were looked at, so about 0.025 after Bonferroni). On task 7 it hit 0 of 50 against 13 of 100 (p
  0.0039 the other way): the 25 noises that solved 7/33 did worse on 7/38 and 7/47 than fresh ones. Tasks 0, 4 and 6
  pooled are 10/92 against 10/300 (p 0.0072). The pre-registered test pools all four within task and does not pass.
- **So carry-over, if any, depends on the pair of states.** All eight 0/14 hits are step 5a ids, found on states 6,
  7 and 8; 0/27 took one of 20. Which states share good noises is not known in advance; the task alone does not decide it.
- **Fixed noise again reached states that sampling never solved.** 7/38 and 7/47 were at 0 of 32 `sampled` in step 8's
  screen; fresh `repeat` ids solved 13 of 100 there. This is not a step 9 test, but it matches step 8's finding.
- Task 6's held-out states stayed at 0 of 140 under both arms.

**Checks.** `analyze_step9.py --out` rerun locally reproduces the Kaggle report and its JSON byte for byte, from both
the inbox and this folder. `tools/validate_run.py` passes on the four `harness` folders with `--seeds 0-2
--init_states 0-9 --expect n_action_steps=10`. `check_step9_rows.py`, which shares no code with the analysis, recounts
the test folders: every file has its arm's ids exactly once (seed 1000 + id) on the right task and state, failures run
to 520 steps, every scene check passed, provenance is one clean commit with the protocol's settings and noise ids, and
200,000 random relabellings give p 0.4074 +- 0.0022 against the exact 0.408. The 28 logs (not committed) have no
traceback, error or `WORK REMAINS` line.

Episodes: 120 harness and 542 test, 662 in all (at most 662 planned). Run files in `run/` (logs left out).
