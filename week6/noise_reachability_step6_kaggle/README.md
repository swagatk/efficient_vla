# Step 6: labels and arm B's gain on tasks 4, 6, 7, 8 (Kaggle 2xT4, 2026-10-31/11-01)

Protocol: `noise_reachability/README.md`, step 6, fixed before the run. Commit `24437d2` (clean tree in all 128
provenance records), `lerobot/smolvla_libero` at `a24f1f3804` and step 4's arm B (SHA-256 `b75872e158799eb4`),
`n_action_steps=10`, init states 0-9 x seeds 16-31. Source: `step6_results.zip`, sha256
`7dbc51dfadc9cdd6a3d5ae2c0ae0778efba95857b6442ba401e222fb96a19453`.

**Result under the rule fixed in advance: the labels hold, and the gain comparison is not informative.** The base solved
1 of 48 episodes on the 3 systematic states (allowed 4; at most 1 on a state; 95% interval 0.1-11.1%). B's mean gain on
the 13 low reachable states was -0.067, against the 0.10 required, so by the fixed reading nothing is read about
unevenness.

Harness (`run/step6_report.txt`), all within 18 of 160: task 4, 71 against 79; task 6, 95 against 97; task 7, 90 against
90; task 8, 70 against 70. State Spearman between seeds 0-15 and 16-31: 0.92, 0.83, 0.85, 0.81.

| Task | Base /160 | B /160 | Net | McNemar p |
|---|---|---|---|---|
| 4 | 71 | 85 | +14 | 0.0759 |
| 6 | 95 | 86 | -9 | 0.281 |
| 7 | 90 | 111 | +21 | 0.00191 |
| 8 | 70 | 59 | -11 | 0.117 |
| All four | 326 | 341 | +15 | |
| 0 (step 4) | 37 | 64 | +27 | 1.12e-07 |

Rule values: `D` -0.168 (required 0.10), exact stratified p 0.9444 over 144 assignments, episode-noise 95% interval
-0.289 to -0.044. Per task: 4, -0.425; 6, -0.042; 8, -0.038.

Secondary: with task 0, `D` -0.012 (p 0.5149 over 8,064 assignments); log-odds `D` -0.804 (p 0.8611); Spearman of the rate
on seeds 0-15 against the gain, 50 states, -0.116.

**Read with care (exploratory, not part of the rule).**

- `D` is below zero because of one state: task 4 state 2, a systematic state, went from 0 to 6 of 16 under B (6 gained,
  0 lost, sign test p 0.031 before correction). The base is 0 of 32 on it across seeds 0-31. So B can reach a state the
  base never solved, but this is one state picked after seeing the data; states 6/8 and 8/4 did not move (0 to 0, 1 to 0).
- B moves success between states more than it raises it. On 40 states, 7 have sign-test p < .05 before correction, in
  both directions: 8 to 16 on 4/9, 3 to 11 on 7/4, 8 to 16 on 7/6, but 14 to 5 on 6/4, 9 to 2 on 4/7, 11 to 4 on 8/1. None
  passes Bonferroni (.05/40). With one B checkpoint, this cannot be split into a checkpoint effect and a training effect.
- Task 0's pattern (large gain on reachable states, little on systematic ones) was not seen here, but this is not a
  failed replication under the rule: the comparison stopped at the informative check.

**Checks.** `analyze_step6.py` rerun locally reproduces the Kaggle report byte for byte; the JSON differs only in file
paths, and all input hashes are equal. `tools/validate_run.py` passes on `eval/base` and `eval/B` with `--tasks "4 6 7 8"
--seeds 16-31 --init_states 0-9 --expect n_action_steps=10`. It finds no logs there, so the 32 logs in `run/logs/` were
scanned by hand: no traceback, segfault, out-of-memory or `Error:` line. Merged files equal their parts, with 640 distinct
episodes per policy. `n_action_steps` is confirmed from provenance, which also shows each arm's own `policy_path`. Two
tasks match their seeds 0-15 count exactly, so this was checked: 44-50 of each task's 160 outcomes differ from the episode
16 seeds lower at the same state, so these are new episodes. `D` and the gains were recomputed separately in `check_step6_rows.py`; output in
`step6_checks.txt`.

Run time: base from 22:16 to about 01:10 and B to 04:01 on the session's clock, about 5.9 h in all.
