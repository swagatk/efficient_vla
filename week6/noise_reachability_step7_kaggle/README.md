# Step 7: deep noise on tasks 4, 6, 8 (Kaggle 2xT4, 2026-11-01/02)

Protocol: `noise_reachability/README.md`, step 7, fixed before the run. Commit `710a3aa` (clean tree in all 123
provenance records), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, states 4/2, 6/8, 8/4, noise ids 0-69
(episode seeds 1000-1069). Source: `step7_results.zip`, sha256
`1273b06d4040a236a041797be737ebf7975bd3b616311cf3ca48e4290b805075`.

**Result under the rule fixed in advance: existence only.** `repeat` hit 16 of 210 against `sampled` 10 of 210 (one-sided
Fisher p 0.156, 0.05 required), so task 0's rate gap was not shown here, though it could not have been ruled out (below).
`repeat` hits fell on 3 of 3 states, but the spread only counts with the rate. Without task 8: 6 of 140 against 6 of 140,
p 0.615.

Harness (`run/step7_report.txt`): 15, 16, 13 of 30 against 14, 20, 15, all within 6. Controls of 35, `sampled` first: 4/5,
8, 14, 10; 6/0, 14, 13, 5; 8/8, 15, 14, 12; passed.

| State | sampled /70 | full /70 | repeat /70 | Jittered successes per read hit, of 4 | Class |
|---|---|---|---|---|---|
| 4/2 | 6 | 10 | 3 | full: 0, 1, 2; repeat: 1, 1, 2 | robust |
| 6/8 | 0 | 0 | 3 | repeat: 1, 2, 2 | robust, fixed-only |
| 8/4 | 4 | 7 | 10 | full: 2, 0, 1; repeat: 1, 0, 0 | robust |
| Total /210 | 10 | 17 | 16 | 16/60 | |

95% intervals: `repeat` 4.4-12.1%, `sampled` 2.3-8.6%. 15 distinct `repeat` noises hit (noise 27 hit 4/2 and 8/4).

**Read with care (exploratory, not part of the rule; `step7_checks.txt`).**

- **"Systematic" means rare, and now on `sampled` too.** Two of the three states are solved by ordinary sampling: with base
  seeds 0-31 they stand at 6/102 (4/2), 0/102 (6/8) and 5/102 (8/4). Task 0's 5 states stayed at 0 of 350. A label from 32
  episodes could not tell these apart: at 5%, 0 of 32 happens about one time in five.
- **The power table did not cover this.** It assumed `sampled` at 1-2%; step 6's 1 of 48 was the only estimate. The
  observed 4.8% is above that range, so this test had less power than stated, and failing it says little either way.
- **6/8 is the only state that matches task 0's pattern**: 0/102 sampled, 0/70 `full`, 3/70 `repeat` (p 0.066 against the
  /102) and 5 of 12 jittered episodes (p 5.4e-06). The jittered episodes are new, so a hit's neighbours work on a state where
  sampling never did. But this is one state of three picked after seeing the data, and 12 episodes around 3 noises.
- **`repeat` is not better in general.** It trails `sampled` on 4/2 (3 against 6) and on the pooled controls (27 against 37 of
  105, `full` 41). Its surplus is all on 8/4 (10 against 4), where the stove varies and where jittered hits mostly failed
  (1 of 12).
- **Step 6's note on 4/2 needs a correction:** arm B's 6 of 16 is not on "a state the base never solved"; the base solves it
  rarely. B's rate still stands above the /102 (p 0.0014), one state picked after seeing the data.

**Checks.** `analyze_step7.py` rerun locally reproduces the Kaggle report byte for byte. `tools/validate_run.py` passes on
the three `harness` folders with `--seeds 0-2 --init_states 0-9 --expect n_action_steps=10`. It does not read the search
files, so `check_step7_rows.py`, which shares no code with the analysis, recounts them: every file has the expected ids,
task and state, and all counts agree. It finds no logs in those folders, so the 42 logs in `run/logs/` were scanned by hand:
no traceback, segfault, out-of-memory or `Error:` line. The 15 confirmation files hold jitter seeds 5000-5003.

Noise id 0 outcomes match the smoke run's on 4/2 and 6/8. `full` on 8/4 succeeded in 394 steps here against 373 in the smoke
run, as expected if task 8 is not deterministic (FAIL-010).

Run time: 6.0 h between the first and last provenance records, against about 6.5 h estimated.
