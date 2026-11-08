# Step 10: can constant-noise successes be learned into the weights? (Kaggle 2xT4, 2026-11-04 to 11-08)

Protocol: `noise_reachability/README.md`, step 10, fixed 2026-11-03. Notebook
`noise_reachability/kaggle_step10_learn_from_noise.ipynb` at commit `f4369f3`, `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, tasks 0, 6, 7. Four Kaggle sessions, each resumed from the previous version's output; 366 provenance
records, one commit, clean trees. Report: `run/step10_report.txt` (+ `.json`), reproduced exactly by re-running
`noise_reachability/analyze_step10.py --root week6/noise_reachability_step10_kaggle/run`.

**Result under the rule fixed in advance: both tests pass.** A LoRA trained on replayed constant-noise successes (arm N)
solves the resistant states it was trained on with ordinary sampled noise, and the same update on the base's own
sampled successes (arm A) does not.

| Test (paired, exact one-sided McNemar) | First | Second | p |
|---|---|---|---|
| 1: N0 against base, trained states (192 episodes) | 77/192, 40.1% (33.1-47.4%) | 1/192, 0.5% (0.0-2.9%) | 1.3e-23 |
| 1: N1 against base | 71/192, 37.0% (30.1-44.2%) | 1/192 | 1.5e-20 |
| 1: N2 against base | 83/192, 43.2% (36.1-50.6%) | 1/192 | 4.4e-24 |
| 2: N against A, task 0 states 2, 5, 6, 7, 8, three seeds pooled (240) | 99/240, 41.2% (35.0-47.8%) | 6/240, 2.5% (0.9-5.4%) | 2.6e-25 |

Test 1 passes in 3 of 3 seeds (2 needed); the hypothesis asked for at least 10% where the base is near 0. Intervals are
exact 95%. Trained states: task 0 states 2, 5, 6, 7, 8, 11 (16 seeds each), task 6 states 8, 12 and task 7 state 33
(32 each).

**Reported, not part of the rule** (per state and policy in `run/step10_report.txt`; training episodes per state in
`session4_checks.txt`):

| Task 0 state | Role | N training episodes | Base | N0, N1, N2 | A0, A1, A2 |
|---|---|---|---|---|---|
| 0 | easy | 0 | 0/16 | 16, 16, 16 | 0, 0, 0 |
| 2 | trained | 1 | 0/16 | 6, 1, 7 | 0, 0, 0 |
| 5 | trained | 0 | 0/16 | 2, 1, 4 | 0, 0, 0 |
| 6 | trained | 9 | 0/16 | 6, 4, 3 | 1, 2, 3 |
| 7 | trained | 2 | 0/16 | 4, 11, 7 | 0, 0, 0 |
| 8 | trained | 18 | 0/16 | 13, 14, 16 | 0, 0, 0 |
| 11 | trained | 7 | 0/16 | 3, 3, 6 | not read |
| 4 | easy | 0 | 3/16 | 0, 0, 0 | 5, 0, 2 |

- *Test 1 rests mostly on two states.* Of each N seed's 71-83 trained-state successes, 7/33 (33 training episodes)
  gives 29, 30, 26 of 32 and 0/8 (18) gives 13, 14, 16 of 16. Task 6's trained states (8: 23 episodes, 12: 10) reach
  3-9 of 32. Arm A trained on 44 successes from states 10-44, none of them a test-2 state.
- *The largest gain is on a state N never trained on.* 0/0 goes from 0 of 16 to 16 of 16 for every N seed (48 against 0),
  while A stays at 0 like the base; 0/0 is "easy" only by step 1's seeds 0-15. 0/2 and 0/7, with 1 and 2 training
  episodes, and 0/5, with none, also gain. So N changes task 0's behaviour broadly, not state by state; A does not.
- *Held out* (step 8's untested eligible states): task 0 (0/14, 0/27, 32 episodes) N 17, 14, 14 against base 1 (p 7e-5
  to 5e-4). Tasks 6 and 7 almost none: task 6 0 of 64 for every seed; task 7 N0 6 of 64, N1 and N2 0, base 0.
- *Easy states:* mostly gains (task 0, 80 episodes: N 61, 60, 58 against base 37; task 7 state 4: 32 of 32 for every
  seed against 10). **One harm the one-sided tests cannot flag:** on 0/4 the base wins 3 of 16 (seeds 19, 23, 31) and
  every N seed 0 of 16. With 3 discordant pairs per seed the exact two-sided p is 0.25, so it is consistent across seeds
  but not shown at 16 episodes.
- *Arm A against base* on task 0's systematic states (80): 1, 2, 3 against 0 (p 0.5, 0.25, 0.125).

**Read with care.**

- *Same scenes.* A trained state's evaluation differs from N's training episodes only in the policy's sampling noise:
  the environment seed changes nothing once the init state is set (`deep_noise_search.py`, `check_step5a.py`). Test 1
  shows that these configurations became solvable with sampled noise, not that new scenes did.
- *Test 2 is not a same-scene comparison.* N trained on task-0 states 2, 6, 7, 8, 11 themselves (hard resets, as the
  replays ran); A on states 10-44 (soft resets, as step 4 collected). A has no successes on the systematic states, which
  is the premise. "Beyond the policy's own successes" therefore holds in practice, but test 2 does not separate
  constant-noise data from data on the same scenes. The 240 pairs reuse the same 80 episodes once per seed.
- *Generalisation* appears on task 0 only, and with the broad shift above it may be the same effect, not transfer
  between similar scenes.
- One round of filtering and training, one LoRA setting, three training seeds, one checkpoint.

**Harness.** The base reproduces step 4's base on all 160 shared task-0 episodes (states 0-9 x seeds 16-31), identical in
success and number of steps (`base_vs_step4.txt`, `compare_base_with_step4.py`); this is also why the base is 0 of 16 on
"easy" state 0/0 (step 4 found the same).

Files:

- `replay_check.txt`: launch check before the run, step 5a's A100 hits replayed on a local RTX 4090.
- `step10_power.txt`: the design's power calculation.
- `run/`: the finished run's results (episodes, provenance, training histories, report), without logs, recordings or
  adapter weights.
- `compare_base_with_step4.py`, `base_vs_step4.txt`: the harness check against step 4.
- `check_step10_session.py`: the checks of one session's zip (harness gate, provenance, replays per state, collection,
  adapters, evaluation episode counts without successes, and that a resume kept the previous session's files).
- `session1_checks.txt` to `session4_checks.txt`: its output on each session's zip (sha256 in the first line).

```bash
python -I -S week6/noise_reachability_step10_kaggle/check_step10_session.py <unzipped run> \
    [--previous <previous session's unzipped run>] --zip <zip> --out <txt>
```

## Sessions

| Session | Zip sha256 (first 12) | Done in the session | Adapters trained | Evaluation episodes run |
|---|---|---|---|---|
| 1 (`STOP_AT_H` 7.0) | `fd38be9f9206` | harness, 122 replays, arm A's 120 collection episodes, A0, A1 | 2 of 12 | 280 of 2,608 |
| 2 (`STOP_AT_H` 9.0) | `3f8391ba8240` | A2, N0_0, N6_0, N7_0; evaluation | 6 of 12 | 810 of 2,608 |
| 3 | `990a4db57627` | N0_1, N6_1, N7_1, N0_2; evaluation | 10 of 12 | 1,330 of 2,608 |
| 4 (`STOP_AT_H` 11.0) | `32948f804b01` | N6_2, N7_2; evaluation; analysis | 12 of 12 | 2,608 of 2,608 |

Every session: one commit (`f4369f3`) and clean trees in every provenance record (90, 155, 221, then 366), the harness
gate passes, no traceback in any log. Each resume kept all of the previous session's CSV files unchanged at their start
(58 after session 1, 66 after session 2, 78 after session 3) and repeated no episode.

## Replays: hits reproduce exactly on the hardware they were found on

Every earlier constant-noise success was replayed on the Kaggle T4 with the same noise id and environment seed
(`session1_checks.txt`).

| Where the hits were found | Succeeded again | Same number of steps as the original |
|---|---|---|
| Kaggle T4 (steps 7 and 8: 0/11, 6/8, 6/12, 7/33) | 73 of 73 | 73 of 73 |
| Colab A100 (step 5a: 0/2, 0/5, 0/6, 0/7, 0/8) | 30 of 49 | 2 of 49 |

So a fixed noise gives the same episode on the same GPU type and often a different one on another, as the A100 to RTX
4090 check found before the run (8 of 12 succeeded again, 1 of 12 with the same steps, `replay_check.txt`). Each task
has at least 20 successful replays (37, 33, 33), so all three are read.

**0/5 has no training data for arm N.** Both of its replays failed (0 of 2), so N0 adapters see no episode of state 5.
It stays a *trained* state in the evaluation, as the protocol fixed; expect it near the base there.

## Adapters

All finished 2,000 steps. Arm A: 44 training and 6 validation episodes each; final validation loss A0 0.2026, A1
0.2018, A2 0.199. Arm N (no validation states, as the protocol says): N0_0, N0_1, N0_2 37 training episodes each, N6_0, N6_1,
N6_2, N7_0, N7_1, N7_2 33 each, the successful replays of each task. A runs took about 5.2-5.6 s per
step, N runs about 4.7-5.2.
