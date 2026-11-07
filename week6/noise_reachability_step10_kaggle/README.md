# Step 10: can constant-noise successes be learned into the weights? (Kaggle 2xT4, in progress)

Protocol: `noise_reachability/README.md`, step 10, fixed 2026-11-03. Notebook
`noise_reachability/kaggle_step10_learn_from_noise.ipynb` at commit `f4369f3`, `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, tasks 0, 6, 7. The run spans several Kaggle sessions; each resumes from the previous version's
output. **No evaluation success has been read; the tests are read only when every policy is finished.**

Files:

- `replay_check.txt`: launch check before the run, step 5a's A100 hits replayed on a local RTX 4090.
- `step10_power.txt`: the design's power calculation.
- `check_step10_session.py`: the checks of one session's zip (harness gate, provenance, replays per state, collection,
  adapters, evaluation episode counts without successes, and that a resume kept the previous session's files).
- `session1_checks.txt`, `session2_checks.txt`: its output on each session's zip (sha256 in the first line).

```bash
python -I -S week6/noise_reachability_step10_kaggle/check_step10_session.py <unzipped run> \
    [--previous <previous session's unzipped run>] --zip <zip> --out <txt>
```

## Sessions

| Session | Zip sha256 (first 12) | Done in the session | Adapters trained | Evaluation episodes run |
|---|---|---|---|---|
| 1 (`STOP_AT_H` 7.0) | `fd38be9f9206` | harness, 122 replays, arm A's 120 collection episodes, A0, A1 | 2 of 12 | 280 of 2,608 |
| 2 (`STOP_AT_H` 9.0) | `3f8391ba8240` | A2, N0_0, N6_0, N7_0; evaluation | 6 of 12 | 810 of 2,608 |

Both sessions: one commit (`f4369f3`) and clean trees in every provenance record (90, then 155), the harness gate
passes, no traceback in any log. Session 2 kept all
58 of session 1's CSV files unchanged at their start and repeated no episode.

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

## Adapters so far

All finished 2,000 steps. Arm A: 44 training and 6 validation episodes each; final validation loss A0 0.2026, A1
0.2018, A2 0.199. Arm N (no validation states, as the protocol says): N0_0 37, N6_0 33, N7_0 33 training episodes,
the successful replays of each task. A runs took about 5.2-5.6 s per step, N runs about 4.8.
