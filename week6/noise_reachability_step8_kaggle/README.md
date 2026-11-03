# Step 8: fixed noise on states that resist sampling (Kaggle 2xT4, 2026-11-02/03)

Protocol: `noise_reachability/README.md`, step 8, fixed before the run. Commit `50a98bd` (clean tree in all 362
provenance records), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, tasks 0, 4, 6, 7. Two
sessions, the second resuming from the first's output: session 1 on host `8155ea7b7423` (237 records), session 2 on
host `eb9f41883bd5` (125 records), one package set. Source: the session-2 `step8_results.zip`, sha256
`1d16be3e7a55138c9011cc2f9ac64b78e1d6943abcc9cf7c0f429d2a4764dfc2`. The session-1 zip
(`7e7fc251144bcd74c9a8e270e775ae7f95b6064994ab36302694608edc10be80`) stopped at its deadline by design and was not read
as a result.

**Result under the rule fixed in advance: both tests pass.** Fixed noise beats sampling on resistant states beyond
task 0, on a state and on unlabelled states chosen in advance (`run/step8_report.txt`).

| Test | States | sampled | repeat | One-sided Fisher p | Holm |
|---|---|---|---|---|---|
| 1 | 6/8, noise ids 100-399 | 2/300 (0.1-2.4%) | 10/300 (1.6-6.0%) | 0.0184 | passes |
| 2 | 0/11, 4/11, 6/12, 7/33 pooled, ids 1000-1599 | 1/600 (0.0-0.9%) | 38/600 (4.5-8.6%) | 4.1e-11 | passes |

Intervals are exact 95%. Per state, of 150 each:

| State | sampled | repeat | Jittered successes per confirmed hit, of 4 |
|---|---|---|---|
| 0/11 | 0 | 3 | 0, 1, 3 |
| 4/11 | 0 | 3 | 0, 1, 1 |
| 6/12 | 1 | 7 | 0, 2, 1 |
| 7/33 | 0 | 25 | 3, 3, 2 |
| 6/8 (of 300) | 2 | 10 | 2, 3, 0 |

Jittered successes, all states: 22 of 60 (descriptive). Three of the four test-2 states stayed at 0 of 150 `sampled`
(one-sided 95% bound 1.98% each).

Gates. Harness, base of 30 on init states 0-9 x seeds 0-2 against step 0: task 0 6 (7), task 4 15 (14), task 6 16 (20),
task 7 14 (18); gaps -1, +1, -4, -4, all within 6. Controls of 35, `sampled` then `repeat`: 0/9 8, 12; 4/5 12, 11; 6/0
14, 8; 7/4 15, 12; all passed, so all four tasks were read. Screen: states at 0 of 8 among 10-29, tasks 0, 4, 6: 5, 4, 5
(14 went on to the second part); eligible at 0 of 32, in selection order: 0/11, 4/11, 6/12, 7/33, 0/14, 4/18, 6/14,
7/38, 0/27, 4/26, 6/20, 7/47. The first four were tested; no state failed the scene check. The design expected about
14 eligible states on tasks 0, 4, 6 (estimate); 9 were.

**Read with care (exploratory, not part of the rule; `step8_checks.txt`).**

- **One state carries most of test 2.** 7/33 gives 25 of the 38 `repeat` hits. Without it, 13/450 against 1/450, p
  0.00085; without any other one state, p under 3e-10.
- **Per state, only 7/33 (p 1e-8) and 6/12 (p 0.033) stand out alone.** 0/11 and 4/11 are 3 against 0 (p 0.124 each).
  So the claim is pooled, as the rule says, not "on every state".
- **`repeat` is not better in general.** On the easy control states it is level with or below `sampled`: 43 against 49
  of 140 pooled, and on 6/0 8 against 14 (p 0.099 the other way). The finding is about states that resist sampling.
- **Unlike step 7, the labels held.** Eligible states tested at 1 of 600 `sampled`, so these were resistant, not only
  rare as step 7's 4/2 and 8/4 were.
- **Harness drift.** The base ran 4 below step 0 on tasks 6 and 7, within the tolerance fixed in advance.
- What it cannot show stays as the protocol says: one checkpoint, families not a findable noise, states not a random
  sample of hard states.

**Checks.** `analyze_step8.py --out` rerun locally reproduces the Kaggle report and its JSON byte for byte.
`tools/validate_run.py` passes on the four `harness` folders with `--seeds 0-2 --init_states 0-9 --expect
n_action_steps=10`. It does not read the other folders, so `check_step8_rows.py`, which shares no code with the
analysis, recounts them: every control and test file has its noise ids exactly once with the right task and state,
the confirmation files are the first three `repeat` hits with jitter seeds 5000-5003, and all counts, both p values
and Holm agree. Every one of the 121 files of session 1 is either unchanged in the final zip or a prefix of its final
version, so the resume rewrote nothing. The 122 logs in `run/logs/` (not committed) have no traceback, error or
`WORK REMAINS` line.

Episodes: 120 harness, 280 control, 912 screen, 1,800 test and 60 jittered, 3,172 in all (at most 3,316 planned). Run
time between first and last provenance records: 10.3 h (session 1) and 5.9 h (session 2), 16.2 h against 18.1 h
estimated.
