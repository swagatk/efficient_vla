# Step 12: does a second round of constant-noise search and training improve on the first? (Kaggle 2xT4, running)

Protocol: `noise_reachability/README.md`, step 12, fixed 2026-11-08 before the run. Notebook
`noise_reachability/kaggle_step12_expert_iteration.ipynb` at commit `844f8db` (pinned by `a4099f6`),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps. Round 1 is step 10's committed run
(`week6/noise_reachability_step10_kaggle/run`). No result yet: the run needs about three sessions (estimate), and
nothing below reads an evaluation success.

## Session 1 (2026-11-08 22:12 to 2026-11-09 08:59 UTC, host `e245339a43e1`)

Checks: `session1_checks.txt`, written by `check_step12_session.py` from `step12_results.zip` (sha256
`257a3c6c44533dc96d88652b08dad59207c4aae686b68d31999b4cd79da6f4c0`; recordings and adapter weights stay in the Kaggle
output, which the next session resumes from).

- Provenance: 93 records, one commit (`844f8db`), clean trees, the protocol's settings in every folder.
- Harness gate passed: 12 of 12 of step 10's episodes reproduced exactly in success and steps (at least 11 needed).
- Search (round 1's N<t>_0, `repeat` noise ids 7000-7049, 50 episodes per state): 21 of 400 successes.
  0/2 2, 0/5 2, 0/6 4, 0/11 5, 6/8 6, 6/12 2, **6/14 0, 7/38 0**.
- Training sets of the four finished round-2 runs are round 1's episodes plus every search success of the task:
  R0_0 and R0_1 37 + 13, R6_0 33 + 8, R7_0 33 + 0.
- Done: harness, search, 4 of 9 trainings (R0_0, R0_1, R6_0, R7_0), task 0's evaluation of N0-N2, R0 and R1, and 41 of
  task 6 N0's 128 episodes: 441 of 1,824 evaluation episodes.

**Read with care.** Task 7's search found nothing, so R7_<s> trains on exactly N7_<s>'s set, with the same seed and
code. R7_0's training loss equals N7_0's at all 41 logged steps: training reproduced bit for bit across sessions, and
R7 is in effect N7. Task 7 (one searched state, 32 of the 192 paired episodes per seed) therefore cannot differ in
test 1 except through evaluation nondeterminism, and its held-out state 7/47 and check state 7/33 cannot show transfer
or forgetting. State 6/14 also gained no training data. The test stays as fixed in advance; its power
(`step12_power.txt`, computed with round 2 improving on every searched state) is an overestimate for this run.
