# Step 12: does a second round of constant-noise search and training improve on the first? (Kaggle 2xT4, 2026-11-08/14)

Protocol: `noise_reachability/README.md`, step 12, fixed 2026-11-08 before the run. Notebook
`noise_reachability/kaggle_step12_expert_iteration.ipynb` at commit `844f8db` (pinned by `a4099f6`),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps. Round 1 is step 10's committed run
(`week6/noise_reachability_step10_kaggle/run`). Three sessions. Final source: `step12_results.zip` of session 3, sha256
`61734257143effb940f1e3e6d498ff63133e17dd4a23f24d4bef421c0e64452f`, committed as `run/` without logs. Report:
`run/step12_report.txt` (+ `.json`), written by the notebook and reproduced exactly by
`noise_reachability/analyze_step12.py --root week6/noise_reachability_step12_kaggle/run`. Adapter weights and
recordings: the Kaggle output of `expert-iteration-step-12`, archived (see `tools/check_archive.py`).

**Result under the rule fixed in advance: test 1 passes**, R beats N on the searched states in all three seeds
(exact one-sided McNemar, 192 paired episodes each):

| Seed | R | N | only R / only N | p |
|---|---|---|---|---|
| 0 | 46 (24.0%) | 29 (15.1%) | 27 / 10 | 0.00382 |
| 1 | 57 (29.7%) | 27 (14.1%) | 39 / 9 | 7.61e-06 |
| 2 | 46 (24.0%) | 32 (16.7%) | 26 / 12 | 0.0168 |

Pooled over seeds 149 of 576 against 88 of 576 (+10.6 points; mean over seeds +10.6, std 4.4). The hypothesis'
10 points is met on average, by seed 1 alone of the three; as the protocol says, the pass shows improvement, not its
size. The gain is where round 2 added data: task 0 60 to 106 of 192 (64 episodes per seed, three seeds pooled; mostly 0/2 and 0/11), task 6 27 to 42 of
288 (96 per seed, pooled);
6/14 0 in every policy, 7/38 1 against 1.

Reported, not part of the rule: held out (6/20, 7/47) 1 against 1 of 192, no transfer; check states 112 against 121
of 144, all on 0/8: R below N in every seed (R0-R2 10, 12, 10 of 16 against N0-N2 12, 13, 16), most of it in seed 2
(N2 was 16 of 16), so round 2 may cost part of round 1's gain on its best state. The report's seeds-pooled McNemar
(two-sided p 0.0225) is not quoted: N0-N2 and R0-R2 share their evaluation episodes and evaluation is deterministic,
so the 3 x 48 pairs are not independent and the pooled p overstates the evidence (independent review, 2026-11-15).
The same holds for the report's pooled searched-state p (1.66e-08); test 1, per seed, is not affected.

**Checks.** R7_<s> trained on N7_<s>'s set with the same seed, and its 96 evaluation episodes equal N7_<s>'s in success
and steps for every seed (288 of 288): evaluation is deterministic here, so the differences on tasks 0 and 6 come from
the adapters. N's rate on the searched states with fresh seeds (15.3%) is close to its step-10 rate (14.2%). Every
(seed, state) is present once per policy (1,824 episodes), no log has a traceback.

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

## Session 2 (2026-11-14 00:27 to 11:18 UTC, host `ad4d77b88f58`)

Checks: `session2_checks.txt`, written by `check_step12_session.py --previous` (session 1's run) from
`step12_results.zip` (sha256 `f750aac72ec6cb9ca94146266caf493215dfa5817431e7f12f28a85161b1d138`).

- Resumed from session 1: all 35 of session 1's CSV files are unchanged at the start of this session's (a resume
  appends, never rewrites). Provenance: 139 records, one commit (`844f8db`), clean trees, the protocol's settings.
- Harness and search carried over unchanged (12 of 12; 21 of 400).
- Training finished: all 9 round-2 adapters reached step 2,000, each on round 1's episodes plus every search success of
  its task. R7_1 and R7_2, like R7_0, trained on N7's set only, and their losses equal N7_1's and N7_2's at all 41
  logged steps.
- Evaluation: 757 of 1,824 episodes (task 0 complete for N0-N2 and R0-R2; task 6 N0 and N1 complete, N2 21 of 128;
  task 7 not started). Counts only; no success read.
- Task 6 N2 part 0 has two provenance records in this session (11:03 and 11:18 UTC): the job started twice. Its 10
  episodes are distinct (seed, state) pairs, so the restart duplicated nothing.

Left for session 3: 1,067 evaluation episodes, no training.

## Session 3 (2026-11-14 16:09 to 22:57 UTC, host `100e66b21b57`)

Checks: `session3_checks.txt` (`check_step12_session.py --previous`, session 2's run). Session 2's 55 CSV files are
unchanged at the start of this session's; 298 provenance records, one commit (`844f8db`), clean trees; evaluation
1,824 of 1,824. The notebook then ran the analysis (above).
