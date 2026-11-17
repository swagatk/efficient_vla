# Step 15: are 6/14 and 6/20 unreachable, and did distillation lose 7/38's solutions? (Kaggle 2xT4, 2026-11-17)

Protocol: `noise_reachability/README.md`, step 15, fixed 2026-11-16 before the run (`5f26ed0`). Notebook
`noise_reachability/kaggle_step15_unreachable.ipynb` at commit `16a03fb` (pinned by `dafd8c0`),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, `--family repeat`. Input: step 10's Kaggle
output (adapters N7_0-N7_2). One session (2026-11-17 08:37 to 18:41 UTC, host `5c6cd394465d`), not the two estimated.
Source: `step15_results.zip`, sha256 `2836beffbc74a155022fb065ab32b5480426b421202fa968745d11fa8e7f6d4a`, committed as
`run/` without logs (as steps 12 and 14). Report: `run/step15_report.txt` (+ `.json`), written by the notebook and
reproduced exactly by `python3 -S noise_reachability/analyze_step15.py --root week6/noise_reachability_step15_kaggle/run`.
Power: `step15_power.txt`.

**Result under the rule fixed in advance: tests A and B both pass.**

Test A, the base with fresh `repeat` noise ids 9000-9399 (400 per state); a success counts only if its replay with the
same id also succeeds. Exact 95% intervals:

| State | Successes of 400 | Ids | Replayed | Interval | Test A |
|---|---|---|---|---|---|
| 6/14 | 1 | 9008 | 1 of 1 | 0.01-1.38% | passes |
| 6/20 | 3 | 9056, 9174, 9212 | 3 of 3 | 0.15-2.18% | passes |

Each replay succeeded in the same number of steps (9008: 391; 9056: 286, 9174: 425, 9212: 412). Both states are rare,
not unreachable, so every resistant state searched with constant noise so far has been solved by one.

Test B, 7/38, the same 200 noise ids 9000-9199 for each policy, exact one-sided McNemar (base better):

| Policy | Successes of 200 | Interval | Only base / only N | One-sided p |
|---|---|---|---|---|
| base | 19 | 5.8-14.4% | | |
| N7_0 | 5 | 0.8-5.7% | 18 / 4 | 0.00217 |
| N7_1 | 7 | 1.4-7.1% | 15 / 3 | 0.00377 |
| N7_2 | 2 | 0.1-3.6% | 18 / 1 | 3.81e-05 |

The rule needs p < 0.05 in at least 2 of 3 seeds; it holds in all three. Distillation on 7/33 removed constant-noise
solutions of 7/38 that the base had, so later rounds should also search with the base. Two-sided p (reported, not
part of the rule): 0.00434, 0.00754, 7.63e-05.

**Checks.**

- Scene check: 6/14, 6/20 and 7/38 all give the same scene on two seeds with nothing moved; no state was dropped.
- Harness gate passed: 14 of 14 committed episodes reproduced in success and steps (base on 7/38, step 9's ids: 10/10;
  N7_0 on 7/33, step 12's seeds 16-19: 4/4; at least 13 needed). Step 9 ran at another commit (`f2a067e`), so its
  constant-noise episodes replay exactly at this commit.
- `analyze_step15.py`'s provenance check: 167 records, one commit with a clean tree, T4 GPUs only, one package set and
  one checkpoint, and each record's settings match the protocol for its folder.
- `tools/validate_run.py --expect n_action_steps=10 --expect max_steps=520` passes on the four test-B folders (seeds
  10000-10199, state 38) and on each test-A part and state (one state per call, the other's file excluded): every
  (seed, state) once, nothing else. None of the 14 logs has a traceback, error or out-of-memory kill.
- The report's counts, ids and McNemar p-values were recounted from the CSVs by a separate script; they match.
