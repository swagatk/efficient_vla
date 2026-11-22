# Step 16: is which constant noise suits a state shared across a task, and predictable from the scene? (Kaggle 2xT4, 2026-11-21/22)

Protocol: `noise_reachability/README.md`, step 16, fixed before the run (`bc39d05`). Notebook
`noise_reachability/kaggle_step16_crossed_map.ipynb` at commit `e7b7cd9` (pinned by `ddcf465`),
`lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520 steps, `--family repeat`, ids 10000-10049 on states
0-29 of tasks 0, 4, 6 and 7 (6,000 map episodes in five parts). Three sessions, not the four estimated (first to last
provenance record): 2026-11-21 10:15-21:03 UTC (host `cb50abc1d970`), 2026-11-21 23:38 to 11-22 10:27 (`8ef05086280b`)
and 2026-11-22 12:50-19:49 (`0269e8c2e205`), 10.8 h, 10.8 h and 7.0 h. Source: the third session's
`step16_results.zip` (it holds every session's rows), sha256
`78d59ede1075b95990197157d39c0901098ed500946b48fc3c79878df1dcb1c1`, committed as `run/` without logs (as steps 12-15).

Files in `run/`: the notebook's reports after parts 1, 3 and 5 (`step16_report_parts<n>.txt` + `.json`) and tables
(`step16_table_parts<n>.csv`); `step16_g1_report.txt` (+ `.json`), tests T1 and T2, written locally by
`python noise_reachability/step16_g1.py --table week6/noise_reachability_step16_kaggle/run/step16_table_parts5.csv
--contexts noise_reachability/step16_contexts.json --out week6/noise_reachability_step16_kaggle/run/step16_g1_report.txt`
(contexts sha256 `cdef6783...1e4f`); `step16_describe.txt` (+ `.json`), the descriptive breakdown below, written by
`noise_reachability/step16_describe.py` with the same arguments. Power: `step16_power.txt`, `step16_level.txt`.

**Result under the rule fixed in advance: T1 and T2 both pass (Holm, 0.05).**

| Test | Statistic | Informative states | Null mean | One-sided p | Holm |
|---|---|---|---|---|---|
| T1: mean AUROC_s(flat) - 0.5 | 0.1689 | 115 of 120 | -0.0001 | 0.0002 | passes |
| T2: mean AUROC_s(near) - AUROC_s(flat) | 0.0028 | 115 of 120 | -0.0320 (sd 0.0098) | 0.0006 | passes |

Reported, not part of the rule (`step16_g1_report.txt`, `step16_describe.txt`):

| Task | Successes of 1,500 | Informative states | Mean AUROC flat | Mean AUROC near (m = 8) | Best ticket (states solved of 30) |
|---|---|---|---|---|---|
| 0 | 448 | 26 | 0.651 | 0.614 | 10009 (19) |
| 4 | 457 | 30 | 0.650 | 0.672 | 10004 (17) |
| 6 | 485 | 29 | 0.640 | 0.666 | 10036 (19) |
| 7 | 659 | 30 | 0.731 | 0.726 | 10017 (23) |

**Reading.** T1: a constant noise's success is strongly shared across a task's states; ranking the ids by their
success on the task's other states gives a mean within-state AUROC of 0.64-0.73. T2: the scene's 8 nearest states rank a
state's ids better than 8 states of similar difficulty with the scene link broken (z 3.5; the null's maximum over 5,000
permutations is 0.0055), so a scene-dependent component exists. It is small: in absolute terms near (8 states) ranks the
ids about as well as flat (all 29 other states), 0.0028 AUROC better on average, worse on tasks 0 and 7. The null mean
is negative because averaging fewer states is noisier. Under the protocol's reading ENEI goes ahead, with the VLA's own
features of the first observation in place of the privileged init state; the claim to beat there is the per-task
ticket (flat), not random noise, and this map gives no evidence that a scene-based choice beats it by much.

**Checks.**

- Harness gate passed: 12 of 12 of step 15's committed 7/38 episodes reproduced (at least 11 needed).
- `analyze_step16.py` (provenance per folder, scene checks, complete parts, part 1's gate: 98 of 120 informative, 12
  needed) on the committed folder reproduces the notebook's parts-5 report and table byte for byte.
- Provenance: 608 records, one commit `e7b7cd9` with a clean tree, Tesla T4 x2 only, one package set, one checkpoint.
- `tools/validate_run.py --expect n_action_steps=10 --expect max_steps=520 --expect family=repeat --expect noise_ids=...`
  passes on each of the 600 (task, part, state) files (one state per call; the `seed` column is 1000 + noise id): every
  episode once, nothing else. None of the 721 logs (not committed) has a traceback.
- Resuming: the rows of sessions 1 and 2 are unchanged in the final zip, and the parts-1 and parts-3 tables equal the
  ones checked after those sessions.
