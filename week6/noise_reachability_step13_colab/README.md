# Step 13: do the distilled adapters change success on initial states they never trained on? (Colab T4, 2026-11-16 to 2026-11-19)

Protocol: `noise_reachability/README.md`, step 13, fixed 2026-11-15 before the run. Notebook
`noise_reachability/colab_step13_unseen_states.ipynb` at commit `26fe1d4`, `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, 520 steps, sampled noise, one process at a time on a standard Colab T4 runtime. Policies: the
base, step 10's N<t>_0-N<t>_2 and step 12's R<t>_0-R<t>_2 (R7_s not run: its weights are N7_s's). Tasks 0, 6, 7, states
30-49 minus every trained state (59 states), seeds 200-207. Eight sessions (started 2026-11-16 10:25, 11-16 20:53,
11-17 08:38, 11-17 18:56, 11-18 10:49, 11-18 19:56, 11-18 22:17 and 11-19 10:53 UTC), each on its own host.
Source: the Drive folder `efficient_vla_runs/step13/run` downloaded after session 8 (`run-20261119T113842Z-1-001.zip`,
sha256 `b7e3db5808e9fab5ae2f22d8327036de5bb5ab636ce381a9176ca2c39283ce56`), committed as `run/` without logs (as steps
12, 14 and 15) and without the empty `harness_pending/`. Report: `run/step13_report.txt` (+ `.json`), written by the
notebook and reproduced exactly by
`python3 -S noise_reachability/analyze_step13.py --root week6/noise_reachability_step13_colab/run`.
Power: `step13_power.txt`.

**Result under the rule fixed in advance: neither N nor R changes task success on unseen initial states.** No change of
10 points or more is shown; smaller changes are not ruled out. This is what the hypothesis expected.

Per training seed, adapter against the base on the same 472 paired episodes, exact two-sided McNemar (from the report):

| | Adapter successes | Base successes | Only adapter / only base | Two-sided p |
|---|---|---|---|---|
| N, seed 0 | 260 | 253 | 92 / 85 | 0.652 |
| N, seed 1 | 286 | 253 | 107 / 74 | 0.0171 |
| N, seed 2 | 253 | 253 | 82 / 82 | 1 |
| R, seed 0 | 268 | 253 | 96 / 81 | 0.293 |
| R, seed 1 | 292 | 253 | 115 / 76 | 0.00582 |
| R, seed 2 | 260 | 253 | 90 / 83 | 0.648 |

The rule needs p < 0.05 in at least 2 of 3 seeds with the same sign; for both N and R only seed 1 has it (a gain).
Reported, not part of the rule: R against N per seed, p 0.466, 0.556 and 0.5; pooled counts N 799, R 820, base 759 x 3
(the seeds share their episodes, so no pooled p).

Per task (reported, not part of the rule; successes of 160 for tasks 0 and 6, of 152 for task 7):

| Task | base | N0 | N1 | N2 | R0 | R1 | R2 |
|---|---|---|---|---|---|---|---|
| 0 | 58 | 99 | 99 | 87 | 89 | 94 | 87 |
| 6 | 89 | 71 | 85 | 77 | 89 | 96 | 84 |
| 7 | 106 | 90 | 102 | 89 | 90 | 102 | 89 |

The per-task counts point in different directions: on task 0 every adapter is above the base, on task 7 every one is
below it, and on task 6 all but R1 are at or below it. The pooled test combines these. No per-task test was fixed in
advance, so any per-task reading is post hoc.

**Checks.**

- Harness gate passed in all eight sessions: 12 of 12 of the Colab T4 check's episodes reproduced each time (at least
  11 needed); no session went to `harness_failed/`.
- One package set in all eight `environment_session*.json` (Tesla T4, driver 580.82.07, torch cu130 as pinned); only
  the start time and available RAM (11.9 or 12.0 GB) differ.
- 316 provenance records, one commit (`26fe1d4`) with a clean tree.
- `tools/validate_run.py --expect n_action_steps=10` passes on every `eval/<policy>/task_<t>` with that task's states
  and seeds 200-207: every (seed, state) once, nothing else; 2,848 rows. (Its "no eval.log" message is expected:
  this notebook writes logs to `run/logs/`, not committed.) No log has a traceback; peak RSS 5.2 GB.
- Each Drive download was compared with the one checked before it: earlier csv/jsonl files changed only by
  appended rows, and earlier harness folders stayed byte-identical.
