# Step 17: does what the policy sees choose a state's noise better than the per-task ticket? (Kaggle 2xT4, 2026-11-23)

Protocol: `noise_reachability/README.md`, step 17, fixed before the run. Notebook
`noise_reachability/kaggle_step17_features.ipynb` at commit `b80c933` (pinned by `3bc5f30`), `lerobot/smolvla_libero` at
`a24f1f3804`, `n_action_steps=10`, 520 steps, `--family repeat`: id 10000 on states 0-29 of tasks 0, 4, 6 and 7, and id
10001 on states 0 of each task (124 episodes). One session, 2026-11-23 21:48-22:57 UTC (first to last provenance
record). Source: `step17_results.zip`, sha256 `2823ab48f9710a6983f36bffb4782ac590346753ecbd2d6687d4a7e3fc450784`,
committed as `run/` without logs (as steps 12-16).

**Checks (2026-11-24).**

- Feature check (the run's harness): 120 of 120 states reproduce step 16's committed id-10000 row (114 needed); id 10001
  on the 4 duplicate states reproduces its row and gives bit-identical features; 968 values per record.
- `analyze_step17.py --root run --out ... --features ...` locally gives a report and `step17_features.json`
  byte-identical to the notebook's (`run/step17_report.txt`, `run/step17_features.json`, sha256 `07abc733...0015`).
- `tools/validate_run.py --expect n_action_steps=10 --expect max_steps=520 --expect family=repeat` passes on each of the
  30 per-state file groups (one call per state; seeds 11000, and 11001 on state 0).
- Provenance: 120 records, one commit `b80c933` with a clean tree, Tesla T4 x2 only, one checkpoint, one package set,
  `noise_ids` 10000-10001 on the 4 duplicate states only. None of the 120 logs (not committed) has a traceback.

**Descriptive, written after the run and before U1 was read** (`run/step17_describe.txt` + `.json`, by
`python noise_reachability/step17_describe.py --features week6/noise_reachability_step17_kaggle/run/step17_features.json
--out week6/noise_reachability_step17_kaggle/run/step17_describe.txt`; contexts `noise_reachability/step16_contexts.json`,
sha256 `cdef6783...1e4f`). The script never fits the critic on the features. Not part of the rule.

| Score | Mean AUROC_s(score) - AUROC_s(ticket) [95%] | p (stratified null) | D, top 5 [95%] |
|---|---|---|---|
| near, features (m = 8, z-scored features) | -0.0196 [-0.0373, -0.0016] | 0.0084 | -0.0300 [-0.0650, +0.0050] |
| near, privileged contexts (= step 16's T2) | +0.0028 [-0.0153, +0.0210] | 0.0006 | +0.0050 [-0.0283, +0.0383] |
| critic (d 8, lambda 10), privileged contexts | +0.0142 [+0.0054, +0.0244] | 0.0010 | -0.0033 [-0.0267, +0.0183] |

115 informative states of 120; intervals are paired bootstraps over states (2,000, seed 19). Even with the
privileged object poses as input, the critic's D is not above 0, so it would not pass U1's rule.

Probe, leave-one-state-out Q^2 from the features to the privileged object coordinates (0 = no better than the
training mean; chance = feature rows permuted, 200 copies, seed 21):

| Input | Pooled Q^2 | Chance mean | Chance 95th percentile |
|---|---|---|---|
| all features, ridge alpha 0.01 / 0.1 / 1 / 10 | +0.140 / +0.159 / +0.164 / +0.058 | -0.395 / -0.311 / -0.094 / -0.005 | -0.301 / -0.234 / -0.061 / +0.003 |
| the critic's input (8 principal components) | +0.081 | -0.118 | -0.087 |

The features carry the object poses above chance but weakly; task 0 least (per task in `run/step17_describe.txt`).

**Test U1:** not yet run with the protocol's 1,000 permutations. The first local attempt (all BLAS threads) was stopped
after 3.5 h without output. It is to run on Kaggle CPU with `noise_reachability/kaggle_step17_u1_cpu.ipynb` (built by
`build_step17_u1_notebook.py`, pinned to `3bc5f30`, single-threaded). A local smoke of that notebook with 3
permutations printed the point estimates, which do not depend on the permutations; they will be reported from the full
run's output.
