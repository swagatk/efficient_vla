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

**Test U1** (`u1/`, from `step17_u1_results.zip`, sha256
`e6a32fa2fcb7a11aa257690f9d0f574391085d11fa8898addf819584d6c5164d`; notebook
`noise_reachability/kaggle_step17_u1_cpu.ipynb` pinned to `3bc5f30`, Kaggle CPU, 2026-11-24). Command (in
`u1/provenance.json`): `noise_reachability/step17_g1b.py --features .../step17/step17_features.json --out
.../step17_u1.txt`. The first local attempt (all BLAS threads) was stopped after 3.5 h without output, so U1 ran on
Kaggle with one BLAS thread, 31.2 min.

Checks (2026-11-24): commit `3bc5f30` with a clean tree; features sha256 `07abc733...0015`, the same as `run/`; one
BLAS thread; numpy 2.2.6, scipy 1.17.1. U1, D, the intervals and the per-task values match a local smoke run with 3
permutations exactly (the point estimates do not depend on the permutations). `step17_u1.txt` and
`step17_u1_stdout.txt` are identical. `tools/validate_run.py` does not apply, since this run has no episodes.

| | Value |
|---|---|
| U1 (115 informative states) | +0.0072, 95% interval +0.0006 to +0.0136 |
| stratified null (1,000 permutations, strata of 5) | mean -0.0035, sd 0.0030; one-sided p 0.000999 |
| D (top 5 ids, critic - ticket, 120 states) | -0.0083, 95% interval -0.025 to +0.008 |
| interaction off (lambda 1e8) | U1 +0.0033, D -0.0033 |
| per task 0 / 4 / 6 / 7 (reported, not tested): U1 | -0.0083 / +0.0170 / +0.0080 / +0.0100 |
| per task: D | -0.0067 / +0.0067 / 0.0000 / -0.0333 |
| per task: mean AUROC critic / ticket | 0.643/0.651, 0.667/0.650, 0.648/0.640, 0.741/0.731 |

**U1 does not pass**: p <= 0.05 and U1 > 0, but D is below 0. The README's reading for this case applies: a scene
signal reaches the features but does not beat the per-task ticket, so ENEI reduces to the ticket and D1 is not run for
it. Per state, D is +0.2 on 10 states, -0.2 on 15 and 0 on 95.
