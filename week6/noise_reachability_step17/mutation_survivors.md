# Step 17 mutation checks: survivors and why they cannot change a result (2026-11-23)

Full checks on the final modules (`mutation_check_<module>.txt`), then the functions whose real gaps got new tests
re-checked (`mutation_recheck_<module>.txt`). Line numbers are those of the committed files.

## step17_g1b.py: 143 mutants, 122 killed in the full check; paired_ci and format_report then 22 of 22

The two real gaps of the full check (`paired_ci`'s quantile levels, line 195; the order of the interval bounds in
`format_report`, lines 263-264) got tests (`test_paired_ci_takes_the_2_5_and_97_5_percent_quantiles`,
`test_report_prints_each_interval_bound_in_its_place`) and are killed in the re-check. The other 17:

- line 74 (`sd > MIN_STD` -> `>=`), line 83 (`scale < MIN_STD` -> `<=`, `1.0` -> `2.0`): tolerance boundaries at 1e-8 and
  the scale given to a zero-variance component (whose scores are zero either way).
- line 76 (`full_matrices=False` -> `True`): the first d rows of `vt` are the same.
- line 81 (`comp * sign` -> `comp / sign`): the sign is +1 or -1.
- line 118 (`maxiter` 2000 -> 2001, `gtol` 1e-6 -> 1.000001e-6): optimiser limits, not reached / not binding.
- line 133 (`X[s:s + 1][0]` -> `X[s:s + 2][0]`): the same row.
- lines 239, 245, 248: text inside error messages.
- line 242 (the id filter's `k[0] == t` -> `!=`, `k[0]` -> `k[1]`): on a complete table the same ids; on an incomplete one
  the missing-cells check still raises ValueError (only the message differs).
- line 279 (`required=True` -> `False` for `--features`): only matters when the flag is left out (then a later error).
- line 292 (`indent=1` -> `2`): JSON layout of the report.

## analyze_step17.py: 59 mutants, 52 killed in the full check; run_rows, read_record, format_report, main then 31 of 34

The real gaps of the full check (a stray row of another state, line 93; `inferences < 1` at its edge, line 111; the
duplicate failure in the report, line 169) got tests (`test_a_row_of_another_state_is_refused`,
`test_one_inference_is_enough`, `test_report_names_failing_duplicates_and_writes_no_features`). The 3 left:

- line 179 (`required=True` -> `False` for `--root`): only matters when the flag is left out.
- line 186 (`g is None` -> `is not None` inside the printed gate message): the message only; the exit code, which the
  notebook reads, is tested.
- line 193 (`indent=1` -> `2`): JSON layout of the report.

## first_features.py: 20 mutants, 18 killed in the full check; write_features and sample_actions then 14 of 15

The real gap (`mkdir(parents=True)` -> `parents=False`, line 57) got a test that writes into a new nested folder
(`test_two_records_go_into_one_folder`). The 1 left:

- line 36: text inside an error message (the batch size it reports).

## step17_describe.py (descriptive, written after the run, before U1): 90 mutants, 68 killed in the full check; context_tasks, format_report, main then 15 of 17

The real gaps of the full check (the held-out-state refusal `any` -> `all`, line 177; the interval order and the
no-informative-state branch of the report, lines 215, 216, 222; the provenance hashes, line 251-252, now full sha256)
got tests (`test_context_tasks_take_the_rows_of_the_kept_states`,
`test_report_prints_intervals_in_order_and_tasks_without_informative_states`, `test_main_runs_end_to_end_on_small_files`).
The survivors kept:

- lines 60, 99 (`sd > MIN_STD` -> `>=`): differ only for a column whose sd is exactly 1e-8.
- line 71 (the tolerance of the near-vs-T2 consistency assertion): a guard; a looser tolerance cannot change a number.
- lines 103, 124, 125 (signs of the centring terms `T - tm`, `F - fm`, `f - fm`): the features are z-scored (ridge) or
  PCA-projected (pca probe) on the training states, so their training columns have mean 0: Z^T 1 = 0 makes the target
  centring a no-op in the ridge solution, and fm = 0 up to rounding. Equivalent mutants.
- line 121 (`X[s:s + 1][0]` -> `X[s:s + 2][0]`): the same row.
- line 132 (`keepdims=True` -> `False`): broadcasting gives the same array.
- line 240 (`required=True` -> `False` for `--features`): only matters when the flag is left out.
- line 258 (`indent=1` -> `2`): JSON layout of the report.
