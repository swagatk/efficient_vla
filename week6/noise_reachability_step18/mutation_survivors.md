# Step 18: mutation-check survivors (`tools/mutation_check.py`, 2026-11-25)

First run on `analyze_step18.py` with `tests/test_analyze_step18.py`: 113 of 135 killed
(`mutation_analyze_step18.txt`). Tests were added for every survivor that could change a number or let a wrong
run through (the sha256 check, the plain-search refusal, the strict "keeps half", the reported counts and the
paired McNemar, the report of a stopped run, the gate's exit code, `ratio_interval` with one success). The re-check of
the changed functions kills 68 of 72 (`mutation_analyze_step18_recheck.txt`). Kept, with reasons:

- line 312 `all -> any` in `analyse()`: an assert that tested conditions have equal episodes; `episodes()` already
  refuses any incomplete block, so it cannot fail on a run the analysis accepts.
- line 399 `required=True -> False` for `--root`: argparse only; without `--root` the call fails either way.
- lines 409 and 416 `indent=1 -> 2`: JSON layout of the written files.
- line 204 `<` -> `<=` in `check_settings()` (first run; not in the re-check's functions): a block index equal to the
  number of blocks raises an IndexError instead of the provenance message, so the record is refused either way.

After the review's fix (per-session harness, one folder per state and block), the changed functions were checked
again (`mutation_analyze_step18_harness.txt`): 43 of 46 killed. Kept: line 175 in `sessions()` (two mutants), the
range printed in the error message; line 217 in `check_settings()` `<` -> `<=`, as above (an out-of-range block index
raises IndexError, so the record is refused either way).
