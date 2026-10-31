# Step 4: weight updates on task 0's systematic states (Kaggle 2xT4, 2026-10-29/31)

Protocol: `noise_reachability/README.md`, step 4, fixed before the run. Commit `9318dd5` (clean tree in all 142 eval and
harness provenance records), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, task 0, init states 0-9 x
seeds 16-31. Source: `step4_results (2).zip`, sha256
`7cf2a96020128343d034cbe8d4bfbc95c503c9c4475039581df27234de7ed495`.

**Result: negative under the rule fixed in advance.** No arm meets the hypothesis in any training seed: the best is a
net +3 of 80 systematic episodes, against the +8 required. Arm B, the larger update, also fails, so the README's reading
applies: the study is written up as the step 0-3b measurements with this limit stated.

Harness: base 6/30 against step 0's 7 (gap -1, limit 6).

| Arm | Systematic /80 | Net | p | Reachable /80 | Net | p |
|---|---|---|---|---|---|---|
| Base | 0 | | | 37 | | |
| A0, self-filtered LoRA | 1 | +1 | 1 | 48 | +11 | 0.00342 |
| A1 | 2 | +2 | 0.5 | 38 | +1 | 1 |
| A2 | 3 | +3 | 0.25 | 47 | +10 | 0.0129 |
| C0, demonstration LoRA | 0 | 0 | 1 | 40 | +3 | 0.549 |
| C1 | 0 | 0 | 1 | 41 | +4 | 0.388 |
| C2 | 0 | 0 | 1 | 40 | +3 | 0.508 |
| B, continued training | 2 | +2 | 0.5 | 62 | +25 | 4.17e-07 |

No arm failed the reachable guard. C was not ahead of A in any seed (0 vs 1, 0 vs 2, 0 vs 3).

Successes per init state, of 16 (systematic states 2, 5, 6, 7, 8):

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0 | 15 | 0 | 15 | 3 | 0 | 0 | 0 | 0 | 4 |
| A0 | 0 | 16 | 0 | 16 | 5 | 0 | 1 | 0 | 0 | 11 |
| A1 | 0 | 16 | 0 | 16 | 0 | 0 | 2 | 0 | 0 | 6 |
| A2 | 0 | 16 | 0 | 16 | 2 | 0 | 3 | 0 | 0 | 13 |
| B | 9 | 16 | 0 | 16 | 8 | 0 | 2 | 0 | 0 | 13 |
| C0 | 0 | 16 | 0 | 16 | 5 | 0 | 0 | 0 | 0 | 3 |
| C1 | 0 | 16 | 0 | 16 | 2 | 0 | 0 | 0 | 0 | 7 |
| C2 | 0 | 15 | 0 | 16 | 4 | 0 | 0 | 0 | 0 | 5 |

**Read with care (exploratory, not part of the rule).**

- All 8 systematic successes are on state 6, one of the two states step 5a classed as robust. States 2, 5, 7 and 8 are
  0 in every arm.
- Arm C is not a clean test of demonstrations. Its loss on the demonstrations started low and did not fall: the mean of
  the first 8 logged values against the last 8 was 0.0431 to 0.0508, 0.0637 to 0.0484 and 0.0306 to 0.0412. Arm A's fell
  from 0.3093, 0.2773, 0.2061 to 0.1140, 0.1210, 0.0969. The adapters did change behaviour (9-12 of 160 outcomes
  differ from the base), but "demonstrations do not help" goes further than this data does. What it shows is that this
  LoRA update on 33 demonstrations did not reach those states.
- The training losses are noisy logged values, not a validation measure.

**Checks.** `analyze_step4.py` rerun locally reproduces the Kaggle report byte for byte. `tools/validate_run.py` passes
on each of the 8 arm folders; run on `eval/` as a whole it fails by construction, because merged files and parts are
both present and base and B share a file name. It cannot check `n_action_steps` in the logs here (they are not named
`eval.log`, and the setting is not printed), so that setting is confirmed from the provenance records. No log has a
traceback. Merged files equal their parts, each arm has all 160 episodes once, and harness rows are 30 distinct episodes.
Rows from the earlier sessions that are also in this zip are unchanged. `step4_base_per_episode.csv` and
`step4_collection.csv` in `week6/noise_reachability_step5b_colab/references/` are identical to these files, so step 5b's
references stand. Output: `step4_checks.txt`, from `check_step4_rows.py` and `check_step4_settings.py`.

A1 and A2 span sessions (20 provenance records rather than 16, from resumed parts); the settings are the same
throughout. Adapter weights are not in the zip, so these models cannot be re-evaluated from this record.
