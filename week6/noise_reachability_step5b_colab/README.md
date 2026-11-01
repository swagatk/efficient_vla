# Step 5b: a learned noise actor on held-out states (Colab A100, 2026-10-31/11-01)

Protocol: `noise_reachability/README.md`, step 5b, fixed before the run. Commit `0ea0015` (clean tree in all 643 run
provenance records, both sessions), `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, task 0, 37 epochs, 3
training seeds on init states 10-49, evaluation on init states 0-9 x seeds 16-31. Source: `step5b_results.zip`, sha256
`da529b34da26b1759f90e8c8e572988cc8c176905867940f924814502c551170`. `references/` holds the numbers fixed before the run.

**Result under the rule fixed in advance: no reading on transfer.** The learner did not learn at this budget, so step 5b
says nothing either way about transfer to held-out states. No seed meets the systematic hypothesis, and only seed 0 beat
the untrained actor's training rate by the 8 points required.

Harness: base 43/160 against step 4's 37 (gap +6, limit 16). Gate after seed 0: 48.0% against the 30% floor.

| Arm | Systematic /80 | Net vs base | p | Net vs prior | p | Reachable /80 | Net vs base | p | Net vs prior | p | Training, last 5 epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0 | | | | | 43 | | | | | |
| Prior | 0 | | | | | 46 | | | | | |
| Actor, seed 0 | 2 | +2 | 0.5 | +2 | 0.5 | 56 | +13 | 0.0106 | +10 | 0.0213 | 48.0% |
| Actor, seed 1 | 2 | +2 | 0.5 | +2 | 0.5 | 48 | +5 | 0.332 | +2 | 0.804 | 40.0% |
| Actor, seed 2 | 5 | +5 | 0.0625 | +5 | 0.0625 | 47 | +4 | 0.541 | +1 | 1 | 42.5% |

The untrained actor's training rate is 37.5% (45/120, epoch 0 of the three seeds), so the gains are 10.5, 2.5 and 5.0
points against 8 required in at least 2 seeds. No arm failed the reachable guard. Secondary, one outcome per state, not
tested: `actor_mean` 3 of 10 in each seed, `zero` 3 of 10.

Successes per init state, of 16 (systematic states 2, 5, 6, 7, 8):

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 0 | 16 | 0 | 16 | 5 | 0 | 0 | 0 | 0 | 6 |
| Prior | 4 | 16 | 0 | 13 | 6 | 0 | 0 | 0 | 0 | 7 |
| Actor 0 | 4 | 16 | 0 | 15 | 8 | 0 | 1 | 0 | 1 | 13 |
| Actor 1 | 3 | 15 | 0 | 15 | 7 | 0 | 0 | 0 | 2 | 8 |
| Actor 2 | 4 | 13 | 0 | 15 | 5 | 0 | 1 | 0 | 4 | 10 |

**Read with care (exploratory, not part of the rule).**

- Seed 0 is the one seed with a clear gain on held-out reachable states, against both base and prior, but the rule asks for
  2 seeds, and a run of three training seeds cannot tell a real effect from a good seed.
- All 9 systematic successes are on states 6 and 8. State 8 was 0 in every arm of step 4 and is 1, 2, 4 here; states 2, 5
  and 7 are 0 in every arm, as in step 4. The best case, 4 of 16 against 0 of 16, is not significant alone (sign test
  p 0.125, picked after seeing the data).
- State 0 is 0 for the base and 4 for the prior: repeating one noise along the chunk moves it with no learning.
- The actor's distribution moved (update log, seeds 0-2: `mean_abs` from about 0.04 to 0.17, `std_mean` from about 0.99
  to 0.80), so training changed the actor; the rule measures whether success rose, and mostly it did not.

**Checks.** `analyze_step5b.py` rerun locally reproduces the Colab report and its JSON byte for byte; separately recomputed
counts, nets, p-values and training gains in `check_step5b_rows.py` agree (`step5b_checks.txt`). `tools/validate_run.py`
reports one failure on each of the 4 eval folders and no other: `per_episode_zero.csv` or `per_episode_actor_mean.csv` has
10 of 160 episodes, by design, since those arms run seed 16 only. All other files pass, and the row check shows every arm
with exactly its expected distinct episodes. It finds no logs in `eval/`. All 585 logs were scanned by hand: one traceback.
Training has 1,480 distinct rows per seed, 40 per epoch, 37 update rows per seed. The base differs from step 4's on 16 of
the same 160 episodes (Kaggle T4 against Colab A100), within the harness.

**The session was lost and resumed.** The first session died at seed 2, epoch 23 (97 of 111 epochs, 330 of 840 evaluation
episodes), after the last log at 07:33, on 2026-11-01. The resume started at 11:00 at the same commit, on the same GPU
type. The traceback is `seed2_e023_part2.log`, `FileNotFoundError` when saving to Drive, consistent with the first session ending (the logs have no
time for it); it lost no result. Epoch 23 was collected again from `actor_e022.pt`, and all 31 episodes logged by the first
session in that epoch came out the same (success and steps) when repeated, as in the rows that entered `train_log.csv`.
So the resume repeated those episodes exactly. Only seed 2 from epoch 23 on and the evaluation of the actors ran in the second session.
