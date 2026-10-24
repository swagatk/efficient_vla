# Task 0 base policy, local GPU, noise seed 1

The planned check from `hybrid_diffusion/README.md` ("Result (task 0, Colab A100)"): the local seed-0 run
solved 8 of the 32 init states that the three Colab runs never solved. Is that machine dependence? This run
repeats Colab seed 1 on the local RTX 4090 laptop GPU and compares the two, paired by init state.

- Command: `OUT_ROOT=hybrid_diffusion/outputs/task0_local_seed1 SEEDS=1 bash hybrid_diffusion/run_base_noise_seeds.sh`
  through `tools/logged_run.sh`, at commit `643773a` with a clean tree (`provenance_base_noise_seeds.txt`).
  The eval config equals Colab seed 1's except for the output folder.
- Interrupted: WSL was shut down at 21:41 on 2026-10-23, while Windows went in and out of standby, after 18
  of the 50 episodes. The re-run at 22:10 reloaded those 18 records and ran init states 18-49
  (`eval.log`, `runs_log_excerpt.txt`). The final `failure_overlap.py` step exits with an error because it
  needs at least two seeds (`runner_resume.log`); the evaluation itself completed.
- Hardware caveat: the laptop has a CPU fault (corrected machine-check errors, about 8% of runs of a
  deterministic Python test failed on 2026-10-16) and is waiting for a motherboard replacement.

## Result: no evidence of machine dependence

From `comparison.txt` (`compare_local_colab.py`):

| Comparison, paired by init state | Only first | Only second | Both | Exact McNemar |
|---|---|---|---|---|
| Local seed 1 (14/50) vs Colab seed 1 (13/50) | 5 | 4 | 9 | p = 1.0 |
| Colab seed 2 vs Colab seed 1, for scale | 3 | 7 | 6 | p = 0.34 |
| Local seed 1 vs local seed 0, for scale | 5 | 7 | 9 | p = 0.77 |

- With the same noise seed, the two machines disagree on 9 init states, about as many as two seeds on one
  machine. Rollouts are not bit-reproducible, so part of this is run-to-run nondeterminism.
- Success rates: local 30/100 (seeds 0 and 1) against Colab 30/150 (seeds 1-3), Fisher p = 0.096; local
  seed 1 alone, 14/50 against 30/150, p = 0.24.
- Local seed 1 solves 3 of the 32 Colab-never-solved states (2, 22, 37; 2 and 37 were also among local
  seed 0's 8). The Fisher test against 0/96 (p = 0.015 here, 7.4e-6 for seed 0) is biased: the 32 states
  were selected because Colab failed on them, so Colab's 0/96 holds by construction and any new run, on any
  machine, will solve some of them. The earlier 7.4e-6 overstated the evidence.

The run was on the faulty CPU, but it agrees with Colab, and the Kaggle `lerobot-eval` rerun (e53cb36) also
showed no machine dependence. A repeat after the motherboard replacement would remove the caveat.

## Files

- `eval_results.json`, `eval_summary.csv`, `eval_config.json`, `eval.log`: from
  `hybrid_diffusion/outputs/task0_local_seed1/eval_base_s1/`.
- `provenance_base_noise_seeds.txt`: both launches.
- `runner_resume.log`, `runs_log_excerpt.txt`: runner output of the re-run and the crash-monitor run log.
  The first launch's runner output was in a temporary folder that was lost when WSL shut down.
- `compare_local_colab.py` -> `comparison.txt`.
