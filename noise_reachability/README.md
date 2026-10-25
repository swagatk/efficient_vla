# Which failures can a frozen SmolVLA fix by itself?

Base policy `lerobot/smolvla_libero@a24f1f3804` on LIBERO-10 through `baseline_performance/evaluate_all.py`
on Kaggle 2 x T4: 171/300 (57.0%) with `n_action_steps=1` (`week6/smolvla_libero_lerobot_eval_kaggle/`), and
216/300 (72.0%) with `n_action_steps=10`, the setting chosen in step 0 (`week6/noise_reachability_step0_kaggle/`).

SmolVLA generates each action chunk by integrating a flow from Gaussian noise. Some failures go away
when only that noise changes; others repeat on the same init states whatever the noise. On task 0 (Colab,
custom harness, 50 init states x noise seeds 1-3) 5 states were solved under all three seeds and 32 under
none; a best-of-3 oracle solves 36% against 20% for one seed, and failures repeat on the same states more
than chance allows (permutation p = 0.0001; `week6/task0_base_noise_seeds_colab/failure_overlap_s1-2-3.json`).

Methods that leave the network frozen and only choose the noise, at test time (RoboMonkey, VGAS, FM-Steer)
or with a learned noise policy (DSRL; "Noise Space Learning for VLA Adaptation", arXiv:2610.13943, which
reports SmolVLA on LIBERO-10 from 50.6% to 95.8% after about 240,000 training rollouts), can only reach
what the frozen generator can produce from some noise. This study measures, per init state, what the
frozen policy can reach (pass@k over noise seeds), and tests whether that predicts where noise-space
methods gain and where a weight update is needed.

Steps run in order; each step's design below is fixed before it runs. Sizes for steps 1-4 are reviewed once
step 0 gives the timing, and any change is written here before that step starts.

## Step 0: action steps per inference (the baseline setting)

**Why.** All our `lerobot/smolvla_libero` evaluations re-plan after every action (`n_action_steps=1`, the
setting LIBERO-CTRL used; `baseline_performance/reference_libero_ctrl/README.md`). The checkpoint predicts
50-action chunks and its config ships `n_action_steps=50`. "Same Scene, Different Answer" (NeurIPS 2026)
reports that re-planning at every step costs pi0.5 9-41 points, and arXiv:2610.13943 executes 10 of 50.
Every later step compares against the base policy, so the setting has to be settled first.

**Hypothesis.** On the four weak tasks (0, 4, 6, 8), executing 10 actions per inference raises success over
`n_action_steps=1` by at least 10 points: from 27/120 (task 0: 1, task 4: 10, task 6: 14, task 8: 2 of 30;
`week6/smolvla_libero_lerobot_eval_kaggle/per_episode.csv`) to at least 39/120.

**Run.** `n_action_steps` = 10 and 50, each on tasks 0, 4, 6, 8 x seeds 0-2 x init states 0-9 (120 episodes),
Kaggle 2 x T4 with `baseline_performance/kaggle_baseline_lerobot_eval.ipynb` (`N_ACTION_STEPS` and `TASKS`
in step 1), everything else as the 171/300 run. `run_step0_action_steps.sh` is the same command for a
Colab or local GPU; reported numbers come from Kaggle only, the machine of the `n_action_steps=1` run.

**Decision rule.** Each setting is compared with `n_action_steps=1` on the same 120 episodes
(`compare_action_steps.py`: paired by task, seed and init state, exact McNemar, Bonferroni alpha = 0.025 per
setting).

- A setting ahead with p < 0.025 qualifies; if both qualify the one with more successes wins, a tie goes to
  10. The winner is then run on all 10 tasks (300 episodes, Kaggle) and becomes the baseline for steps 1-4
  unless it is behind 171/300 with paired p < 0.05 (a gain on weak tasks bought with losses elsewhere).
- Otherwise `n_action_steps=1` stays the setting and its 171/300 stays the baseline.

**Result (October 2026): `n_action_steps=10`, 216/300 (72.0%).** Weak tasks: 57/120 at 10 (+25.0 points,
exact 95% interval +13.6 to +33.3, p < 0.0001) and 37/120 at 50 (p = 0.14, does not qualify). All 10 tasks
at 10: 216/300 against 171/300, p < 0.0001, no task lower. The hypothesis is supported, and the base policy
for steps 1-4 is `lerobot/smolvla_libero` with `n_action_steps=10`. Records and discussion:
`week6/noise_reachability_step0_kaggle/`.

## Step 1: per-state reachability (pass@k)

*Changed after step 0, before step 1 ran.* The first plan took tasks 0, 1, 4, 6, 8, 9: the four tasks then
weakest, and two with room in both directions. At `n_action_steps=10` tasks 1 and 9 succeed in 28/30 and
27/30 episodes, leaving little to measure, and task 7 (18/30) has become one of the weakest. Step 1 therefore
takes the five tasks below 70% at the new baseline: 0, 4, 6, 7, 8 (7, 14, 20, 18 and 15 of 30;
`week6/noise_reachability_step0_kaggle/nas10_all_tasks/`). These are "the weak tasks" from here on.

Base policy only, `n_action_steps=10`: tasks 0, 4, 6, 7, 8 x init states 0-9 x noise seeds 0-15 (800
episodes, Kaggle; seeds 0-2 are reused from `nas10_all_tasks/`), plus a repeat of seed 0 on the same 50 states
to measure run-to-run nondeterminism (step 0 already gave one: 115 of 120 weak-task episodes repeated).

Per (task, init state): successes out of 16, pass@k for k = 1..16 (unbiased estimator), and the class
*reachable* (pass@16 > 0) or *systematic* (pass@16 = 0).

**Hypothesis** (*restated after step 0, before step 1 ran*: the first version, "pass@16 at least twice pass@1
and at least a quarter of states systematic", was written for a 22% baseline; at the new one pass@1 is about
0.49, so the two parts could not both hold). On the weak tasks pooled:

- *headroom* = (pass@16 - pass@1) / (1 - pass@1), the share of single-sample failures that some noise seed
  fixes, is at least 0.5; and
- at least 5 of the 50 init states (10%) are systematic.

Consequences, fixed now: headroom below 0.5 means noise-space methods have little to gain, and steps 2-3 are
reconsidered before they run; fewer than 5 systematic states means step 4 is dropped.

Run-to-run nondeterminism: the repeat of seed 0 gives the share of episodes whose outcome changes with
nothing changed, reported against the share expected to change under a fresh noise seed (2p(1-p) per state,
p its success rate over the 16 seeds). `reachability.py` computes all of this.

## Step 2: training-free noise selection per chunk

At each inference draw N = 8 noise samples in one batch, generate 8 chunks and execute the one a score
picks. Candidate scores, none trained: (a) denoising acceleration, the curvature of the flow trajectory
(arXiv:2607.27933); (b) medoid, the chunk closest to the others (agreement between samples). The score and
N are chosen on init states 10-19 of the step-1 tasks; reported numbers are on init states 0-9, seeds 0-2.

**Hypothesis.** The chosen selector raises success on the weak tasks over the step-0 baseline by at least 10
points (paired exact McNemar p < 0.05), and its gain lands on reachable states: it solves at most 1 in 10 of
the systematic episodes. Selection per chunk can combine chunks that no single sampled trajectory produced,
so systematic states solved here are reported as a finding in their own right.

## Step 3: a learned noise policy (DSRL-style), small budget

A small actor maps the frozen policy's features to the noise fed into the flow, trained with SAC on task
success, network frozen (`vla_core/policy.py` for the base, `rl_based_tuning/` for the SAC code). Two tasks
picked from step 1: the weak task with the most reachable states and the one with the most systematic
states. Train on init states 10-49, report on init states 0-9 x seeds 0-2. Budget at most 3,000 training
episodes per task (Colab A100), far below the roughly 240,000 rollouts of arXiv:2610.13943.

**Hypothesis.** On each task the actor raises success by at least 10 points over the base (paired p < 0.05),
and on systematic states its success stays below a third of its success on reachable states.

## Step 4: a weight update aimed at systematic states

Only if steps 1-3 confirm that the noise-space methods leave the systematic states unsolved. A LoRA update
(`vla_core/lora.py`) trained on successful episodes from neighbouring init states, collected by the base
or the step-3 actor, and evaluated on the systematic states. The data source and size are fixed after step 3.

**Hypothesis.** It raises success on the systematic states of the step-3 tasks by at least 10 points without
lowering success on their reachable states by more than 5.

## Compute

Kaggle 2 x T4 (45 h a week) for base-policy evaluations, Colab A100 (50 h a month) for steps 2-4. At
`n_action_steps=10` the 12 weak-task runs of step 0 (120 episodes) took 32 minutes on Kaggle and all 10 tasks
65 minutes (`week6/noise_reachability_step0_kaggle/`).

| Step | Where | Estimate |
|---|---|---|
| 0 | Kaggle | done: about 2 h over three versions |
| 1 | Kaggle | 700 new episodes (seeds 3-15 and the repeat), roughly 3 h at `n_action_steps=10` |
| 2 | Colab A100 | to be measured; 8 samples per inference in one batch |
| 3 | Colab A100 | 3,000 training episodes x 2 tasks |
| 4 | Colab A100 | after step 3 |

## Files

- `run_step0_action_steps.sh`: step 0 on a Colab or local GPU (`N_ACTION_STEPS`, `OUT_DIR`).
- `compare_action_steps.py`: step-0 comparison and decision against the `n_action_steps=1` run.
- `reachability.py`: step-1 pass@k per init state, headroom, systematic states and the repeat check.

## Status

Step 0 done (above). Step 1 not yet run.

Before step 0 ran, smoke tests on the local RTX 4090 (shared with another job), one episode each of task 0,
seed 0, init state 0: `n_action_steps=10` through `evaluate_all.py` and `n_action_steps=50` through
`run_step0_action_steps.sh` (`SMOKE=1`). Both ran the full 520 steps with the setting applied (it appears in
the policy config printed in `eval.log`) and failed, about 3 minutes each including the model load. They
check that the setting runs, not its success rate.
