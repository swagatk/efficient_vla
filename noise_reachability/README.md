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

**Result (October 2026): both parts hold.** Pooled pass@1 0.471 (377/800), pass@16 0.840, headroom 0.697;
8 of 50 init states systematic (task 0: 5; tasks 4, 6, 8: 1 each; task 7: none). Steps 2-4 go ahead as planned.
The repeat of seed 0 changed 0 of 50 outcomes against 12.4 expected under a fresh seed. Step-3 tasks by the
rule of step 3: 7 (most reachable states) and 0 (most systematic). Records and discussion:
`week6/noise_reachability_step1_kaggle/`.

## Step 2: training-free noise selection per chunk

At each inference draw N = 8 noise samples in one batch, generate 8 chunks and execute the one a score
picks. Candidate scores, none trained: (a) denoising acceleration, the curvature of the flow trajectory
(arXiv:2607.27933); (b) medoid, the chunk closest to the others (agreement between samples). The score and
N are chosen on init states 10-19 of the step-1 tasks; reported numbers are on init states 0-9, seeds 0-2.

**Hypothesis.** The chosen selector raises success on the weak tasks over the step-0 baseline by at least 10
points (paired exact McNemar p < 0.05), and its gain lands on reachable states: it solves at most 1 in 10 of
the systematic episodes. Selection per chunk can combine chunks that no single sampled trajectory produced,
so systematic states solved here are reported as a finding in their own right.

*Protocol, fixed after step 1 and before step 2 ran.*

- **Harness.** `eval_noise_selection.py` runs the base and the selectors through `vla_core` (LeRobot's
  `LiberoEnv` and processors, hard resets), not through lerobot-eval: the selector replaces SmolVLA's
  `sample_actions` on the policy instance (`noise_selection.py`), so it cannot run inside `lerobot-eval`. The
  base is therefore re-run in this harness on the same episodes, with the same environment and torch seed per
  (seed, init state), and "the step-0 baseline" above means this paired base. Its success on init states 0-9 x
  seeds 0-2 is compared with step 0's lerobot-eval run on those episodes (74/150, different noise draws, so not
  paired); a gap of more than 17 episodes (about two standard errors of the difference) means the harness is
  investigated before any selector result is read.
- **Selectors.** Scores in the policy's normalised action space, on the 10 executed actions and the 7 real
  action dimensions: `accel` (summed squared change of the denoising velocity between Euler steps; smallest
  wins) and `medoid` (smallest summed distance to the other samples). `first_n8` executes sample 0 of 8, the
  same computation without selection: a check that batching alone does not change success.
- **Selection stage.** Tasks 0, 4, 6, 7, 8 x init states 10-19 x seeds 0-1 (100 episodes per policy):
  `base`, `first_n8`, `medoid_n8`, `accel_n8`, `medoid_n16`, `accel_n16` (600 episodes). Rule: the
  `accel`/`medoid` policy with the most successes, ties to the smaller N and then to `medoid`. If it is not
  ahead of `base` on these episodes, step 2 is recorded as negative and the report stage does not run.
- **Report stage.** `base` and the chosen policy on tasks 0, 4, 6, 7, 8 x init states 0-9 x seeds 0-2
  (150 paired episodes each). "Systematic episodes" are the 8 step-1 systematic states x 3 seeds (24).
  `analyze_step2.py` computes both stages.
- **Checks before running** (local RTX 4090, shared): with one sample the wrapper returns the base policy's
  chunk bit for bit; 8 batched chunks match 8 single-sample calls with the same noise to 0.011 (bfloat16;
  the spread between samples is 0.35); one inference took 740 ms with 1 sample and 816 ms with 8.

**Result (October 2026): not supported.** The selection stage chose `accel_n8` (55 of 100 against base 52;
`first_n8`, no selection, 54). On the report episodes base 64/150, `accel_n8` 71/150: +4.7 points, exact 95%
interval -3.5 to +11.9, McNemar p = 0.30. Systematic episodes: 0 of 24 for both. Harness check: base 64 here
against 74 through lerobot-eval, within the 17 allowed. Step 3 goes ahead as planned. Records and discussion:
`week6/noise_reachability_step2_colab/`.

## Step 3: a learned noise policy (DSRL-style), small budget

A small actor maps the frozen policy's features to the noise fed into the flow, trained with SAC on task
success, network frozen (`vla_core/policy.py` for the base, `rl_based_tuning/` for the SAC code). Two tasks
picked from step 1: the weak task with the most reachable states and the one with the most systematic
states. Train on init states 10-49, report on init states 0-9 x seeds 0-2. Budget at most 3,000 training
episodes per task (Colab A100), far below the roughly 240,000 rollouts of arXiv:2610.13943.

**Hypothesis.** On each task the actor raises success by at least 10 points over the base (paired p < 0.05),
and on systematic states its success stays below a third of its success on reachable states.

*Protocol, fixed after step 2 and before step 3 ran.* Tasks 7 and 0 (step 1). Systematic states exist only on
task 0 (init states 2, 5, 6, 7, 8), so the second part of the hypothesis is tested on task 0 alone.

- **Decision and action.** One decision per inference, that is per executed chunk of 10 actions. The actor
  chooses the initial noise of the 10 executed positions on the 7 real action dimensions (70 numbers); the
  remaining entries of SmolVLA's 50 x 32 noise are drawn from N(0, 1) as usual. The actor's output is
  `noise_scale * tanh(x)` with `noise_scale = 2.5`, x Gaussian (SAC's squashed Gaussian), and its initial
  spread (std 0.45 before the tanh) makes an untrained actor's noise close to N(0, 1).
- **Observation.** The frozen policy's own prefix features: the VLM output over the image, language and state
  tokens, averaged over valid tokens (960 numbers), plus the first 8 entries of the normalised state. One extra
  prefix pass per inference computes them; the chunk itself comes from SmolVLA's unchanged `sample_actions`
  with the chosen noise. Nothing in SmolVLA trains.
- **Reward and MDP.** Reward 1 on the decision whose chunk reaches success, which ends the episode (terminal);
  0 otherwise. Discount 0.99 per decision. At the 520-step limit the last transition bootstraps from the
  final observation (not terminal).
- **SAC.** Twin Q and a tanh-Gaussian actor, MLPs of 3 x 256 with layer-normalised inputs, Adam 3e-4, batch
  256, tau 0.005, automatic entropy temperature starting at 0.01 with target entropy -70, 8 gradient updates per
  decision collected, done after each episode once the replay holds a batch. Replay holds every transition of
  the run. The first 20 episodes are warm-up: the executed noise is
  the base's own N(0, 1) draw clipped to +-0.999 * noise_scale, and only the critics train.
- **Training.** Init states 10-49, uniformly at random per episode; soft resets (`vla_core/envs.py`);
  `n_action_steps=10`; 1,500 episodes per task, one training seed, the two tasks in parallel on one A100.
  Training runs in processes of 25 episodes; each saves the learner, the log and its own new transitions as
  a replay part (parts are never rewritten), and the next process resumes. A crashed process is retried from
  the last save, at most 3 times in a row. These settings are not tuned: no run's result changes them.
- **Report.** The final actor, deterministic (`noise_scale * tanh(mean)`), and the base, on init states 0-9 x
  seeds 0-2 (30 paired episodes per task), hard resets, same environment and torch seed per (seed, init
  state). Exact McNemar and the exact 95% interval of the paired difference per task. Task 0 systematic
  episodes: init states 2, 5, 6, 7, 8 x seeds 0-2 (15), reachable: the other 15. "Below a third" is strict,
  so if the actor solves no reachable episode the second part counts as not met.
- **Checks before running.** CPU tests: the chosen noise replaces exactly the executed part of the drawn
  noise; the training loop, with a fake environment and policy, stores the transitions it executed (features,
  noise, reward, bootstrap) across a resume. On the real model (`check_step3_steering.py`, the notebook's step
  5): the wrapper with no chosen noise reproduces the base chunk bit for bit, passing the drawn noise back
  unchanged does too, and a fixed chosen noise gives the same chunk twice. Then a smoke run of 3 training
  episodes and 1 evaluation episode per policy through the runner (step 6).

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
| 1 | Kaggle | done: 170 min (seeds 3-15) and 16 min (repeat), plus 9.6 h lost to a version at the wrong setting |
| 2 | Colab A100 | done: 600 (selection) + 300 (report) episodes, one process per task in parallel; the last processes started 82 and 37 minutes after launch |
| 3 | Colab A100 | 1,500 training episodes x 2 tasks in parallel, then 120 evaluation episodes; estimated 20-30 h wall clock from the step-2 episode times (not measured yet) |
| 4 | Colab A100 | after step 3 |

## Files

- `run_step0_action_steps.sh`: step 0 on a Colab or local GPU (`N_ACTION_STEPS`, `OUT_DIR`).
- `compare_action_steps.py`: step-0 comparison and decision against the `n_action_steps=1` run.
- `reachability.py`: step-1 pass@k per init state, headroom, systematic states and the repeat check.
- `noise_selection.py`: step-2 selectors (scores, and the wrapper around SmolVLA's `sample_actions`).
- `eval_noise_selection.py`, `run_step2_noise_selection.sh`: step-2 paired evaluation (`STAGE=select|report`,
  `SMOKE=1`); `colab_step2_noise_selection.ipynb` runs it on Colab.
- `analyze_step2.py`: step-2 selection rule and report.
- `noise_sac.py` (actor, critics, replay, transitions), `noise_steering.py` (the wrapper that lets the actor
  choose the noise), `train_noise_sac.py` and `run_step3_noise_sac.sh`: step-3 training and paired evaluation;
  `check_step3_steering.py`: the real-model checks; `colab_step3_noise_sac.ipynb` runs it all on Colab;
  `analyze_step3.py`: step-3 report.

## Status

Steps 0, 1 and 2 done (above); step 2 is a negative result. Step 3 code written and checked on CPU (above); the
real-model checks and the smoke run run on Colab before training (the local GPU was busy); not yet run.

Before step 0 ran, smoke tests on the local RTX 4090 (shared with another job), one episode each of task 0,
seed 0, init state 0: `n_action_steps=10` through `evaluate_all.py` and `n_action_steps=50` through
`run_step0_action_steps.sh` (`SMOKE=1`). Both ran the full 520 steps with the setting applied (it appears in
the policy config printed in `eval.log`) and failed, about 3 minutes each including the model load. They
check that the setting runs, not its success rate.
