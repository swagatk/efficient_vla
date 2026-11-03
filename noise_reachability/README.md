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

**Result (October 2026): not a test of the hypothesis; SAC's critic diverged.** Q passed 100 by episode
100 on both tasks (rewards are 0 or 1) and ended at 1.0e11 (task 7) and 4.0e12 (task 0), with the
automatic temperature at 5.2e6 and 8.7e7. As measured, the final actors against the paired base: task 7
11/30 vs 18/30 (McNemar p = 0.092), task 0 8/30 vs 5/30 (p = 0.25), task 0 systematic episodes 0/15 for
both. Offline re-runs on the recorded task-7 replay reproduce the divergence; a critic regressed on Monte
Carlo returns is stable and predicts success (held-out AUC 0.89), but no critic separated noises clearly,
and the run's data cannot test that cleanly because the stored noise depends on the state. Records,
diagnosis and offline runs: `week6/noise_reachability_step3_colab/`.

## Step 3b: is there a noise signal a critic can learn?

Step 3's data cannot say whether the executed noise affects success in a way a critic can learn from the
policy's features: after warm-up its noise came from the actor, a function of the state. Here the noise is
the base policy's own N(0, 1) draw, independent of the state, so any held-out link between noise and success
is the noise's effect. No RL: base rollouts are logged, then a critic is fitted offline.

**Hypothesis.** On task 7 a critic regressed on Monte Carlo returns ranks held-out transitions of successful
episodes above failed ones by the noise's part of its value with AUC at least 0.55, and the 95% interval's
lower end is above 0.5. (0.55 is a judgement of the smallest signal worth a causal test, not derived.) If
so, the next test is causal: a critic-guided choice among sampled noises against the paired base on init
states 0-9 x seeds 0-2. If not, noise-space learning on task 7 has no detectable signal at this data size,
and step 4 follows.

*Protocol, fixed before the run.*

- **Data.** `lerobot/smolvla_libero` at revision `a24f1f3804` (as steps 0-1), `n_action_steps=10`, 520
  steps, the camera rename map, soft resets; task 7, init states 10-49 drawn per episode as in step 3 (none
  of the report states 0-9). 4 collection streams (`--seed` 0-3) x 250 episodes = 1,000 episodes, through
  `train_noise_sac.py` with every episode a warm-up episode and no updates (`--warmup_episodes 250
  --num_episodes 250 --updates_per_decision 0`). Per inference it stores the features (step 3's 968) and
  the executed noise of the 70 executed entries, scaled by `--noise_scale 5`: the clip at 0.999 x 5 touches
  only draws beyond 4.995 (about 2 of the run's roughly 3 million entries, expected), and the others equal
  the base's draw up to float rounding. Kaggle 2 x T4, two streams per GPU.
- **Split.** Episodes whose index in their stream is 4 mod 5 are held out (200 episodes); the critics train on
  the other 800. Nothing is tuned on the held-out episodes.
- **Critic.** Step 3's twin critic (3 x 256, layer-normalised input), min of the two heads, regressed by MSE
  on the Monte Carlo return of each decision (0.99^(decisions to the end) if the episode succeeded, else 0),
  Adam 3e-4, batch 256, as many updates as training transitions; 3 critics with seeds 0-2, averaged.
- **Measure.** For a held-out transition, the noise's part of its value is Q(s, a) minus the mean of Q(s, a_k)
  over 8 noises a_k drawn like the base's. Action AUC: the probability that this score is higher for a
  transition of a successful episode than of a failed one. 95% interval: percentile bootstrap over held-out
  episodes (2,000 resamples). Also reported: success AUC of Q(s, a) itself (whether the critic learned the
  state value at all), and a control with the noises shuffled across training transitions, which should give
  an action AUC near 0.5; if the control's interval excludes 0.5 the measure is biased and the result is not
  read.
- **Checks before running.** CPU tests of the analysis on synthetic data with and without a noise effect;
  on 10 synthetic datasets without one, 1 of the 20 intervals (critic and control) excluded 0.5
  (`calibration/step3b_null_calibration.json`; small critics on 4 features, so a sanity check of the
  measure, not an estimate of its error rate in the real run). The collection path is step 3's warm-up; on Kaggle the
  notebook repeats `check_step3_steering.py` (no chosen noise reproduces the base chunk bit for bit) and a
  2-episode smoke run through the runner before launch (the local GPU was busy).

**Result (2026-10-29): not read.** The shuffled-noise control's action AUC was 0.547 (95% interval 0.532 to
0.562), so its interval excludes 0.5 and the measure is biased on this data; the critic's action AUC was 0.566
(0.549 to 0.583), success AUC 0.930. A local re-run of the analysis on the same files and commit gave 0.570 and
0.557 (not bit-identical across GPUs), with the same decision. Under the protocol, step 4 follows. Records and
checks: `week6/noise_reachability_step3b_kaggle/`.

## Step 4: a weight update aimed at systematic states

Only if steps 1-3 confirm that the noise-space methods leave the systematic states unsolved. A LoRA update
(`vla_core/lora.py`) trained on successful episodes from neighbouring init states, collected by the base
or the step-3 actor, and evaluated on the systematic states. The data source and size are fixed after step 3.

**Hypothesis.** It raises success on the systematic states of the step-3 tasks by at least 10 points without
lowering success on their reachable states by more than 5.

*Design after step 3b, fixed 2026-10-29 before any step-4 code or run (reviewed: arm B's checkpoint is on
Drive; arm C added; 16 evaluation seeds per state; the step-2 harness tolerance).*

**How far the condition holds.** It holds only weakly. No noise-space method solved a systematic episode
(step 2: 0 of 24; step 3: 0 of 15 on task 0), but none raised success clearly on reachable states either
(step 2: +4.7 points, p = 0.30; step 3 was not a test). So steps 2-3 do not show that noise-space methods
*can* gain where step 1 says they could; they only show that they did not reach the systematic states. Step 4
goes ahead on that basis, and the write-up has to say so.

**What changes from the plan above, and why.**

- *Task 0 only.* Step 3 picked tasks 7 and 0, but task 7 has no systematic state (step 1). Task 0 has 5
  (init states 2, 5, 6, 7, 8) and 5 reachable ones (0, 1, 3, 4, 9; 41/80 successes over noise seeds 0-15).
- *No "neighbouring" init states.* The plan never defined a distance between init states, and choosing one
  now would be a free parameter. Training data comes from init states 10-49, as in steps 2-3.
- *Two data sources instead of one, because of an earlier result.* Self-filtered BC on task 6
  (`week6/self_filtered_bc_task6/`, `n_action_steps=1`) cloned the base's own successful episodes with
  this LoRA setup: pooled 46/90 against the base's 45/90, and init state 8, which the base never solved,
  stayed at 0/9. That predicts that the base's own successes do not fix systematic states. So step 4 also
  tests a source of behaviour from outside the base's rollouts: the LIBERO-10 demonstrations, through the
  same LoRA update (arm C) and through the existing continued-training checkpoint (arm B).

**Question.** Can a weight update solve task 0's systematic states, and does that need behaviour from
outside the base policy's own rollouts?

**Arms.**

- *Base:* `lerobot/smolvla_libero@a24f1f3804`, re-run in this harness on every evaluation episode.
- *A, self-filtered LoRA:* `self_filtered_bc/self_filtered_bc.py` with `--n_action_steps 10`, otherwise the
  task-6 settings unchanged (LoRA r16/alpha32 on the action expert's seven projections, 2,000 steps, batch
  16, lr 1e-4, only LoRA trains). Data: the base on task 0, init states 10-49 x collection seeds 0-2 (120
  episodes, soft resets), successful episodes only; init states 45-49 held out for the validation loss.
  Training seeds 0-2.
- *C, demonstration LoRA:* the same training as A (same code, settings, step budget and training seeds), on
  the task-0 demonstrations of `lerobot/libero@a1aaacb7f6` (the data `continued_training/` used) instead of the
  base's rollouts; every fifth demonstration held out for the validation loss. A against C changes only the
  data source, which is the comparison the main claim rests on.
- *B, continued training:* the existing checkpoint of `continued_training/` (the full model, 25,000 more steps
  on the 379 LIBERO-10 demonstrations, one training seed; `week6/continued_training_colab/`), reused as it is.
  It was evaluated only at `n_action_steps=1` (174/300 against 171/300; task 0: 0/30 against 1/30). It
  differs from C in several ways at once (all weights, all ten tasks, 12x the steps, one seed), so it is a
  secondary arm: it shows what a larger demonstration update does, not why.
- *What "outside the base's rollouts" means.* The 33 demonstrations of arms B and C were part of the data the
  base itself was trained on (its loss on them is 0.04, below). So C adds no behaviour the base never saw in
  training; it re-weights the base toward task 0's demonstrated behaviour, against A, which re-weights it
  toward its own successful closed-loop behaviour. "Neither A nor C" is therefore weak evidence about genuinely
  new behaviour, and the write-up says so.

**Evaluation.** Task 0, init states 0-9 x evaluation seeds 16-31 (160 episodes per policy and training seed:
base, 3 x A, 3 x C, B), `n_action_steps=10`, 520 steps, hard resets, the same environment and torch seed per
(seed, init state) for every policy. Seeds 0-15 chose the systematic states, so on them the base is 0 by
selection. Fresh seeds give an unbiased base rate: a state that was 0/16 can succeed by chance.

- Systematic episodes: init states 2, 5, 6, 7, 8 x 16 seeds (80). Reachable: the other 80.
- Exact McNemar per arm and training seed against the base on the same episodes, plus the exact 95% interval
  of the paired difference. Arms A and C are also pooled over their 3 training seeds. The base episodes appear
  once per training seed in that pooling, so the pooled test overstates the evidence (as on task 6); the
  per-seed tests are the primary ones.
- A against C directly: per training seed (same seed, same episodes), exact McNemar on the systematic episodes.

**Hypothesis, per arm.**

- *Systematic:* at least 10 points above the base on the 80 systematic episodes (episodes solved only by the
  arm minus those solved only by the base: at least 8), with paired p < 0.05.
- *Reachable, guard:* on the 80 reachable episodes, a point estimate no more than 5 points below the base
  (at most 4 episodes net), and not significantly below it (paired p >= 0.05). With 80 episodes this guard
  only catches large losses.
- An arm meets the hypothesis in a training seed when both parts hold in that seed; A and C meet it when that
  is so in at least 2 of their 3 training seeds. (*Wording clarified 2026-10-29 before any run, after the code
  review: the first version put "2 of 3 seeds" under the systematic part only; `analyze_step4.py` already
  counted seeds where both parts hold.*)

**Reading the outcome, fixed now.**

- *C meets it, A does not, and C is ahead of A on the systematic episodes (paired p < 0.05 in at least 2 of 3
  seeds):* systematic failures need behaviour the base does not produce; imitating the base's own successes
  with the same update is not enough. Together with steps 2-3, this is the study's main claim.
- *C meets it, A does not, but C is not significantly ahead of A:* consistent with the claim, not a
  demonstration of it; reported as such.
- *A meets it:* contradicts the task-6 result. Re-check the harness and the data first (CLAUDE.md), then it is
  a finding in its own right.
- *Neither A nor C meets it:* at this budget no LoRA update tested solves the systematic states; B is read as
  the larger update. If B also fails, step 4 is a negative result and the study is written up as the step 0-3b
  measurements with that limit stated.
- *An arm fails the guard:* its losses on reachable states are reported as the price of any gain, not hidden
  in the total.

**Checks before running.**

- Harness: the base on task 0, init states 0-9 x seeds 0-2 through this eval path, against step 0's 7/30 at
  `n_action_steps=10` (`week6/noise_reachability_step0_kaggle/`). The noise draws differ, so the episodes are
  not paired. As in step 2, a gap of more than two standard errors of the difference
  (2 x sqrt(2 x 0.233 x 0.767 / 30) x 30 = 6.6 episodes), that is 7 or more episodes either way, means the
  harness is investigated before any arm is read.
- Demonstrations: the task-0 episodes are found by their task string, not by index (LIBERO-10 task 0 is the
  dataset's task index 5: 33 episodes among 0-378, 245-367 frames each); their state and actions
  are in the units the policy uses (checked against `meta/stats.json`, as in `continued_training/`); and the
  base policy's flow-matching loss on demonstration frames is below 0.3 (under 3x its 0.112 on this data at
  the start of continued training), with both images rotated by 180 degrees giving at least twice that
  (`check_step4.py`). *Changed on 2026-10-29 before the check ran:* the first version compared the
  demonstration loss with the loss on the base's own rollouts, but rollout targets span several replans,
  so the two are not comparable; the rollout loss is printed for reference only.
- CPU tests for the step-4 analysis (systematic/reachable split, McNemar, pooling, decision) and the
  demonstration converter, then a mutation check.
- Real model: a fresh LoRA adapter reproduces the base chunk bit for bit at `n_action_steps=10`; arm B's
  checkpoint loads in this harness, applies `n_action_steps=10`, and runs an episode. (It cannot be matched
  episode by episode with its Kaggle evaluation, which ran through lerobot-eval with different noise draws.)
- A smoke run of every stage (2 collection episodes, a few training steps, 1 evaluation episode per policy).

*Checks run so far (2026-10-29, local RTX 4090 shared with another job; `outputs/step4_checks/`, not in git).*

- Demonstrations: the first 2 task-0 demonstrations (dataset episodes 8 and 13) converted through LeRobot's
  loader (pyav): stored actions and states equal the dataset's columns exactly, images pass the exact-uint8
  check, and the image orientation matches a base rollout frame (checked by eye).
- `check_step4.py`: base loss 0.0373 on the 273 demonstration frames (limit 0.3); rotated 1.6947 (45x, limit
  2x); a fresh LoRA reproduces the base chunk bit for bit at `n_action_steps=10` (9 queued actions, max
  difference 0) and nudged LoRA weights change it (0.087). All passed.
- For reference, the base's loss on its own task-0 rollout frames is 1.2335 (2 episodes, 395 frames). The
  review asked whether that comes from replanning or from a recording mismatch, so it was measured: at the
  frames where the base replanned (steps 0, 10, 20, ...), counting only the first 10 target actions (the
  chunk it executed), the loss is 0.0333 on 79 rollout frames, against 0.0444 on 55 demonstration frames under
  the same masking. Rollouts are recorded consistently; the 1.23 comes from targets after the first 10, which
  later replans produced. Arm A therefore trains toward closed-loop sequences the base does not predict as one
  chunk, arm C toward sequences it already fits; that difference is part of what "data source" means here and
  is reported with the result.
- Collection at step 4's settings (`--n_action_steps 10`, task 0, init states 10-11, seed 0): 1 of 2
  succeeded (269 steps, 38 s; the failure ran 520 steps, 65 s).
- Train and eval stages, smoke run on 2 converted demonstrations: 5 training steps (1 demonstration
  trained, 1 held out by `--val_demo_every 2`; 0.5 s per step, 4.2 GB peak GPU memory), then base and LoRA
  on init state 2, seed 16 (both failed; CSVs and provenance written).
- `check_step4.py --b_policy_path` exercised with the base standing in for arm B: the queue check passed
  and "arm B's chunk differs from the base's" failed (max difference 0), as it must for identical weights.
- The notebook's control flow, simulated locally with a fake runner in place of the simulator: a full run
  (checks, harness, collection merge, 6 adapters, 8 x 160 episodes, then `analyze_step4.py`); a stop at the
  deadline during evaluation and a resumed session, after which every evaluation part held exactly its 40
  episodes, none repeated. A harness result outside the tolerance stopped the run.
- Mutation checks (`tools/mutation_check.py`): `analyze_step4.py` 61 of 74 mutants killed; the 13 survivors
  cannot change a decision (an exact McNemar p never equals 0.05, a net of 0 has p = 1 and a net of 1 is never
  significant; report formatting, argparse and JSON details). `demo_episodes.py` (pure functions) 12 of 13,
  then a test for the episode-378 boundary; the train-split changes in `self_filtered_bc.py` 11 of 13, then a
  test for `--val_demo_every 1`.
- Independent review (fresh-context `research-code-reviewer`, 2026-10-29): no blockers. Acted on: the
  rollout-loss question above; T4 training speed and the bf16 backward pass, now measured by a 20-step run in
  the notebook's checks (which also sets the time a training run needs before it may start); the converter
  held every decoded frame in memory (about 15 GB for 33 demonstrations, by arithmetic) and now holds one
  episode at a time (outputs byte-identical, 2.6 GB peak for 2 episodes); the "2 of 3 seeds" wording; the
  interpretation of arm C; the analysis JSON now records the commit and a SHA-256 of every input; the
  notebook clones a pinned commit. Not changed: up to 10 hard-reset episodes per process (FAIL-004's limit of
  3 was seen on WSL2; steps 2, 3 and 3b ran 10 per process on Colab and Kaggle; a crash fails the job loudly).
- Not yet run: arm B's real checkpoint (on Drive; the notebook checks it before anything else), the harness
  check, T4 training speed and the full 33-demonstration conversion on Kaggle.

**Compute (estimates, not measured).** Episode time from step 3b: about 57 s on a T4 with two streams per
GPU, longer for task 0, whose episodes mostly fail and run all 520 steps. Collection: 120 episodes. Training:
6 x 2,000 LoRA steps. Evaluation: 8 x 160 = 1,280 episodes, roughly 5-7 h on Kaggle 2 x T4 at 60-80 s per
episode over 4 streams. Probably two Kaggle sessions; arm B's checkpoint (on Drive) has to be uploaded as a
private Kaggle dataset first.

**Result (Kaggle 2 x T4, 2026-10-29/31, commit `9318dd5`): negative under the rule above.** Neither A nor C meets the
hypothesis in any training seed, and B also fails, so step 4 is a negative result and the study is written up as the
step 0-3b measurements with that limit stated. Of 80 systematic episodes, against 0 for the base: A 1, 2, 3; C 0, 0, 0;
B 2 (best net +3, required +8). Of 80 reachable, against 37: A 48, 38, 47; C 40, 41, 40; B 62. Harness 6/30 against 7.
Exploratory: all 8 systematic successes are on state 6, and arm C's loss started low and did not fall, so C does not
show that demonstrations cannot help. Records and checks: `week6/noise_reachability_step4_kaggle/`.

## Step 5: how deep is a systematic state? (design 2026-10-30)

Steps 2-4 cannot say whether noise can solve a systematic state, because no step tried hard enough: step 1 drew 16
seeds per state, and steps 2 and 3 were not tests of depth (step 3 was not a test at all). Two papers point opposite
ways. arXiv:2610.13943 reports a learned noise actor taking a SmolVLA from 50.6% to 95.8% on LIBERO-10 after about
240,000 rollouts (60 epochs x 4,000; its Table 1, best of 60 evaluations, one seed listed), on the resets it trained
on, so at most 21 of 500 resets failed in its best evaluation; SimpleVLA-RL (ICLR 2026, weights
of OpenVLA-OFT) reports that RL fails at 0% success and barely helps below 5%, per task. Neither reports per state.
Step 5 measures it per state, on task 0, with the VLA's weights unchanged.

**Pilot (before 5a).** On the A100: 5 episodes per process of state 2 (all 520 steps) at 2, 4, 6, 8 processes, measured
as episodes over wall time, model loads included. The run uses the count reached while each step up gains at least 15%. Not results; noise ids
above 9000 keep these episodes apart from the search.

### Step 5a: deep noise search

**Hypothesis.** Written before any 5a episode: the search reaches at least one of the five systematic states. State 6
is the likeliest, an expectation informed by step 4, where weight-updated policies reached it and the base did not
(interim counts, not yet in the repo); that says nothing about noise. If no family reaches a state in 210 episodes, the
one-sided 95% bound on that state's success rate is 1.4% for the equal mix of the three families (4.2% for one family
on its own, and about 2.2% for all five states jointly), and "systematic" stands as: unreached by these noises at this
depth, not unreachable.

**Protocol (fixed before the run).** `lerobot/smolvla_libero` at `a24f1f3804` (a snapshot at that revision is passed to
every stage), `n_action_steps=10`, 520 steps, hard resets, task 0.

- *Families.* `sampled`: the base, a fresh noise per inference. `full`: one fixed 50 x 32 noise used at every
  inference of the episode (the form of "Golden Ticket", arXiv:2603.15757). `repeat`: one fixed 32-vector repeated along
  the chunk, used at every inference (the form of arXiv:2610.13943, whose actor chooses it per inference; this
  search does not). Fixed noises come from their own CPU generator, the same on every machine.
- *Episodes.* Noise ids 0-69 per family on systematic states 2, 5, 6, 7, 8 (210 per state, 1,050). Noise ids 0-34 per
  family on control state 9 (105). Episode seed 1000 + noise id, which overlaps no seed of step 1 (0-15), step 4 (16-31)
  or the harness (0-2). With a fixed noise the environment seed changes nothing once the init state is set
  (`check_step5a.py`), so fixed-noise episodes are deterministic.
- *Confirmation.* Because of that, a rerun would replay a hit. Instead the first 3 hits of each read fixed family on
  each systematic state are run with 4 jittered copies of the noise (jitter seeds 5000-5003, Gaussian std 0.3 in the
  family's form; at most 120 episodes). This asks whether a hit sits in a region of noise or on an isolated point. A
  sampled hit has no noise to perturb and stands as a hit.
- *Harness and control, run first; each stops the run if it fails.* The base on init states 0-9 x seeds 0-2 against
  step 0's 7/30, gap at most 6 (30 episodes, through `eval_noise_selection.py`). Then control state 9: `sampled` under 3
  of 35 stops the run. A fixed family under 3 of 35 is not read, and any hit it has is reported apart: it happened,
  but the family is not used in the classes. At most 1,305 episodes in all, plus the pilot.

**Decision.** Per state: *robust* (a fixed noise succeeds in at least 2 of 4 jittered copies), else *sampled* (a sampled
hit), else *fragile* (only fixed hits, not held up), else *unreached*. A fragile state counts as reached: a success
happened. Overall: no state reached, "unreached at this depth"; 1-2 states, "mixed"; 3 or more, "mostly reachable",
and "systematic" then means rare.

**Post-run validation.** `tools/validate_run.py` expects cross-product layouts, so `analyze_step5a.py` does this
itself: it refuses a missing or repeated episode, more than one commit, a dirty tree, other settings, or more than one
policy path across the provenance records.

**What it cannot show.** A fixed noise across an episode is not a sequence of noises chosen per inference, so
"unreached" here says nothing about a learned actor; that is step 5b. Everything is task 0.

**Result (2026-10-31, `week6/noise_reachability_step5a_colab/`).** "Mostly reachable": all 5 states reached, so the
hypothesis holds and "systematic" means rare. States 6 and 8 are robust; 2, 5 and 7 fragile. Of 350 episodes per
family: `sampled` 0, `full` 5, `repeat` 20. Harness 5/30 against 7; control sampled 8, full 12, repeat 11 of 35. Fixed
noises were chosen after seeing the outcome, so this shows that such noises exist, not that a policy can find them.

### Step 5b: a learned noise actor on held-out states (design 2026-10-31)

Step 5a showed that fixed noises reach all five systematic states, and that they were picked after seeing the outcome.
Step 5b asks whether an actor that never saw those states can find such noises from the observation.

**Hypothesis.** Written before any 5b code: NSL-AWR after arXiv:2610.13943, trained on init states 10-49 only, does not
raise success on the held-out systematic states. In full, it fails the rule below in at least 2 of 3 training seeds,
because AWR learns only from successes, and these states are unlike the states that supply them. Being wrong is the
interesting outcome: it would mean that a noise actor transfers to failures it has never seen solved.

**What a null cannot show.** The budget is about 1,500 rollouts per training seed against about 24,000 per task in that
paper, and no init state of 0-9 is trained on. A null does not bound what such an actor can do with more data or
with these states in training; the paper's own result used its evaluation resets in training.

**Protocol (fixed before the code and the pilot).** `lerobot/smolvla_libero` at `a24f1f3804`, `n_action_steps=10`, 520
steps, hard resets, task 0, VLA frozen.

- *Actor.* From that paper, Appendix B.1: pooled VLA features plus the state as input (`noise_steering.py`'s 960 + 8
  numbers), four hidden layers of 512 with LayerNorm and ReLU, zero-initialised 32-dimensional mean and log-std heads,
  log-std clamped to [-20, 2], unbounded noise. One 32-vector per inference, repeated along the 50 positions.
- *Value model.* Four hidden layers of 512 with ReLU, 16 bins over [0, 1], cross-entropy on two-hot targets (the paper
  gives the bins but not the target form).
- *Update.* Its equation 10: `w = Y min(exp(A / 0.5), 20)`, with log-probabilities averaged over noise dimensions. `Y`
  is episode success. AdamW at 1e-5 (actor) and 2e-5 (value), betas 0.9 and 0.95, weight decay 0.01, gradient-norm
  clip 1.0, batch 2,048, half from successful episodes. Sampling is with replacement.
- *Advantage.* `A = G - V(s)`, with `G = 0.99^(steps to the success decision)` and 0 for a failed episode. The paper does
  not spell out AWR's target. A failed episode's truncated return counts as 0 and has no bootstrap, so `V` is
  a success-probability estimate under the balanced batches; this is a choice of this study.
- *Departure from the paper.* Its 1x ratio would give about 8 actor updates per epoch here, about 300 in all at a learning
  rate of 1e-5. So each epoch has 100 joint updates and 100 value-only updates, what the paper runs per collection
  batch: 3,700 actor updates per seed. This is not tuned: no 5b result changes it.
- *Training.* Init states 10-49, one episode each per epoch (40 episodes), 37 epochs, 1,480 episodes per training seed;
  training seeds 0, 1, 2 in turn. Actor updated between epochs; stochastic rollouts. Replay keeps every episode.
  Episodes of training seed `t`, index `e`: seed 200,000 + 10,000 `t` + `e`.

**Evaluation (init states 0-9 x seeds 16-31, 160 episodes per arm).** A deterministic noise makes an episode deterministic
(`check_step5a.py`), so the actor mean would give the same outcome on all 16 seeds of a state. The primary arm is
therefore the stochastic actor, as the base is stochastic.

- `base`: the frozen policy on the A100, through this path (step 4's base ran on T4s, so it does not pair exactly).
- `actor`: the final actor, sampled, per training seed.
- `prior`: the untrained actor, sampled: N(0, 1) per inference, repeated along the chunk. It separates what learning adds
  from what the repeated form adds, which step 5a says may be large.
- Secondary, one episode per init state (seed 16): `actor_mean` per training seed and `zero` (the untrained mean). These
  are 10 outcomes each, reported and not tested.

At most 5,280 episodes (4,440 + 160 + 160 + 480 + 30 + 10). Estimated at about 16 h and 108 compute units at step 5a's
rate; the pilot replaces this estimate. Same pairing as step 4: same (seed, init state) across arms, exact McNemar.

**Harness and gates, each stops the run.** `base` runs first and must score within 16 of step 4's 37/160 (the 20%
tolerance of the other steps; this count and the 50/120 below are in `week6/noise_reachability_step5b_colab/references/`).
After training seed 0, its success over the last 5 epochs (200 episodes) must be at least
30%, against 41.7% for the base on the same states (50/120, step 4's collection); 30% is about 3 standard errors below
that, so a lower value means that the actor has broken the policy.

**Pilot.** One training epoch of a separate seed (9, not a result), with its update, on the A100, timing both; evaluation
time is estimated from that rate. The run then uses 37 epochs if the estimate fits in 17 h, otherwise the largest
number of epochs that does. That number is written down before training seed 0 starts.

**Decision, per training seed (`analyze_step5b.py`).** As step 4: *meets* if `actor` beats `base` by a net of at least 8 of
the 80 systematic episodes with paired p < 0.05, and the guard on the 80 reachable episodes holds (no more than 4 net
below the base, and not significantly below). *Learned* if `actor` also beats `prior` on the same systematic
episodes by a net of at least 8 with p < 0.05.

- *Meets and learned in at least 2 of 3 seeds:* the hypothesis is rejected: the actor transfers to held-out systematic
  states. Re-check for bugs first (CLAUDE.md), then it is the study's main claim.
- *Meets in at least 2 seeds, not learned in 2:* the repeated form explains the gain, not learning.
- *Otherwise, with `actor` at least 8 net above both `base` and `prior` on reachable episodes (p < 0.05 each) in at least
  2 seeds:* the actor transfers to held-out reachable states and not to systematic ones. The hypothesis stands, and this is the clean form
  of the split between the two kinds of state.
- *Otherwise, training success over the last 5 epochs at least 8 points above the untrained actor's in at least 2 seeds:*
  the actor learned its training states and did not transfer. The untrained actor's rate is epoch 0 of the three seeds
  (120 episodes), so a gain from the repeated form alone does not count here.
- *Otherwise:* the learner did not learn at this budget, so 5b says nothing about transfer.

**Result (Colab, 2026-11-01; `week6/noise_reachability_step5b_colab/`).** No reading on transfer: the learner did not learn
at this budget. Harness: base 43/160 against 37. Systematic /80: base 0, prior 0, actors 2, 2, 5 (nets +2, +2, +5; none
meets). Reachable /80: base 43, prior 46, actors 56, 48, 47. Training success over the last 5 epochs against the untrained
actor's 37.5%: 48.0, 40.0, 42.5, so 1 of 3 seeds reaches the 8 points required. The session was lost at 97 of 111 epochs and
resumed; the episodes repeated at the seam came out the same.

## Step 6: do the labels and arm B's uneven gain hold beyond task 0? (design 2026-10-31)

Steps 4 and 5a rest on task 0 and five of its states, with one checkpoint for arm B. Step 1's labels on the other weak
tasks were never checked on fresh seeds. This step needs no training: the base and arm B's checkpoint (step 4's, same
SHA-256) on tasks 4, 6, 7, 8 x init states 0-9 x seeds 16-31, `n_action_steps=10`, through step 4's evaluation path
(`self_filtered_bc.py --stage eval`): 2 policies x 640 episodes. Task 0's base and B come from step 4's files, same
protocol. Step 4 is why the claim is tested here: this design was written after seeing task 0, so task 0 is not
out-of-sample and the primary test leaves it out.

**Groups, from the base on seeds 0-15 only (step 1; fixed before any new episode).** *Systematic:* 0 of 16. *Low
reachable:* 1 to 8 of 16, states with room to gain, so a ceiling cannot explain a difference. States above 8 of 16 are
in no group. Successes of 16 per init state 0-9:

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Systematic | Low reachable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 13 | 4 | 0 | 16 | 16 | 5 | 1 | 8 | 5 | 11 | 2 | 1, 5, 6, 7, 8 |
| 6 | 7 | 5 | 15 | 9 | 9 | 16 | 12 | 16 | 0 | 8 | 8 | 0, 1, 9 |
| 7 | 8 | 16 | 3 | 6 | 7 | 1 | 13 | 16 | 4 | 16 | none | no test |
| 8 | 11 | 8 | 2 | 1 | 0 | 10 | 14 | 13 | 5 | 6 | 4 | 1, 2, 3, 8, 9 |

**Hypotheses, each with its rule (`analyze_step6.py`).** A state's *gain* is (B successes - base successes) / 16 on seeds
16-31, the same episodes for both.

- *Labels hold:* the base solves at most 4 of the 48 episodes on the 3 systematic states of tasks 4, 6, 8, and at most 2
  of 16 on any one of them. This cannot exclude rates near 10%; the report gives the exact interval and each state's count.
- *Informative:* B's mean gain on the 13 low reachable states of those tasks is at least 0.10. If not, B has too
  little effect there to compare states.
- *Uneven gain:* `D`, the mean over tasks 4, 6, 8 of (mean gain of the task's low reachable states minus its systematic
  state's gain), is at least 0.10, with one-sided exact stratified permutation p < 0.05. Within each task, every choice
  of which state plays the systematic one is enumerated (6 x 4 x 6 = 144 assignments), so states are the unit and a task's
  overall gain cannot produce `D`; ties count against the hypothesis. The smallest possible p is 1/144. The report also
  gives each task's gap and a 95% bootstrap interval on `D` (seeds resampled within state, paired; states fixed, so it
  shows episode noise only).

**Reading, fixed now, in this order.**

- *Labels fail:* the two classes do not carry beyond task 0. The paper reports per-state rates, not classes.
- *Not informative:* B's gain on the low reachable states was too small to compare states; nothing is read about
  unevenness.
- *Labels hold, uneven gain holds:* gain is smaller on the systematic states on tasks not used to form the claim, with one
  B checkpoint. This does not show a separate class: on a probability scale a uniform log-odds lift also shrinks at the
  floor, and 48 episodes per arm cannot separate the two. The log-odds `D` below is the fixed check on this.
- *Labels hold, informative, uneven gain missing:* not replicated at this size. It is not evidence that the unevenness
  is absent beyond task 0; see the power below.

*Changed after the review, before any episode ran (2026-10-31):* the per-state limit on the labels, the intervals, the
log-odds secondary, and the wording of the three readings, which claimed more than this size allows.

**Power (model-based estimates, `calibration/step6_power.py`, output in `week6/noise_reachability_step6_kaggle/`).** Rates of the states are drawn from their seeds 0-15 counts, 20,000 simulations per row, on the rule above. A true class
effect (low reachable + x, systematic + 0) passes `D` given that the labels hold and B is informative with probability
0.39 at x = 0.15, 0.64 at 0.20 and 0.96 at 0.30. A uniform log-odds lift with no classes passes with probability 0.42 at
+1 and 0.67 at +1.5, which is why a pass is not read as a class. The informative check passes half the time at a true
gain of 0.10. The pooled labels rule passes with probability 0.91, 0.47 and 0.13 at true rates of 5%, 10% and 15%.


Secondary, not part of the rule: the same `D` with task 0 included; the same `D` on the log-odds scale ((successes + 0.5) /
(failures + 0.5), same assignments); Spearman correlation across all 50 states between
the rate on seeds 0-15 and the gain; each task's B against base by exact McNemar. These 50 states carry their per-state
table in the JSON report for a figure. Limits: 3 systematic states, one B checkpoint, and four tasks chosen in step 1 as
the weakest. B's task-level results on these tasks at `n_action_steps=1`, seeds 0-2, existed before this design
(`week6/continued_training_kaggle/`), so the design is not blind to them.

**Harness, stops the run.** The base runs first, then B. Per task, the base on seeds 16-31 must be within 18 of 160 of
its seeds 0-15 count (79, 97, 90, 70 for tasks 4, 6, 7, 8): 2 standard errors of a difference at p = 0.5, 2 x sqrt(2 x
0.25 / 160) x 160 = 17.9. Step 4 passed the same path on task 0 (37 against 41). B's checkpoint must match step 4's
SHA-256 `b75872e158799eb4`.

Same protocol on both sides: same tasks, states, seeds, horizon, control and code path. **Compute (estimate).** 1,280
episodes at step 4's 60-80 s per episode over 4 streams: about 6 h, one Kaggle session, probably less because these
tasks fail less often.

**Result (Kaggle, 2026-11-01; `week6/noise_reachability_step6_kaggle/`).** The labels hold: the base solved 1 of 48
episodes on the systematic states (95% interval 0.1-11.1%). Not informative: B's mean gain on the 13 low reachable states
was -0.067, against the 0.10 required, so nothing is read about unevenness. B against base over the four tasks: 341 to
326 of 640, with +14, -9, +21, -11 per task. Exploratory: B reached 6 of 16 on task 4 state 2, 0 of 32 for the base.

## Step 7: does deep noise reach systematic states beyond task 0? (design 2026-11-01)

Step 5a is the study's most unusual result: all 5 systematic states of task 0 were reached by fixed noises, none by
sampled noise. It rests on one task, and on hits picked from the same episodes they are counted on. This step asks the
same question on the 3 systematic states of step 6, whose labels held on fresh seeds: task 4 state 2, task 6 state 8,
task 8 state 4. Step 6 is why these states: they were fixed before this design, by seeds 0-15. No training.

**Hypothesis.** Written before any step-7 episode: pooled over the 3 states, the fixed `repeat` noise succeeds more often
than sampled noise, with `repeat` hits on at least 2 of the 3 states. Step 5a's rate (5.7%, 20/350) is the expectation; the
rule passes well below it, so a pass does not confirm that rate.

**Protocol (fixed before the run).** As step 5a through `deep_noise_search.py`: `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, 520 steps, hard resets, families `sampled`, `full`, `repeat`, the same noises by id as step 5a.

- *Episodes.* Noise ids 0-69 per family on each of the 3 states (210 per state, 630). Episode seed 1000 + noise id, as in
  step 5a, which overlaps no seed of steps 0, 1, 4 or 6 on these tasks.
- *Control, one per task, ids 0-34 per family (315).* The state of the task's low reachable group (step 6) whose base
  count on seeds 0-31 is closest to one third, 10.67 of 32, ties to the lower state: task 4 state 5 (9), task 6 state 0
  (11), task 8 state 8 (10). Step 5a's control ran near that rate. The rule uses seeds 16-31 as well, so it is chosen after
  those results; these states are controls only and enter no test.
- *Confirmation.* As step 5a: the first 3 hits of each read fixed family on each state, with jitter seeds 5000-5003, Gaussian
  std 0.3 (at most 72 episodes).
- *Harness, first, stops the run.* Per task, the base on init states 0-9 x seeds 0-2 through `eval_noise_selection.py`
  against step 0's count (14, 20, 15 of 30 for tasks 4, 6, 8), gap at most 6, step 5a's limit (90 episodes).
- *Control gate, stops the run.* `sampled` under 3 of 35 on any task's control. A fixed family under 3 of 35 on a task is
  not read on that task, and any hit it has there is reported apart.

At most 1,107 episodes.

**Rules (`analyze_step7.py`).**

- *Rate.* Pooled over the states of the tasks where `repeat` is read: `repeat` against `sampled`, one-sided Fisher exact
  on episodes, p < 0.05. The two have the same episode count; with all 3 tasks read that is 210 each, and `repeat` needs 5
  hits if `sampled` has 0, 7 if 1, 9 if 2. Only `repeat` is tested, because step 5a is what predicts it; `full` is reported.
- *Spread.* States with at least one `repeat` hit, of the states where it is read: at least 2 of 3. A `sampled` hit on a
  state does not affect this: the rate rule carries the comparison with sampling.
- *Reached.* Per state, step 5a's classes: robust, sampled, fragile, unreached. Reported, with each state's fixed-only
  status (a read fixed family hit it and `sampled` did not), and not part of the reading.

The report gives exact 95% intervals on both pooled rates and the number of distinct `repeat` noises that hit: noises are
reused across states, so 5 hits may be fewer than 5 noises.

**Reading, fixed now, in this order.**

- *A harness or a sampled control failed:* not read.
- *No fixed family read on any task:* not read.
- *Rate passes, spread at least 2:* the result extends beyond task 0: pooled over these 3 states, `repeat` noise succeeds
  more often than sampled noise, and it does so on more than one state.
- *Rate passes, spread under 2:* a higher rate, concentrated on too few states to call general.
- *Rate fails, some state reached by a fixed family:* existence only. Hits happened, at a rate not shown above sampling.
- *No state reached by a fixed family:* unreached at this depth. Step 5a's result stays a result about task 0.

*Changed after the review, before any step-7 episode ran (2026-11-01):* the first design required at least 2 "fixed-only"
states, so one sampled hit could change the reading however many `repeat` hits a state had, and one `full` hit could count
towards a claim about `repeat`. Spread replaces it. Also added: the second "not read" branch, intervals, distinct noises.

**Power (exact, assuming independent episodes; `calibration/step7_power.py`, output in
`week6/noise_reachability_step7_kaggle/`).** Step 6 is the only unbiased estimate of `sampled` on these states: 1 of 48, or
2.1%; on seeds 0-31, 1 of 96. With the same true `repeat` rate of 5.7% on each state, the rule with spread passes with
probability 0.82 if `sampled` is 1% and 0.54 if 2%; at a true `repeat` rate of 3%, 0.28 and 0.09. With step 5a's uneven
rates (1, 1, 7, 3, 8 of 70) over its subsets of 3 states, 0.73 and 0.52. So a miss says little: if `sampled` is near 2% it barely
rules out even step 5a's rate. At equal true rates of 1% and 2% the rule passes with probability 0.009 and 0.020.

**Task 8's scene is not fixed by its init state (found in the smoke run, before any step-7 episode).** On task 8 the stove
is a fixture placed when the simulator is rebuilt, and the init state does not restore it: between resets with the same
init state its position changed by 0.012 and 0.018 m in two comparisons, also with the same seed and every generator seeded.
No body moved on tasks 0, 4, 6 or 7 at state 4 (`week6/noise_reachability_step7_kaggle/fixture_check.txt`, 2 comparisons per task). So on task 8 a fixed-noise
episode is not deterministic, `check_step5a.py` fails its environment-seed check there, and the notebook runs it with
`--scene_varies`. Task 8 stays in the rules: `repeat` and `sampled` meet the same stove variation, so the rate
comparison is fair, but a task-8 hit may owe something to the stove. The report also gives the rate on tasks 4 and 6 alone
(140 episodes each), as a secondary: `repeat` needs 5 hits there if `sampled` has 0. Step 1 and 6's task-8 counts ran the same
way, so "init state" on task 8 means a family of scenes.

The same smoke run saw 3 search outcomes at noise id 0, which are part of the protocol's episodes and were not chosen by
this design: `repeat` on 4/2 failed, `sampled` on 6/8 failed, `full` on 8/4 succeeded in 373 steps. They will run again
on Kaggle and those runs are what count. Nothing above changed because of them.

**What it cannot show.** As in step 5a, a noise is picked after seeing the outcome, so this shows that such noises exist,
not that a policy can find them. Noises are the same ids as step 5a, so overlap of hits between tasks can be looked at, as
exploration only. The Fisher test treats episodes as independent, but the 210 `repeat` episodes are 70 noises reused on 3 states, so
it is valid for these 3 states and says nothing about other states. Three states on three tasks, with one checkpoint. Task 4 state 2 also moved under arm B in step 6 (exploratory, 6
of 16): a report of both on that state stays exploratory.

**Compute (estimate).** 1,107 episodes on Kaggle 2 x T4 in 4 streams. Step 6 took 16.2 s per episode at a mean of 396 steps,
so about 21 s at 520 steps: about 6.5 h, one session.

**Result (Kaggle, 2026-11-02; `week6/noise_reachability_step7_kaggle/`).** Existence only: hits happened, but `repeat` 16 of
210 against `sampled` 10 of 210 did not pass (p 0.156), and without task 8 it was 6 against 6 of 140. Exploratory: `sampled`
solved 4/2 and 8/4, so with base seeds 0-31 they stand at 6/102 and 5/102, and on them "systematic" means rare, unlike task
0's 0 of 350. The observed `sampled` rate, 4.8%, is above the 1-2% the power table assumed, so the test had less power than
stated. Only 6/8 kept task 0's pattern: 0/102 sampled, 3/70 `repeat`, 5 of 12 jittered, one state picked after seeing the data.

## Step 8: does fixed noise beat sampling on states that resist sampling? (design 2026-11-02)

Step 7 raised two questions it could not answer. Task 6 state 8 kept task 0's pattern, but it was picked after seeing the
data. And a label of 0 of 32 turned out to cover states that sampling solves at about 5%, so a test on such states mixes
rare states with resistant ones. This step tests 6/8 on fresh noises, and tests states that were never labelled or searched,
with far more sampled episodes than any label so far. LIBERO has 50 init states per task (confirmed locally on tasks 0, 4, 6,
7, 8). Labels and deep searches used only states 0-9. States 10-49 are not unused: steps 2, 3, 3b, 4 and 5b collected or
trained on them. The frozen base was never trained on them by this study, but arm B and the learned actors were, so those
checkpoints are not read on these states here. No training.

**Hypothesis.** Written before any step-8 episode. On 6/8, `repeat` noise succeeds more often than sampled noise; expected
about 4.3% (step 7, 3 of 70) against under 1%. On unlabelled states that fail 32 of 32 sampled episodes, the same holds, pooled:
expected about the same rates. This is a claim about the `repeat` family against the `sampled` family. It does not say that
any one chosen noise is good, or that a policy could find one.

**Protocol (fixed before the run).** As step 7 through `deep_noise_search.py`: `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, 520 steps, hard resets. Families `sampled` and `repeat` only: `full` is dropped because no rule uses it and
the episodes go to the tests. Tasks 0, 4, 6, 7. Task 8 is excluded because its stove is not fixed by the init state (FAIL-010).

- *Harness, first, stops the run.* Per task, the base on init states 0-9 x seeds 0-2 against step 0's count (7, 14, 20, 18 of 30
  for tasks 0, 4, 6, 7), gap at most 6 (120 episodes).
- *Control gate, stops the run.* Each task's control state from steps 5a and 7, and for task 7 the state closest to one third
  on seeds 0-31: 0/9, 4/5, 6/0, 7/4. Noise ids 35-69, new on the states steps 5a and 7 ran, both families (280 episodes).
  `sampled` under 3 of 35 on any control stops the run; `repeat` under 3 of 35 on a task drops that task's states from the
  screen's second part and from the tests, so if that task is 6, test 1 does not run.
- *Test 1, 6/8.* Noise ids 100-399, 300 episodes each of `sampled` and `repeat` (600).
- *Screen, unlabelled states.* States 10-29 of tasks 0, 4, 6 (60 states), `sampled`, ids 400-407 (480 episodes). States with 0 of 8
  go on to ids 408-431, at most 20 states (480 episodes), taken in rounds over tasks 0, 4, 6 with the lowest state first. Task 7
  skips the first part: its states 33, 38 and 47 run ids 400-431 directly (96 episodes). They are the only task-7 states among
  10-49 with 0 hits in step 3b's base rollouts (0 of 19, 23, 24, soft resets), and no state in 10-29 has fewer than 3 hits, so a
  screen of 10-29 would likely leave task 7 out. Those step-3b episodes do not count towards eligibility.
  States with 0 of 32 are *eligible*, the study's usual label on the same episode count.
- *Selection.* Up to 4 eligible states, taken in the same rounds. This uses no outcome beyond eligibility. Fewer than 2
  eligible states: test 2 is not run. Each state's scene, 6/8's included, is checked before its test (`check_scene.py`): one noise
  (id 3000, in no test) runs twice for 120 steps from the state with different environment seeds, and no body pose may differ by
  more than 1e-6 between the two scenes, FAIL-010's measure. Outcomes could not show this, because a failed episode ends at step 520
  whatever the scene. Action differences are recorded but do not decide, so a last-bit GPU difference cannot change which states
  are tested. These short runs are not counted. An eligible state that fails is replaced by the next one; if 6/8 fails, test 1
  does not run, and the report says why.
- *Test 2, unlabelled states.* The i-th selected state (i = 0-3) has noise ids 1000 + 150i to 1149 + 150i, 150 episodes each of `sampled`
  and `repeat` (up to 1,200). The screen's episodes are not counted, because those states were picked for failing them; a
  fresh `sampled` count removes that bias.
- *Confirmation.* As step 7: the first 3 `repeat` hits on each tested state, jitter seeds 5000-5003, std 0.3 (at most 60).

Episode seed is 1000 + noise id. Every tested state has its own noise ids, so unlike step 7 the pooled episodes are
independent. At most 3,316 episodes.

**Rules.** Two tests, each a one-sided Fisher exact test of `repeat` against `sampled`, Holm-corrected at 0.05: the smaller p must
be under 0.025 and then the other under 0.05.

- *Test 1:* 6/8, 300 against 300. If only one of the two tests runs, it needs p under 0.05.
- *Test 2:* pooled over the selected states read, up to 600 against 600.

**Reported, not part of the reading.** Per state: both counts with exact 95% intervals. Sampled depth: how many tested states stay
at 0 hits; 0 of 300 on 6/8 would put a one-sided 95% bound under 1%. How well the label predicts: the pooled `sampled` rate on
eligible states, and how many of them were hit. `repeat` hits by distinct noise and jittered successes, which bear on
whether a good noise has good neighbours. The count of eligible states, which is a count over the screened states only.

**Reading, fixed now, in this order.**

- *A harness or a sampled control failed, or neither test ran:* not read.
- *Both tests pass:* fixed noise beats sampling on resistant states beyond task 0, on a state and on unlabelled states chosen in
  advance.
- *Test 1 passes, test 2 fails:* 6/8 is confirmed, but the pattern did not appear on unlabelled states.
- *Test 2 passes, test 1 fails:* the pattern holds on unlabelled states, and step 7's result on 6/8 did not replicate.
- *Test 2 not run, test 1 passes:* 6/8 is confirmed; too few resistant states to test anything else.
- *Test 1 not run, test 2 passes:* the pattern holds on unlabelled states; 6/8 was not tested.
- *Otherwise:* not shown. Claim 2 stays a result about task 0.

**Power (exact, independent episodes; `calibration/step8_power.py`, output in `week6/noise_reachability_step8_kaggle/`).** Written
as true `sampled`, true `repeat`, power at 0.025 and 0.05. Test 1: 0.5%, 4.3%, 0.86, 0.92; 1%, 4.3%, 0.66, 0.77; 2%, 4.3%, 0.29,
0.40. Test 2 with 4 states: 0.5%, 4.3%, 1.00, 1.00; 1%, 4.3%, 0.95, 0.97; 2%, 4.3%, 0.58, 0.69; step 7's own rates (4.8%, 7.6%), 0.48,
0.60. So the design is good if eligible states are truly resistant, and weak if they are only rare as step 7's were. False pass at
equal rates stays under 0.05 (0.015-0.03 at 1%, 0.03-0.04 at 5%). With 2 eligible states test 2 has 300 against 300, as in test 1's row.
A true rate on a state of 5% passes the 0 of 32 screen with probability 0.19, so some eligible states will be rare, not resistant.
On tasks 0, 4 and 6, 7 of the 30 states 0-9 are at 0 of 32, so about 14 of 60 are expected there (estimate), and task 7 has 3
candidates.

These figures assume task 6's `repeat` control passes. At step 7's rate on 6/0, 5 of 35, it fails with probability about 0.11
(estimate), which would drop test 1 and task 6's states. The threshold stays at step 5a's 3 of 35.

**What it cannot show.** Three weak spots remain. One checkpoint. `repeat` against `sampled` compares two families, so a pass does
not show that a good noise can be found. The screened states are not a random sample of hard states: the selection takes what the rounds
give, and task-level differences may decide the pooled result, so the report gives each state's counts.

**Compute (estimate).** At step 7's rate (1,095 episodes in 6.0 h, 19.7 s each), 3,316 episodes take about 18.1 h: two Kaggle
sessions, with the second resuming. The screen ends in the first session, so eligibility is known early.

**Result (Kaggle, 2026-11-02/03; `week6/noise_reachability_step8_kaggle/`).** Both tests pass: fixed noise beats sampling on
resistant states beyond task 0, on a state and on unlabelled states chosen in advance. Test 1, 6/8: `repeat` 10 of 300 against
`sampled` 2 of 300, p 0.0184. Test 2, 0/11, 4/11, 6/12 and 7/33 pooled: 38 of 600 against 1 of 600, p 4.1e-11. Holm passes both.
Exploratory: 7/33 gives 25 of the 38 hits, and without it test 2 is 13/450 against 1/450 (p 0.00085); 0/11 and 4/11 alone are 3
against 0. On the easy control states `repeat` is no better than `sampled` (43 against 49 of 140), so the result is about states
that resist sampling, not a better policy. Jittered successes 22 of 60.

## Compute

Kaggle 2 x T4 (45 h a week) for base-policy evaluations, Colab A100 (50 h a month) for steps 2-4. At
`n_action_steps=10` the 12 weak-task runs of step 0 (120 episodes) took 32 minutes on Kaggle and all 10 tasks
65 minutes (`week6/noise_reachability_step0_kaggle/`).

| Step | Where | Estimate |
|---|---|---|
| 0 | Kaggle | done: about 2 h over three versions |
| 1 | Kaggle | done: 170 min (seeds 3-15) and 16 min (repeat), plus 9.6 h lost to a version at the wrong setting |
| 2 | Colab A100 | done: 600 (selection) + 300 (report) episodes, one process per task in parallel; the last processes started 82 and 37 minutes after launch |
| 3 | Colab A100 | done: 1,500 training episodes x 2 tasks in parallel (episode time summed to 13.7 h and 15.9 h), then 120 evaluation episodes; launched 2026-10-27 14:13 UTC, last evaluation process started 2026-10-28 09:18 UTC, including a disconnect of about 2 h |
| 3b | Kaggle 2 x T4 | done: 1,000 base episodes in 4 streams (3.9-4.0 h of episode time per stream, two streams per T4), then 6 critic fits; launched 2026-10-28 19:51 UTC, analysis at 2026-10-29 00:13 UTC |
| 4 | Kaggle 2 x T4 | estimate: 120 collection episodes, 6 x 2,000 LoRA steps, 1,280 evaluation episodes (roughly 5-7 h of evaluation); two sessions |
| 6 | Kaggle 2 x T4 | done: 1,280 evaluation episodes, about 5.9 h, one session |
| 5a | Colab A100 | done: 1,219 episodes after the pilot on 4 processes; first record 2026-10-30 22:04 UTC, last process started 2026-10-31 01:52 UTC |
| 5b | Colab A100 | done: 111 epochs, 840 evaluation episodes, two sessions (lost at 97 epochs); about 110 compute units is an estimate |
| 7 | Kaggle 2 x T4 | done: 1,095 episodes, 6.0 h between first and last provenance records, one session |
| 8 | Kaggle 2 x T4 | done: 3,172 episodes, 10.3 h and 5.9 h between first and last provenance records, two sessions |

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
- `analyze_noise_signal.py`: step-3b critics, action AUC and decision; `kaggle_step3b_noise_signal.ipynb` runs the
  collection and the analysis on Kaggle; `calibration/step3b_null_calibration.py`: the intervals' false-positive check.
- `analyze_step4.py`: step-4 comparisons and decision; `demo_episodes.py`: arm C's task-0 demonstrations from
  `lerobot/libero` in `self_filtered_bc`'s episode format (trained with `self_filtered_bc.py --val_demo_every 5`);
  `check_step4.py`: the real-model checks; `kaggle_step4_weight_update.ipynb` (built by `build_step4_notebook.py`)
  runs checks, collection, training, evaluation and the analysis on Kaggle, resuming across sessions.
- `deep_noise_search.py`: step-5a episodes (fixed-noise wrapper, families, confirmation); `check_step5a.py`: the
  real-model checks; `analyze_step5a.py`: pilot choice, classes and decision; `run_step5a_deep_noise.sh`: the exact
  commands; `colab_step5a_deep_noise.ipynb` (built by `build_step5a_notebook.py`) runs it all on Colab.
- `noise_awr.py`: step-5b actor, value model, returns, replay; `train_noise_awr.py`: collection, update, evaluation;
  `noise_steering.py` (`install_repeated`): the repeated noise; `check_step5b.py`: the real-model checks;
  `analyze_step5b.py`: gates and decision; `run_step5b_noise_awr.sh`: the exact commands;
  `colab_step5b_noise_awr.ipynb` (built by `build_step5b_notebook.py`) runs it all on Colab.
- `analyze_step6.py`: step-6 groups, harness, permutation test and reading; `kaggle_step6_replication.ipynb` (built by
  `build_step6_notebook.py`, from step 4's notebook cells) runs the evaluation and the analysis on Kaggle.
- `analyze_step7.py`: step-7 gates, rate test and reading; `kaggle_step7_deep_noise.ipynb` (built by
  `build_step7_notebook.py`, from step 6's notebook) runs `deep_noise_search.py` on Kaggle.
- `analyze_step8.py`: step-8 gates, screen, selection, the two tests and reading; `check_scene.py`: the scene check;
  `kaggle_step8_resistant_states.ipynb` (built by `build_step8_notebook.py`, from step 7's notebook) runs them on Kaggle.

## Status

Steps 0, 1 and 2 done (above); step 2 is a negative result. Step 3 ran on Colab and failed as a test: the
critic diverged (above). Step 3b (whether a Monte Carlo critic can tell the base's own noises apart) ran on
Kaggle and is not read: its shuffled-noise control was biased (above). Step 4 ran on Kaggle (above, 2026-10-31): negative, no arm meets the hypothesis. Step 5a ran on Colab (above, 2026-10-31): "mostly reachable", all 5 systematic states reached by fixed noises, none by
sampled noise. Step 5b ran on Colab (above, 2026-11-01): no reading on transfer, the learner did not learn at this budget. Step 6 ran on Kaggle (above, 2026-11-01): the labels hold; B's gain was not informative. Step 7 ran on Kaggle (above, 2026-11-02): existence only, `repeat` did not beat `sampled`, and on two of the three states sampling also solved the state. Step 8 ran on Kaggle (above, 2026-11-03): both tests pass, `repeat` beats `sampled` on 6/8 and on four unlabelled states chosen in advance, pooled.

Before step 0 ran, smoke tests on the local RTX 4090 (shared with another job), one episode each of task 0,
seed 0, init state 0: `n_action_steps=10` through `evaluate_all.py` and `n_action_steps=50` through
`run_step0_action_steps.sh` (`SMOKE=1`). Both ran the full 520 steps with the setting applied (it appears in
the policy config printed in `eval.log`) and failed, about 3 minutes each including the model load. They
check that the setting runs, not its success rate.
