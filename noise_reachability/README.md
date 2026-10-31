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
| 5a | Colab A100 | done: 1,219 episodes after the pilot on 4 processes; first record 2026-10-30 22:04 UTC, last process started 2026-10-31 01:52 UTC |
| 5b | Colab A100 | not run: at most 5,280 episodes, estimated at about 16 h; the pilot replaces this |

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

## Status

Steps 0, 1 and 2 done (above); step 2 is a negative result. Step 3 ran on Colab and failed as a test: the
critic diverged (above). Step 3b (whether a Monte Carlo critic can tell the base's own noises apart) ran on
Kaggle and is not read: its shuffled-noise control was biased (above). Step 4 is running on Kaggle (two sessions so far, not finished, no decision). Step 5a ran on Colab (above, 2026-10-31): "mostly reachable", all 5 systematic states reached by fixed noises, none by
sampled noise. Step 5b's design is fixed (above, 2026-10-31); not run.

Before step 0 ran, smoke tests on the local RTX 4090 (shared with another job), one episode each of task 0,
seed 0, init state 0: `n_action_steps=10` through `evaluate_all.py` and `n_action_steps=50` through
`run_step0_action_steps.sh` (`SMOKE=1`). Both ran the full 520 steps with the setting applied (it appears in
the policy config printed in `eval.log`) and failed, about 3 minutes each including the model load. They
check that the setting runs, not its success rate.
