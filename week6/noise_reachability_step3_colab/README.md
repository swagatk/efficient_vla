# noise_reachability step 3: SAC noise policy (Colab), a failed training run

Step 3 of `noise_reachability/README.md`: a DSRL-style actor chooses the 70 executed noise entries of each
SmolVLA inference, trained with SAC on task success; tasks 7 and 0, 1,500 training episodes each on init
states 10-49, then the deterministic actor against the paired base on init states 0-9 x seeds 0-2.
`lerobot/smolvla_libero` at `n_action_steps=10`, 520 steps. Colab A100-SXM4-40GB,
`noise_reachability/colab_step3_noise_sac.ipynb`, the two tasks in parallel, every process at commit
`66d2a58` with a clean tree (`provenance.jsonl`, `run/repo_commit.txt`).

Stated before the run: on each task the actor raises success by at least 10 points over the base (paired
p < 0.05), and on task 0's systematic states (init states 2, 5, 6, 7, 8) its success stays below a third of
its success on reachable states.

**Outcome: the run is not a test of the hypothesis. SAC's critic diverged on both tasks early in training**,
so the evaluated actors were trained against a critic with no usable values. The evaluation numbers below are
what was measured, not evidence about learned noise policies.

`run/` passes `tools/validate_run.py --tasks "0 7" --seeds 0-2 --init_states 0-9 --expect n_action_steps=10
--expect policy_path=lerobot/smolvla_libero` (one clean commit, every (task, seed, init state) exactly once
per policy, no tracebacks in `eval.log`), and `analyze_step3.py` run locally on it reproduces
`step3_report.txt` byte for byte.

## Evaluation as measured (`step3_report.txt`)

| Task | Base | Actor | Only base | Only actor | McNemar p | Difference (95% interval) |
|---|---|---|---|---|---|---|
| 7 | 18/30 | 11/30 | 10 | 3 | 0.092 | -23.3 (-39.0 to +3.3) |
| 0 | 5/30 | 8/30 | 0 | 3 | 0.25 | +10.0 (-4.2 to +10.0) |

Task 0 systematic episodes: actor 0/15, base 0/15 (reachable: actor 8/15, base 5/15). The interval is
conditional on the number of discordant pairs, so with 3 discordant pairs all one way its upper end equals the
estimate; it understates the uncertainty here. Neither task meets "+10 points with p < 0.05".

## The divergence (`step3_diagnosis.txt`, from `run/task_<t>/train_log.csv`)

With reward 0 or 1 at the end and discount 0.99, Q should stay near [0, 1] plus the entropy bonus.

- **At the end of warm-up Q was already about 6** (task 7: 5.86, task 0: 6.31 at episode 20). With alpha 0.01
  and a 70-dimensional action the entropy bonus (about 0.35 per decision at entropy 35) outweighs the
  reward of 1.
- **Then the critic diverged on its own**: Q reached 113 (task 7) and 443 (task 0) by episode 100 while alpha
  was 0.01-0.04, too small for the entropy term to explain it.
- **Then alpha ran away**: entropy stayed just below the target of -70, the automatic temperature kept
  rising, and Q followed: task 7 ended at Q 1.0e11, alpha 5.2e6; task 0 at Q 4.0e12, alpha 8.7e7.
- Training success (stochastic actor, init states 10-49; `step3_report.txt`): task 7 69/100 in the first 100
  episodes after warm-up, 46/100 in the last 100; task 0 37/100 and 38/100.

Nothing in the run stopped it: the progress cell showed success counts only, and no check bounded Q.
The SAC update itself was re-read (target critic, signs, the action scaling between replay and actor) and
looks correct; the divergence reproduces offline (below), so it is the method's instability in this setting,
not an implementation slip that has been found.

## Offline diagnosis on the recorded task-7 replay (`offline/`)

`offline/replay_sac.py` re-runs the learner on the run's 60 replay parts (60,198 transitions, equal to the
logged decisions; 841 terminal rewards, equal to the logged successes) in the training schedule: the
buffer grows episode by episode, every 5th episode is held out. No simulator, so the actor never collects
data (harsher than online). Two held-out measures every 25 episodes:

- **success AUC**: does min(Q1, Q2)(s, a) rank transitions of successful episodes above failed ones;
- **action AUC**: the same for Q(s, a) minus Q averaged over 8 random noises, the action's own part.

0.5 is no information. Table from the result files (`step3_diagnosis.txt`):

| Variant | Episodes | Q mean at end | Success AUC: best, end | Action AUC: min-max, end |
|---|---|---|---|---|
| TD, 8 updates/decision (the run's settings) | 86 (stopped: Q > 1e4) | 1.05e+04 | 0.480, 0.480 | 0.437-0.610, 0.437 |
| TD, 1 update/decision | 300 | 10.6 | 0.625, 0.328 | 0.392-0.664, 0.392 |
| TD, 1 update/decision, no entropy term in the target | 300 | 1.35 | 0.882, 0.328 | 0.401-0.593, 0.464 |
| TD target clipped to [0, 1], 8 updates/decision | 300 | 1 | 0.688, 0.540 | 0.459-0.698, 0.459 |
| TD target clipped to [0, 1], 1 update/decision | 300 | 1 | 0.702, 0.541 | 0.436-0.676, 0.437 |
| clipped, 1 update/decision, fixed alpha 0.001 | 300 | 1 | 0.734, 0.539 | 0.407-0.599, 0.407 |
| Monte Carlo return, 1 update/decision | 300 | 0.446 | 0.885, 0.882 | 0.451-0.519, 0.500 |
| Monte Carlo return, 8 updates/decision | 300 | 0.443 | 0.880, 0.880 | 0.456-0.512, 0.464 |
| Monte Carlo return, critic only, all 1500 episodes | 1500 | 0.385 | 0.907, 0.887 | 0.474-0.587, 0.581 |

- The run's settings diverge offline too (Q 1.05e4 by episode 86).
- Bootstrapped (TD) critics drift above 1 even with one update per decision and no entropy term, and their
  success AUC falls to 0.33. Clipping the target pins Q at 1 for every transition (held-out Q 0.99 for
  success and failure alike): stable, but uninformative.
- Regressing Q on the episode's own discounted return (Monte Carlo) is stable and predicts success well
  (AUC 0.88-0.91; at 1,500 episodes held-out Q 0.64 for successful and 0.08 for failed episodes).
- No critic separates actions clearly. The Monte Carlo critic on all 1,500 episodes reaches action AUC 0.58,
  but after warm-up the stored noise came from the actor, a function of the state, so this measure can pick
  up state information carried by the action; 0.58 is an upper bound on a noise effect, not evidence of one.

## What it means

Step 3 as specified does not answer its question: the critic diverged, so the actor learned from noise in a
diverged critic (task 7 lost 7 successes against the paired base). The offline runs say what would have to
change: a bounded, non-bootstrapped target (Monte Carlo returns suit a single terminal reward and at most 52
decisions), a check that stops training when Q leaves its possible range, and evidence that the critic can
tell noises apart before an actor is trained on it. That evidence needs data whose noise is independent of
the state, which the base policy's own N(0, 1) draws provide: logging features and noise from base rollouts
on init states 10-49 and fitting the Monte Carlo critic would show whether there is any action signal to
learn, without RL.

## Files

- `run/`: the Colab result folder without `checkpoint.pt` and the replay parts (on Drive,
  `efficient_vla_runs/noise_sac/run/task_<t>/`): `task_<t>/train_log.csv` (one row per training episode with
  the update means), `train.log`, `provenance.jsonl` (61 records: 60 processes of 25 episodes and one repeated
  after the disconnect, below), `eval/per_episode_{base,actor}.csv`, `eval/provenance.jsonl`, `eval.log`; `runner.log`,
  `repo_commit.txt`.
- `step3_report.txt` (+ `.json`): `python -S noise_reachability/analyze_step3.py --root
  week6/noise_reachability_step3_colab/run --tasks 7 0 --step1
  week6/noise_reachability_step1_kaggle/step1_reachability.txt.json --out <file>`.
- `step3_diagnosis.txt`: `python -S week6/noise_reachability_step3_colab/summarize_step3.py > <file>`.
- `offline/replay_sac.py`, `offline/results/<variant>.json`, `offline/<variant>.log`: the offline runs, e.g.
  `python -I week6/noise_reachability_step3_colab/offline/replay_sac.py <dir with task 7's replay parts and
  train_log.csv> --label mc_full_critic --target mc --critic_only --utd 1 --episodes 1500 --eval_every 100`
  (the other variants: `--episodes 300` with the flags in each JSON's `args`). Local RTX 4090; the seeds fix
  the result (a re-run of the first 50 episodes of `mc_utd1` from this folder matched the stored log exactly).

Run history: launched 2026-10-27 14:13 UTC on one Colab runtime (host `5fee5174a772` in `provenance.jsonl`).
It disconnected after 15:31; the task-7 process then running (episodes 150-174) crashed writing
`train_log.csv` because Drive was gone (`run/task_7/train.log`, `run/runner.log`). The runtime was
restarted (host `a0d57abc5eb9`) and step 7 resumed at 17:44 from the last saves, repeating one 25-episode
process per task (task 7: 150-174, task 0: 125-149); that is the 61st record in each `provenance.jsonl`.
Resuming truncates `train_log.csv` to the saved episodes, and it holds episodes 0-1499 exactly once and in
order for both tasks. Training episode time summed to 13.7 h (task 7) and
15.9 h (task 0); the last evaluation process started 2026-10-28 09:18 UTC.
