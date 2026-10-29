# noise_reachability step 3b: noise signal on base rollouts (Kaggle), not read

Step 3b of `noise_reachability/README.md`: the base policy (`lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, 520 steps) ran task 7 on init states 10-49 in 4 streams (`--seed` 0-3) x 250 episodes,
logging per inference its features and the executed N(0, 1) noise; Monte Carlo critics fitted offline then
measured whether the noise's part of the value ranks held-out transitions of successful episodes above
failed ones (action AUC). Kaggle 2 x T4, `noise_reachability/kaggle_step3b_noise_signal.ipynb`, every process
at commit `47cb550` with a clean tree.

Stated before the run: action AUC at least 0.55 with the 95% interval's lower end above 0.5; the result is not
read if the shuffled-noise control's interval excludes 0.5.

**Outcome: not read. The control's interval excludes 0.5**, so the measure is biased on this data and says
nothing about whether the noise carries a learnable signal. Under the protocol, step 4 follows.

## Result (`step3b_checks.txt`, from `run/step3b_report.txt.json` and `local_rerun/step3b_report.txt.json`)

| Measure | Kaggle (2 x T4) | Local re-run (RTX 4090) |
|---|---|---|
| Critic action AUC (95% interval) | 0.566 (0.549 to 0.583) | 0.570 (0.553 to 0.587) |
| Control action AUC, noises shuffled | 0.547 (0.532 to 0.562) | 0.557 (0.542 to 0.572) |
| Success AUC of Q(s, a), critic / control | 0.930 / 0.925 | 0.929 / 0.932 |
| Decision | not read (control biased) | not read (control biased) |

200 held-out episodes (7,513 transitions), critics trained on 28,775 transitions. The critics learned the
state value (success AUC 0.93), so the critic fit is not what failed. A control critic that never sees a
state's own noise still scores 0.547-0.557, so most of the critic's 0.566-0.570 is a property of the measure,
not of the noise. The critic minus control gap (+0.019 on Kaggle, +0.013 locally) was not part of the
protocol and is no larger than the change in the control between the two runs of identical code and data
(0.010), so it is not read either.

Base success on init states 10-49: 167, 174, 171, 167 of 250 per stream.

## Checks (`step3b_checks.txt`)

- Each stream: episodes 0-249 exactly once and in order, 10 provenance records (one per 25-episode process),
  all at `47cb550` with no dirty flag, settings equal to the protocol, no traceback in `train.log` or
  `runner.log`.
- Replay (from `step3b_replay.zip`, not in git): each `checkpoint.pt` saved 250 episodes and lists the 10
  parts on disk; transitions equal the logged decisions and terminal rewards the logged successes per stream;
  the replay zip's `train_log.csv` and `provenance.jsonl` equal those in `run/`.
- The stored noise times 5 is N(0, 1) as intended (36,288 decisions x 70 entries: mean 0.0003, std 1.0003,
  skew and excess kurtosis about 0, KS p 0.20, no repeated rows). So the bias does not come from stored noise
  distributed differently from the alternatives it is compared with (the cause of a "control biased" result in
  the pre-run simulation, where fake noise was uniform).
- Reproduction: the local re-run of `analyze_noise_signal.py` on the same files, commit and seeds is not
  bit-identical to Kaggle's (action AUC +0.004 critic, +0.010 control); the most likely cause is GPU arithmetic
  differing between the T4 and the 4090, not tested. The decision is the same.

## What it means

The pre-registered test cannot say whether the noise matters on task 7: the shuffled control, which should sit
at 0.5 by construction, is above it. One untested explanation: the score Q(s, a) minus the mean of Q(s, a_k)
has a spread that differs between states of successful and failed episodes, which can move a rank measure away
from 0.5 even when the noise carries nothing. A corrected measure (for example critic minus control on the same
bootstrap draws) would be a new protocol chosen after seeing this run and would have to be stated as such.
Together with step 2 (training-free noise selection negative) and step 3 (critic diverged), there is still no
evidence that steering the noise helps on task 7.

## Notebook gaps found

- Neither zip contains the streams' `checkpoint.pt`, which `analyze_noise_signal.py` needs to re-run the
  analysis; they were downloaded one by one from the Output tab. Kaggle's download unpacked each
  `checkpoint.pt` (a torch zip) into a folder, which was packed back uncompressed with the same member bytes
  before loading (all four load with `weights_only=True`).
- Time per episode on a T4 with two streams per GPU: 3.9-4.0 h of episode time per stream for 250 episodes
  (56-58 s per episode); the first processes started 19:51-19:54 UTC, the last ones 23:28-23:33 UTC, and the
  analysis at 00:13 UTC, so collection took a little over 4 h of wall time.

## Files

- `run/`: `step3b_results.zip` from Kaggle: `repo_commit.txt`, `seed_<s>/runner.log`,
  `seed_<s>/task_7/{train_log.csv, train.log, provenance.jsonl}`, `step3b_report.txt` (+ `.json`, with the
  analysis provenance). The replay parts and checkpoints are kept locally (`outputs/step3b_kaggle/`), not in git.
- `local_rerun/`: `step3b_report.txt` (+ `.json`) and `run.log` from
  `python -I -u noise_reachability/analyze_noise_signal.py --roots <4 stream folders with checkpoint.pt>
  --device cuda --out <file>` at `47cb550`, local RTX 4090, 459 s.
- `step3b_checks.txt`: `python -I week6/noise_reachability_step3b_kaggle/summarize_step3b.py --replay
  outputs/step3b_kaggle/analysis > <file>`.

Run history: launched 2026-10-28 on one Kaggle session (host `1e48e6d60617`, in every provenance record);
first process records 19:51-19:54 UTC, last 23:28-23:33 UTC; the analysis ran at 2026-10-29 00:13 UTC.
