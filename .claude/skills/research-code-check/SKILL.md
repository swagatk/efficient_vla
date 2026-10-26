---
name: research-code-check
description: >-
  Gated verification for research code in efficient_vla before it is committed or launched on Kaggle/Colab,
  and before results from a finished run are read. Use when new or changed code will produce reported numbers,
  before asking the user to start an expensive run, and when a results zip comes back. Covers tests that can
  fail, mutation checks, real-model equivalence checks, an independent fresh-context review, launch-config
  checks against the README protocol, and post-run validation.
---

# Research code check

A rerun on Kaggle or Colab costs hours of a weekly budget, so code is checked in the gates below, cheapest first,
before it is committed or launched. A gate that cannot be run (GPU busy, simulator unavailable) is reported as
not run; the code is then not described as verified (CLAUDE.md, "Verification before commit").

The failures this is built from (all in this repo):

- A Kaggle version ran 9.6 h at `n_action_steps=1` on the wrong tasks because only two of four settings were
  edited by hand (`week6/noise_reachability_step1_kaggle/README.md`). No test could see it; a launch-config
  check (gate 5) and a post-run check (gate 7) do.
- A Clopper-Pearson interval came out reversed; a known-answer test (gate 2) caught it.
- A notebook smoke cell asserted `inferences=52`, which is true only for failed episodes; an independent review
  (gate 4) caught it before the run.
- The week-5 RL harness clamped base actions and discarded init states; nobody checked that the method with its
  new part switched off reproduces the base policy (gate 3).

## Gate 1: protocol before code

The idea's README states, before the run: hypothesis with numbers, baseline number and how it is measured,
tasks x init states x seeds per stage, which episodes choose hyperparameters and which are reported, the
decision rule, and every run setting (checkpoint, `n_action_steps`, rename map). The code is then checked
against this text, not against memory.

## Gate 2: tests that fail when the code is wrong

CPU-only tests in `tests/`, as CLAUDE.md lists. For research code, in particular:

- **Known answers**: a statistic reproduces a value computed by hand or reported before (an interval, a
  p-value, a pass@k on a small table); an analysis script reproduces the numbers of an existing results file.
- **Edges of every decision rule**: values exactly at, just above and just below each threshold; ties.
- **Refusals**: incomplete or mismatched episode sets, duplicated rows, wrong settings raise, not average.
- Then the **mutation check** on every new or changed module that produces numbers:

  ```bash
  /home/swagat/miniconda3/envs/lerobot_env/bin/python tools/mutation_check.py <module.py> <tests...> [--functions f g]
  ```

  Every survivor gets either a new test or a one-line reason it cannot change a result (equivalent mutant, e.g.
  an index inside an error message). Report the killed/total count and the survivors kept.

## Gate 3: real stack, real entry point

- **Switched-off equivalence**: with the new part disabled (N=1 samples, zero residual, LoRA scale 0) the new
  path reproduces the base policy's output bit for bit on a real observation. Batched vs single calls agree
  within the dtype's rounding, measured against the spread that matters.
- **Smoke run through the exact entry point** the expensive run will use (the `run_*.sh` script, and the
  notebook cell itself where it can execute locally, with paths substituted): one task, one episode, a real
  rollout for anything touching the model, observations or the eval loop. Check the outcome line, the applied
  setting in the log, and that provenance was written.
- **Determinism**: the same seed run twice gives the same outcome (and, where logged, the same choices).
- Local runs: `MUJOCO_GL=egl`, `tools/crash_monitor.sh` running, `tools/logged_run.sh`, `nvidia-smi` first.

## Gate 4: independent review with a fresh context

The author of code is the worst reviewer of it. Before an expensive launch, run a reviewer that has not seen
the conversation: either the `code-review` skill on the diff, or an Agent (general-purpose) with this brief:

> Review <files> at commit <hash> for bugs that make the experiment's numbers wrong or the run fail. The
> protocol is <README section>; read it first. Do not edit files. Check: pairing and seeding per episode; state
> leaking between policies or episodes in one process; shapes, padding and units; which actions are executed;
> resume/skip logic and partial runs; that every setting reaches every process with the README's values;
> that decision-rule code matches the README (ties, strict vs non-strict); selection and report sets
> disjoint; any way a notebook can run the wrong setting silently. Rank BLOCKER / MAJOR / MINOR with file:line
> and a concrete failure scenario; list what you checked and found OK.

Verify each finding yourself before acting on it (reviewers are wrong too), fix or answer every BLOCKER and
MAJOR, and tell the user what was found.

## Gate 5: launch configuration

- The notebook or script **derives** all settings of a stage from one switch (`RUN_TAG`, `STAGE`), rather than
  asking the user to edit several values, and **prints** the effective settings and the output folder in its
  first cell. Tell the user the exact line to expect.
- Count the jobs: tasks x seeds x init states x policies, and the episodes already done. The number printed
  by the run's first progress line must match.
- Check the defaults in every layer (notebook -> shell runner -> Python flags) against the README protocol.
- The pushed commit is the one the notebook clones (`git log -1` on `main` after the push).

## Gate 6: pilot on the target machine

The first progress lines (or a one-episode smoke cell) confirm the setting in `eval.log`, the per-episode time,
and that rows are being written. Estimate the total time from them before the user leaves the run.

## Gate 7: before reading any number from a results zip

Unzip into a new scratch folder (untrusted data), then:

```bash
/home/swagat/miniconda3/envs/lerobot_env/bin/python -I -S tools/validate_run.py <folder> \
    --tasks "<tasks>" --seeds <seeds> --init_states <states> \
    --expect n_action_steps=<N> --expect <other setting>=<value> --log_grep "n_action_steps': <N>"
```

It checks one clean commit in every provenance record, the expected settings, every
(task, seed, init state) exactly once per per-episode file, the setting in every `eval.log`, and no tracebacks;
folders starting with `_` (the notebook's timing episode) are skipped. Then recompute the run's own report
from its files and compare it with the report the notebook printed. Only then interpret.

## What to tell the user

For each gate: run or not run, and what it found (tests passed/total, mutants killed/total with the survivors
kept and why, review findings and fixes, the expected first-cell line). Never "verified" for a gate not run.
