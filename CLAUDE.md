# CLAUDE.md

Research workspace for improving SmolVLA on LIBERO-10. Project history, entry points, environment assumptions and the RL investigation workflow live in AGENTS.md, imported here:

@AGENTS.md

Some paths in AGENTS.md are stale. The current locations are:

- Hybrid diffusion: `hybrid_diffusion/train_hybrid_diffusion.py`, `hybrid_diffusion/hybrid_diffusion_agent.py`
- Residual RL: `rl_based_tuning/train_online_sac.py`, `rl_based_tuning/sac_residual.py`, run via `rl_based_tuning/run_sac_residual.sh` (see `rl_based_tuning/README.md`). `rl_based_tuning/train_online_ppo.py` is retired. The root `sac_residual_agent.py` and `ppo_residual_agent.py` are pre-fix code kept only for `reproduce_task.py`; there is no `train_online_rl.py` or `rl_residual_agent.py`

The rest of this file is about how to write, verify and commit code here.

## Environment

- Interpreter: `/home/swagat/miniconda3/envs/lerobot_env/bin/python` (conda env `lerobot_env`). Use it explicitly; the system `python3` lacks the stack.
- Run everything from the repo root, or with the repo root on `PYTHONPATH`. `sitecustomize.py` loads `robosuite_mujoco_patch.py` only when the root is importable, and without that patch MuJoCo 3 corrupts memory (FAIL-003).
- Headless runs need `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.
- WSL2 with one GPU. Check `nvidia-smi` before launching anything on the GPU; a training run may already be using it.

## Developing a new idea

1. **Write down the hypothesis first.** One or two sentences in the idea's README: what should change, on which tasks, by how much, against which baseline number. An experiment with no stated expectation cannot fail, so it cannot teach anything.
2. **Give the idea its own directory** (as `hybrid_diffusion/`, `Gated_Residual_strategy/` and `supervised_task_specific_lora_adapters/` do), with a `README.md`, the training and eval entry points, and a `run_*.sh` that records the exact command used for reported numbers.
3. **Reuse before writing.** Search for an existing implementation before adding env construction, observation preprocessing, checkpoint I/O, or an eval loop. See "Modularity" below.
4. **Reproduce the baseline in the same harness** before measuring the new method. If the baseline number through your eval path does not match `baseline_performance/results_300926/`, the harness is broken, not the idea (FAIL-006 was exactly this).
5. **Go smallest to largest.** Shape/unit tests on CPU, then overfit a single batch, then one task with few episodes, then the full 10-task, 3-seed run. Do not launch the expensive step while a cheaper one is still failing.
6. **Record negative results.** A method that did not help gets a short write-up in its weekly folder with the numbers, the same as one that did.

## Modularity

The repo has grown by copying scripts, so the same logic exists in several places: the `torch.load` `weights_only` patch, the `OffScreenRenderEnv` evaluation pipeline (init states, 10-step stabilisation, delta control, image flip, pre/post-processing) in `test_base.py`, `supervised_task_specific_lora_adapters/eval_lora_policy.py` and `hybrid_diffusion/train_hybrid_diffusion.py`, and the frozen-SmolVLA wrapper in each agent file. Divergent copies of the eval pipeline have already caused a 0% success regression.

Rules for new code:

- **Shared code goes in `vla_core/`**, an importable package at the repo root. Create it the first time something needs to be shared. Expected modules: `envs.py` (LIBERO env construction and reset), `evaluation.py` (the canonical rollout/eval loop), `policy.py` (loading and freezing the SmolVLA base, including the `eval` recursion guard), `compat.py` (the `torch.load` patch), `seeding.py`.
- **Extract on second use.** When a new idea needs logic that already lives in another script, move it into `vla_core/` and import it from both places. Do not copy it a second time.
- **Do not refactor working experiment code speculatively.** Migrate an old script onto `vla_core/` only when you are already changing it, and confirm its numbers are unchanged afterwards.
- **Separate the three layers.** Model and agent classes (`nn.Module`, no env, no W&B, no argparse) → training/eval functions that take explicit arguments → a thin `main()` that parses flags and calls them. Anything a test must import has to be importable without loading a checkpoint, building an env, or touching the GPU.
- **No import-time side effects.** Model loading, env creation and `wandb.init` belong under `if __name__ == "__main__":` or inside functions.
- **No hard-coded hyperparameters or paths inside functions.** Thread them through arguments or a config dataclass and expose them as CLI flags with the current value as the default.
- **Keep existing interfaces stable.** Preserve CLI flags, W&B logging field names and checkpoint key/shape layouts unless the task is explicitly a migration; old checkpoints and old runs have to stay comparable.

## Testing

Tests live in `tests/` and run with pytest:

```bash
/home/swagat/miniconda3/envs/lerobot_env/bin/python -m pytest tests/ -q
```

Always pass the `tests/` path. The root-level `test_*.py` files (`test_base.py`, `test_lang.py`, …) are GPU evaluation scripts that load SmolVLA at import time; a bare `pytest` would collect and run them.

- **Default tests are fast and CPU-only**: no checkpoint download, no simulator. Use small randomly initialised modules and synthetic tensors.
- Mark anything that needs the GPU, the real SmolVLA checkpoint, or LIBERO with `@pytest.mark.slow` and run those deliberately with `-m slow`.
- Every new module under `vla_core/` and every new agent/model class gets tests. Worth covering for this kind of code:
  - output shapes and dtypes for the action chunk, including the bfloat16 base policy path;
  - frozen parameters stay frozen: `requires_grad` is false and the weights are bit-identical after an optimiser step;
  - gradients do reach every parameter that is supposed to train;
  - a zero-initialised or disabled residual/adapter reproduces the base policy's action exactly;
  - checkpoint save → load round-trips to identical outputs;
  - the same seed gives the same result twice;
  - the model can overfit one small batch (loss falls by a large factor).
- A bug fix comes with a test that fails before the fix. If the bug cost real debugging time, also record it (see "Recording failures").
- Simulator tests must respect FAIL-004: at most 3 environment resets per Python process under headless EGL.

## Verification before commit and push

Nothing is committed or pushed until all of the following have been run on the code as it will be committed, and their real output has been read:

1. Invariant checker passes:
   ```bash
   bash .gemini/skills/vla_research_validator/scripts/check_invariants.sh
   ```
2. `pytest tests/ -q` passes.
3. Every changed entry point has actually been executed on a short smoke run (one task, minimal steps/episodes), not just compiled. For anything touching the model forward pass, observation pipeline or eval loop, that smoke run is a real rollout.
4. If the change could affect reported numbers, the baseline has been re-evaluated through the changed path and still matches.

If a step cannot be run (GPU busy, long run in progress), say so explicitly and do not describe the change as verified. Never weaken, skip or delete a failing check to get a commit through; fix the cause or report it.

Git rules:

- Commit and push only when asked. Pushing to `origin` is a separate step from committing and needs its own go-ahead.
- Stage files by name. Never `git add -A` or `git add .`: the working tree routinely contains W&B churn, `__pycache__`, LaTeX build products, checkpoints and probe output.
- Do not commit generated artifacts (`wandb/`, `checkpoints*/`, `outputs/`, `*.pyc`, `latex/*.aux|log|fls|fdb_latexmk|synctex.gz`) unless the task is to publish results.
- One logical change per commit. The message states what changed and, for experiment code, what was run to verify it.

## Experiment and reporting standards

- **Seeds.** Every training and eval entry point takes `--seed` and seeds Python, NumPy, PyTorch and the environment. Reported comparisons use at least 3 seeds.
- **Same protocol on both sides.** Baseline and method are compared on identical tasks, init states, episode counts, horizon and control mode. Residual evaluation keeps `--eval_replan_each_step 1` (FAIL-005).
- **Report what was measured.** Give mean ± std over seeds together with the episode count. With 10 episodes per task, one episode is 10 percentage points, so do not call a difference that small an improvement.
- **Provenance.** Each run logs its full config, the git commit hash, and whether the tree was dirty, to W&B and next to the checkpoint.
- **Never tune on what you report.** Hyperparameters are chosen on separate seeds or episodes from the ones in the final table.
- **Numbers in docs and the paper come from files.** Any figure quoted in a README, weekly summary or `latex/` must trace to a result artifact in the repo; never fill in a number from memory or expectation.
- When success drops to 0% or a run segfaults, read `.gemini/skills/vla_research_validator/FAILURES.md` before debugging; most past causes were harness problems, not model problems.

## Recording failures

When a failure took real effort to diagnose, add it to `.gemini/skills/vla_research_validator/FAILURES.md` (symptom, root cause, resolution, invariant) using `scripts/record_failure.py` in that skill, and where the invariant can be checked mechanically, add a check to `check_invariants.sh` or a test under `tests/`.
