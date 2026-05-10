# AGENTS.md

## Project Scope

This repository is an experiment log and code workspace for improving SmolVLA on LIBERO-10.

- Week 1 locked a reproducible SmolVLA baseline.
- Week 2 added GroundingDINO visual boxes and text hints for evaluation and observed degradation.
- Week 3 swept visual-prompt hyperparameters and selected the best wrapper settings.
- Week 4 shifted focus to lightweight online RL with a residual policy instead of further prompt-wrapper optimization.

Read [plan.md](plan.md) first for the intended 12-week roadmap. Use the weekly summary folders (`week1/`, `week2/`, `week3/`, `week4/`) as result artifacts, not as source code.

## Key Entry Points

- [train_online_rl.py](train_online_rl.py): main residual RL training loop on one LIBERO task (`--task_id 0-9`).
- [rl_residual_agent.py](rl_residual_agent.py): frozen SmolVLA plus trainable residual actor and critic.
- [visual_prompt_wrapper.py](visual_prompt_wrapper.py): GroundingDINO and FastSAM overlay plus text-hint injection.
- [eval_week2_visual_prompting.py](eval_week2_visual_prompting.py): evaluation wrapper that injects visual prompting into LeRobot eval.
- [week1_repro_baseline_lock.sh](week1_repro_baseline_lock.sh): baseline reproduction script.
- [week2_visual_prompting_3seed.sh](week2_visual_prompting_3seed.sh): week-2 evaluation runs.
- [week3_stage1_performance_sweep.sh](week3_stage1_performance_sweep.sh): week-3 prompt sweep.
- [week4_stage2_eval.sh](week4_stage2_eval.sh): best visual-prompt configuration eval.

## Working Conventions

- Treat `wandb/`, `checkpoints_task_*/`, `week*/`, and `outputs/` as generated artifacts unless the task is explicitly about analysis.
- Keep edits narrow. This repo is an experiment workspace, so preserve command-line flags, logging fields, and checkpoint shapes unless the task requires a migration.
- Prefer linking to existing docs and scripts instead of duplicating run instructions in code comments.

## Environment Assumptions

- Expected Python stack includes `lerobot==0.4.0`, `libero`, `robosuite`, `groundingdino-py`, `ultralytics`, and `wandb`.
- Headless eval and training assume `MUJOCO_GL=egl` and usually `PYOPENGL_PLATFORM=egl`.
- LIBERO expects `LIBERO_CONFIG_PATH/config.yaml`; the shell scripts and [eval_week2_visual_prompting.py](eval_week2_visual_prompting.py) can generate it.
- GroundingDINO weights are expected under `~/model_weights/groundingdino_swint_ogc.pth` and the config file in this repo.
- `train_online_rl.py` patches `torch.load` for PyTorch 2.6+ compatibility with LIBERO init-state loading.

## Common Commands

Baseline reproduction:

```bash
bash week1_repro_baseline_lock.sh
```

Visual prompting eval:

```bash
bash week2_visual_prompting_3seed.sh
```

Prompt sweep:

```bash
bash week3_stage1_performance_sweep.sh
```

Best week-3 config eval:

```bash
bash week4_stage2_eval.sh
```

Residual RL training on a single task:

```bash
python train_online_rl.py --task_id 0
```

## RL Investigation Workflow

If the task is about why residual RL is not improving performance, inspect these areas in order before widening scope:

1. [train_online_rl.py](train_online_rl.py): reward shaping, rollout storage, PPO update inputs, and success logging.
2. [rl_residual_agent.py](rl_residual_agent.py): what observations the residual head actually consumes and whether the update path matches rollout-time inputs.
3. [visual_prompt_wrapper.py](visual_prompt_wrapper.py): whether RL training is still using week-2 or week-3 prompting even though the wrapper path was deprioritized.
4. Existing W&B logs under `wandb/` and checkpoints under `checkpoints_task_*/` for evidence of reward trends, entropy collapse, or resume behavior.

When reviewing RL behavior, verify these assumptions explicitly rather than assuming they are correct:

- `done` means true task success rather than time-limit termination.
- PPO updates are replaying the same observation modalities used during rollout.
- The reward scale is compatible with the BC anchor and PPO losses.
- The residual head receives the information needed to correct base-policy failures.
- Visual prompting is intentional in RL training rather than an inherited week-2 default.

## Notes For Future Agents

- Prefer [plan.md](plan.md) and the weekly shell scripts for experiment intent and reproducibility.
- Ignore live W&B file churn unless the task is log analysis; the repo may be dirty because training is running or was interrupted.
- If changing evaluation or training protocol, update the relevant shell script or the root instruction file instead of scattering protocol notes across Python files.