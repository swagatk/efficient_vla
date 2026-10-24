# Continued training on LIBERO-10, Colab training record

Training run behind `week6/continued_training_kaggle/` (result and discussion there). Notebook
`continued_training/colab_continued_training.ipynb`, script `continued_training/run_continued_training.sh`,
at commit `643773a` with a clean tree (`run_seed0/git_commit.txt`, empty `run_seed0/git_dirty.txt`), on an
NVIDIA A100-SXM4-40GB (`run_seed0/versions.txt`: lerobot 0.6.2 at `b6ec006`, torch 2.11.0).

- **Plan** (`run_seed0/plan.json`): 0.537 s per step in the 100-step speed trial; 25,000 steps at batch 32,
  checkpoints every 2,500 steps. The notebook first planned 20,000 steps (`TRAIN_HOURS = 3.0`); the budget
  was raised to 25,000 and the run restarted from scratch within minutes, before any evaluation.
- **Training** (`run_seed0/train.log.gz`): peak learning rate 5e-5 after 500 warm-up steps, cosine decay to
  2.5e-6 over 25,000 steps (`run_seed0/final_train_config.json`). Training loss 0.112 at step 600 and 0.046
  at step 25,000.
- **Interruption:** the runtime was lost when the laptop running the browser was closed, after the step-15,000
  checkpoint had been mirrored to Drive. On a new runtime the run restored that checkpoint and continued from
  step 15,000 (`run_seed0/runner.log`; lerobot: "Resuming data order at epoch 4"). The mirrored `last_step`
  file still said 012500, so the restored folder was named `012500`; its `training_step.json` said 15000 and
  all its files were written within about a minute of each other at the step-15,000 save. The log has no
  lines between step 12,600 and 15,000, because the log on Drive was last copied at the step-12,500 sync.
- **Rollout check** (`rollout_check/`): one episode of the final checkpoint (task 3, seed 0, init state 0)
  failed. It only checks that the checkpoint loads and runs; the Kaggle run gives the same outcome for this
  episode (the base succeeded on it).
