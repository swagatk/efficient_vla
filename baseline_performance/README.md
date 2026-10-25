# Baseline Performance Evaluation

This folder contains the scripts for evaluating the baseline SmolVLA model on the LIBERO-10 benchmark.

## How to Run the Evaluation

To evaluate all 10 LIBERO tasks across 3 seeds (0, 1, 2) safely and reliably, use the `evaluate_all.py` script:

```bash
python evaluate_all.py
```

### What the protocol measures

`lerobot-eval` starts episode *i* from LIBERO init state *i*, so with the default 10 episodes every seed
evaluates init states 0-9 of each task; the seed changes only the sampling noise and the environment seed.
`results_300926` therefore covers 10 of the 50 init states per task, and its std over seeds does not include
the variation between init states. `results_300926/per_episode.csv` lists every episode with its init state.

### Options

- `--n_episodes N`: episodes per (task, seed), i.e. init states 0..N-1 (default 10; 50 covers every init state).
- `--jobs N` and `--gpus 0,1`: run N evaluations at once, assigning GPUs round-robin. Each process needs
  about 4.5 GB of RAM and 2.7 GB of GPU memory.
- `--launch_deadline_min`, `--hard_stop_min`: stop starting new runs, and kill unfinished ones, after that
  many minutes (for time-limited sessions). Killed runs are rerun on resume.
- `--policy_path`, `--episode_length`, `--observation_size`, `--lerobot_eval`: defaults are the protocol above.
- `--n_action_steps N`: override the checkpoint's value. `--rename_map JSON`: map LIBERO observation keys
  onto a checkpoint with other camera names. Both are left out of the command unless given, so the
  default command is unchanged. `lerobot/smolvla_libero` needs `--n_action_steps 1 --rename_map
  '{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'`.
- `--order seed`: run all tasks of one seed before the next seed (default `task`: all seeds of one task first).

Each run appends a record to `provenance.jsonl` (git commit and dirty flag, lerobot, torch, MuJoCo and
robosuite versions, GPUs, render backend, arguments). Runs that crashed or did not finish are listed as
INCOMPLETE in `consolidated_summary.txt` and left out of the means, and the script exits with code 2.

`kaggle_baseline_lerobot_eval.ipynb` runs this protocol on Kaggle (2 x T4) for either checkpoint, or for a
checkpoint from `continued_training/` uploaded as a Kaggle dataset (`CHECKPOINT` in its first cell) and compares it per task and per init state with `results_300926` and with the external
LIBERO-CTRL reference in [`reference_libero_ctrl/`](reference_libero_ctrl/).

### Why `results_300926` is below the published SmolVLA number

The SmolVLA paper reports 71% on LIBERO-Long; `results_300926` measures 40.67%. Two findings so far:

- `HuggingFaceVLA/smolvla_libero`, the checkpoint used here, is not the checkpoint LIBERO-CTRL evaluated
  (`lerobot/smolvla_libero`); they differ in weights and architecture (see `reference_libero_ctrl/README.md`).
- LIBERO-CTRL measured `lerobot/smolvla_libero` at 56.8% on LIBERO-10 (284/500), 59/100 on init states 0-9,
  and reports that this released checkpoint was trained on one eighth of the paper's sample budget.

So part of the gap is the released checkpoints. Whether our evaluation path also loses performance was
tested by running `lerobot/smolvla_libero` through it (the notebook's default, Kaggle 2 x T4, commit 82df15a,
clean tree): [`week6/smolvla_libero_lerobot_eval_kaggle/`](../week6/smolvla_libero_lerobot_eval_kaggle/),
comparison in `comparison_with_results_300926_and_libero_ctrl.txt` there.

- `lerobot/smolvla_libero`: 171/300 (57.00%; per seed 58, 58, 55 of 100) on init states 0-9, seeds 0-2.
- Against LIBERO-CTRL on the same init states (59/100): exact McNemar p = 1.00, 1.00 and 0.45 for seeds 0, 1, 2.
  Our evaluation path reproduces the external measurement; it does not lose performance.
- Against `results_300926` on the same episodes: 171/300 vs 122/300, exact McNemar p < 0.0001 (34 episodes
  succeed only with the old checkpoint, 83 only with the new one).

This paragraph first concluded that the rest of the gap was the checkpoint's smaller training budget. **That
was wrong** (October 2026): the rest of the gap was `n_action_steps=1`, the setting we took over from
LIBERO-CTRL. With 10 actions executed per inference and nothing else changed, `lerobot/smolvla_libero` scores
216/300 (72.0%; per seed 73, 71, 72 of 100) against 171/300 on the same episodes, exact McNemar p < 0.0001
(62 episodes succeed only at 10, 17 only at 1), and no task is lower; 50 actions per inference is worse than
10 ([`week6/noise_reachability_step0_kaggle/`](../week6/noise_reachability_step0_kaggle/)). Our evaluation
path is sound; the setting was not.

`HuggingFaceVLA/smolvla_libero` is still the weaker checkpoint. The baseline for new methods is
`lerobot/smolvla_libero` with `--n_action_steps 10`, 216/300 (72.0%). Comparisons made at `n_action_steps=1`
(171/300) hold for that setting only.

### Reproduction on Kaggle (October 2026)

The full protocol (10 tasks x seeds 0-2 x init states 0-9) was rerun on Kaggle with the notebook above at
commit 74ad990 (clean tree), 2 x Tesla T4, lerobot 0.6.2 (b6ec0060), torch 2.11.0, MuJoCo 3.13.0, EGL, 4 runs at
a time, in two sessions (25 runs, then the remaining 5 on resume). Results:
[`week6/baseline_lerobot_eval_kaggle/`](../week6/baseline_lerobot_eval_kaggle/), comparison:
[`comparison_with_results_300926.txt`](../week6/baseline_lerobot_eval_kaggle/comparison_with_results_300926.txt),
produced by `python -S compare_per_state.py results_300926/per_episode.csv
../week6/baseline_lerobot_eval_kaggle/per_episode.csv`.

- Overall: Kaggle 131/300 (43.67%) vs local 122/300 (40.67%); paired exact McNemar p = 0.34. Per task the
  difference is at most 3 of 30 episodes.
- Per init state: 16 of 100 (task, init state) pairs were solved on one machine and never on the other; a
  permutation test gives p = 0.89, so this is what sampling noise alone produces.

The baseline therefore reproduces across machines through `lerobot-eval`. This does not explain the task 0
difference between Colab and local runs of the custom harness (`week6/task0_base_noise_seeds_colab/`), which
used a different machine and a different evaluation path.

### Why use `evaluate_all.py`?
- **Process Isolation**: It launches each task and seed combination in its own completely isolated subprocess. This perfectly sidesteps known state-leakage and language-instruction caching bugs in `lerobot-eval` that cause 0% success rates when evaluating multiple tasks sequentially.
- **Resiliency**: If a specific task triggers a MuJoCo 3.10 EGL rendering segmentation fault (e.g., Task 2 often crashes `mjr_readPixels` when turning on the stove), the script will simply catch the crash, report the run as missing, and continue with the next task instead of aborting the 24-hour run.
- **Resumption**: If your evaluation stops or is interrupted, you can resume exactly where you left off by pointing it to the generated output directory:
  ```bash
  python evaluate_all.py --resume_dir=./outputs/evaluate_all_<TIMESTAMP>/
  ```

### Reports
When the script finishes or is interrupted, it will automatically parse all the JSON outputs from each subprocess and generate a clean `consolidated_report.csv`, a human-readable `consolidated_summary.txt` and `per_episode.csv` (task, seed, init state, success) inside your specific output folder.

## File Information

- **`evaluate_all.py`**: The robust, multi-process evaluation orchestrator. You should use this.
- **`week1_repro_baseline_lock.sh`**: The original shell script from Week 1. It is kept here *only* as a historical reference/artifact to comply with the project roadmap and `AGENTS.md`. You do not need to execute this file.
- **`robosuite_mujoco_patch.py`**: *(Located in the repository root)*. This patch enables compatibility between Robosuite, MuJoCo 3.10, and PyTorch 2.6+. It **must** remain in the root directory because `evaluate_all.py` and other training scripts inject the root directory into `PYTHONPATH` so that this patch can be dynamically loaded via `sitecustomize.py` to fix memory corruptions. Do not move it into this folder!
