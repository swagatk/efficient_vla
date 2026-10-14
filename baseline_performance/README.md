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

Each run appends a record to `provenance.jsonl` (git commit and dirty flag, lerobot, torch, MuJoCo and
robosuite versions, GPUs, render backend, arguments). Runs that crashed or did not finish are listed as
INCOMPLETE in `consolidated_summary.txt` and left out of the means, and the script exits with code 2.

`kaggle_baseline_lerobot_eval.ipynb` runs this protocol on Kaggle (2 x T4) and compares it per task and per
init state with `results_300926`.

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
