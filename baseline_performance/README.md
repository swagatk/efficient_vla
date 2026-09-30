# Baseline Performance Evaluation

This folder contains the scripts for evaluating the baseline SmolVLA model on the LIBERO-10 benchmark.

## How to Run the Evaluation

To evaluate all 10 LIBERO tasks across 3 seeds (0, 1, 2) safely and reliably, use the `evaluate_all.py` script:

```bash
python evaluate_all.py
```

### Why use `evaluate_all.py`?
- **Process Isolation**: It launches each task and seed combination in its own completely isolated subprocess. This perfectly sidesteps known state-leakage and language-instruction caching bugs in `lerobot-eval` that cause 0% success rates when evaluating multiple tasks sequentially.
- **Resiliency**: If a specific task triggers a MuJoCo 3.10 EGL rendering segmentation fault (e.g., Task 2 often crashes `mjr_readPixels` when turning on the stove), the script will simply catch the crash, mark it as "Failed", and gracefully continue to the next task instead of aborting the 24-hour run.
- **Resumption**: If your evaluation stops or is interrupted, you can resume exactly where you left off by pointing it to the generated output directory:
  ```bash
  python evaluate_all.py --resume_dir=./outputs/evaluate_all_<TIMESTAMP>/
  ```

### Reports
When the script finishes or is interrupted, it will automatically parse all the JSON outputs from each subprocess and generate a clean `consolidated_report.csv` and a human-readable `consolidated_summary.txt` inside your specific output folder.

## File Information

- **`evaluate_all.py`**: The robust, multi-process evaluation orchestrator. You should use this.
- **`week1_repro_baseline_lock.sh`**: The original shell script from Week 1. It is kept here *only* as a historical reference/artifact to comply with the project roadmap and `AGENTS.md`. You do not need to execute this file.
- **`robosuite_mujoco_patch.py`**: *(Located in the repository root)*. This patch enables compatibility between Robosuite, MuJoCo 3.10, and PyTorch 2.6+. It **must** remain in the root directory because `evaluate_all.py` and other training scripts inject the root directory into `PYTHONPATH` so that this patch can be dynamically loaded via `sitecustomize.py` to fix memory corruptions. Do not move it into this folder!
