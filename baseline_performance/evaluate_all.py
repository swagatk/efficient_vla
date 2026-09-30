import os
import sys
import json
import subprocess
import argparse
import signal
from pathlib import Path
import csv
import statistics
import math
from datetime import datetime

# Graceful interruption handler
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print("\n[!] Ctrl+C detected! The script will stop after the current evaluation finishes...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def generate_report(output_root, tasks, seeds):
    """Generates a consolidated CSV and TXT report from all eval_info.json files."""
    results = []
    
    for t in tasks:
        for s in seeds:
            eval_info_path = output_root / f"task_{t}_seed_{s}" / "eval_info.json"
            if not eval_info_path.exists():
                continue
            try:
                with open(eval_info_path, 'r') as f:
                    data = json.load(f)
                
                overall = data.get("overall", {})
                success_rate = overall.get("pc_success", 0.0)
                
                results.append({
                    "task_id": t,
                    "seed": s,
                    "success_rate": success_rate,
                })
            except Exception as e:
                print(f"Error reading {eval_info_path}: {e}")

    if not results:
        return
        
    # Write CSV
    csv_path = output_root / "consolidated_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "seed", "success_rate"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Write TXT summary
    txt_path = output_root / "consolidated_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Baseline Evaluation Consolidated Summary ===\n\n")
        
        # Aggregate by task
        by_task = {}
        for r in results:
            by_task.setdefault(r["task_id"], []).append(r["success_rate"])
            
        overall_rates = []
        for t in sorted(by_task.keys()):
            rates = by_task[t]
            mean = statistics.fmean(rates)
            std = statistics.stdev(rates) if len(rates) > 1 else 0.0
            overall_rates.extend(rates)
            
            f.write(f"Task {t:2d} | Mean Success: {mean:6.2f}% | Std: {std:5.2f}% | Seeds: {len(rates)}\n")
            
        f.write("\n")
        f.write("-" * 40 + "\n")
        if overall_rates:
            overall_mean = statistics.fmean(overall_rates)
            f.write(f"Overall Mean Success: {overall_mean:.2f}%\n")
            
    print(f"\nConsolidated report generated at: {output_root}")
    print(f" - CSV: {csv_path}")
    print(f" - TXT: {txt_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline across all tasks and seeds.")
    parser.add_argument("--resume_dir", type=str, help="Directory to resume evaluation from. If not provided, creates a new one.")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)), help="Tasks to evaluate (default: 0-9)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to evaluate (default: 0, 1, 2)")
    args = parser.parse_args()

    # Environment setup
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")
    os.environ["PYTHONFAULTHANDLER"] = "1"
    
    # Configure Libero
    libero_config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    libero_config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = libero_config_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"Please run the bash script at least once or set up {cfg_path} properly.")
        sys.exit(1)

    if args.resume_dir:
        output_root = Path(args.resume_dir)
        if not output_root.exists():
            print(f"Resume directory {output_root} does not exist.")
            sys.exit(1)
        print(f"Resuming evaluation in: {output_root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(f"outputs/evaluate_all_{timestamp}")
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"Starting new evaluation in: {output_root}")

    total_runs = len(args.tasks) * len(args.seeds)
    completed = 0

    print(f"Total combinations to evaluate: {total_runs}")
    print("-" * 50)

    for task in args.tasks:
        for seed in args.seeds:
            if interrupted:
                print("\nInterruption requested. Stopping evaluation loop.")
                generate_report(output_root, args.tasks, args.seeds)
                sys.exit(0)
                
            run_dir = output_root / f"task_{task}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this run is already completed
            eval_info = run_dir / "eval_info.json"
            if eval_info.exists():
                print(f"[{completed+1}/{total_runs}] Skipping Task {task}, Seed {seed} (Already completed)")
                completed += 1
                continue
                
            print(f"[{completed+1}/{total_runs}] Evaluating Task {task}, Seed {seed} ... ", end="", flush=True)
            
            log_file = run_dir / "eval.log"
            
            cmd = [
                "lerobot-eval",
                "--policy.path=HuggingFaceVLA/smolvla_libero",
                "--policy.device=cuda",
                "--env.type=libero",
                "--env.task=libero_10",
                "--env.render_mode=rgb_array",
                "--env.max_parallel_tasks=1",
                "--env.observation_height=256",
                "--env.observation_width=256",
                "--env.episode_length=520",
                "--eval.batch_size=1",
                "--eval.use_async_envs=false",
                "--eval.n_episodes=10",
                f"--output_dir={run_dir}",
                f"--env.task_ids=[{task}]",
                f"--seed={seed}"
            ]
            
            env_vars = os.environ.copy()
            repo_root = Path(__file__).resolve().parent.parent
            env_vars["PYTHONPATH"] = f"{repo_root}:{env_vars.get('PYTHONPATH', '')}"

            try:
                with open(log_file, "w") as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env_vars, check=True)
                print("Done ✓")
            except subprocess.CalledProcessError:
                print("Failed ✗ (Check logs)")
            except KeyboardInterrupt:
                print("\n[!] Ctrl+C detected during subprocess! The current evaluation was aborted.")
                # We do not generate report immediately here, we will exit the loop via `interrupted` flag
                break
                
            completed += 1

    print("\nEvaluation loop finished.")
    generate_report(output_root, args.tasks, args.seeds)

if __name__ == "__main__":
    main()
