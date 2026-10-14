"""Evaluate base SmolVLA on LIBERO-10 with lerobot-eval, one isolated process per (task, seed).

Each run evaluates --n_episodes episodes. lerobot-eval ties episode i to LIBERO init state i, so every
seed sees init states 0..n_episodes-1; the seed changes the sampling noise and the env seed, not the
init states. Results go to <output_root>/task_<t>_seed_<s>/eval_info.json, plus consolidated_report.csv,
consolidated_summary.txt, per_episode.csv and provenance.jsonl in the output root.
"""
import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_command(task, seed, run_dir, lerobot_eval="lerobot-eval", policy_path="HuggingFaceVLA/smolvla_libero",
                  n_episodes=10, episode_length=520, observation_size=256):
    """lerobot-eval command for one (task, seed); the defaults are the protocol of results_300926."""
    return [
        lerobot_eval,
        f"--policy.path={policy_path}",
        "--policy.device=cuda",
        "--env.type=libero",
        "--env.task=libero_10",
        "--env.render_mode=rgb_array",
        "--env.max_parallel_tasks=1",
        f"--env.observation_height={observation_size}",
        f"--env.observation_width={observation_size}",
        f"--env.episode_length={episode_length}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        f"--eval.n_episodes={n_episodes}",
        f"--output_dir={run_dir}",
        f"--env.task_ids=[{task}]",
        f"--seed={seed}",
    ]


def run_dir_for(output_root, task, seed):
    return Path(output_root) / f"task_{task}_seed_{seed}"


def load_successes(output_root, task, seed):
    """Per-episode success list of a finished run, or None if the run has no eval_info.json."""
    path = run_dir_for(output_root, task, seed) / "eval_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [bool(x) for x in data["per_task"][0]["metrics"]["successes"]]


def generate_report(output_root, tasks, seeds, n_episodes=None):
    """Write consolidated_report.csv, consolidated_summary.txt and per_episode.csv; return missing runs.

    A run without eval_info.json (crashed, interrupted or not started) is listed as missing and left
    out of the means; it is never counted as a success or a failure. With n_episodes set, a run with a
    different episode count is treated as missing too.
    """
    output_root = Path(output_root)
    results, episodes, missing = [], [], []
    for t in tasks:
        for s in seeds:
            successes = load_successes(output_root, t, s)
            if successes is None or (n_episodes is not None and len(successes) != n_episodes):
                missing.append((t, s))
                continue
            results.append({"task_id": t, "seed": s, "success_rate": 100.0 * sum(successes) / len(successes)})
            # lerobot-eval with batch size 1 starts episode i from init state i.
            episodes += [{"task_id": t, "seed": s, "init_state": i, "success": int(x)} for i, x in enumerate(successes)]

    with open(output_root / "consolidated_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "seed", "success_rate"])
        writer.writeheader()
        writer.writerows(results)
    with open(output_root / "per_episode.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success"])
        writer.writeheader()
        writer.writerows(episodes)

    by_task = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["success_rate"])
    with open(output_root / "consolidated_summary.txt", "w") as f:
        f.write("=== Baseline Evaluation Consolidated Summary ===\n\n")
        overall_rates = []
        for t in sorted(by_task):
            rates = by_task[t]
            std = statistics.stdev(rates) if len(rates) > 1 else 0.0
            overall_rates.extend(rates)
            f.write(f"Task {t:2d} | Mean Success: {statistics.fmean(rates):6.2f}% | Std: {std:5.2f}% | Seeds: {len(rates)}\n")
        f.write("\n" + "-" * 40 + "\n")
        if overall_rates:
            f.write(f"Overall Mean Success: {statistics.fmean(overall_rates):.2f}%\n")
        if missing:
            f.write(f"\nINCOMPLETE: {len(missing)} of {len(tasks) * len(seeds)} runs missing, not included above:\n")
            f.write("".join(f"  task {t} seed {s}\n" for t, s in missing))
    return missing


def run_all(output_root, tasks, seeds, command_kwargs, jobs=1, gpus=None, launch_deadline_s=None,
            hard_stop_s=None, launch_delay_s=5.0, poll_s=5.0, should_stop=lambda: False, log=print):
    """Run every unfinished (task, seed) as its own process, at most ``jobs`` at a time.

    Runs that already have eval_info.json are skipped, so calling this again resumes. No new run starts
    after ``launch_deadline_s`` seconds; runs still going at ``hard_stop_s`` are killed and stay unfinished.
    GPUs in ``gpus`` are assigned round-robin through CUDA_VISIBLE_DEVICES. Returns {(task, seed): exit code}.
    """
    output_root = Path(output_root)
    pending = [(t, s) for t in tasks for s in seeds if load_successes(output_root, t, s) is None]
    log(f"{len(tasks) * len(seeds) - len(pending)} runs already finished, {len(pending)} to run, {jobs} at a time")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"  # loads robosuite_mujoco_patch (FAIL-003)
    start = time.monotonic()
    running, codes, launched = {}, {}, 0

    def elapsed():
        return time.monotonic() - start

    while pending or running:
        for key, (proc, log_f) in list(running.items()):
            if proc.poll() is not None:
                log_f.close()
                codes[key] = proc.returncode
                del running[key]
                status = "done" if proc.returncode == 0 and load_successes(output_root, *key) is not None else f"FAILED (exit {proc.returncode})"
                log(f"[{elapsed() / 60:6.1f} min] task {key[0]} seed {key[1]}: {status}")
        if hard_stop_s is not None and elapsed() >= hard_stop_s and running:
            for key, (proc, log_f) in running.items():
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
                log_f.close()
                codes[key] = proc.returncode
                log(f"[{elapsed() / 60:6.1f} min] task {key[0]} seed {key[1]}: killed at the hard stop, will rerun on resume")
            running.clear()
        stop_launching = should_stop() or (launch_deadline_s is not None and elapsed() >= launch_deadline_s) \
            or (hard_stop_s is not None and elapsed() >= hard_stop_s)
        if stop_launching and pending:
            log(f"[{elapsed() / 60:6.1f} min] not starting {len(pending)} remaining runs (deadline or interrupt)")
            pending = []
        while pending and len(running) < jobs:
            task, seed = pending.pop(0)
            run_dir = run_dir_for(output_root, task, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_env = dict(env)
            if gpus:
                run_env["CUDA_VISIBLE_DEVICES"] = str(gpus[launched % len(gpus)])
            log_f = open(run_dir / "eval.log", "w")
            proc = subprocess.Popen(build_command(task, seed, run_dir, **command_kwargs), stdout=log_f,
                                    stderr=subprocess.STDOUT, env=run_env, start_new_session=True)
            running[(task, seed)] = (proc, log_f)
            launched += 1
            log(f"[{elapsed() / 60:6.1f} min] task {task} seed {seed}: started"
                + (f" on GPU {run_env['CUDA_VISIBLE_DEVICES']}" if gpus else ""))
            if pending and len(running) < jobs:
                time.sleep(launch_delay_s)  # staggered starts under headless EGL (FAIL-006)
        if running:
            time.sleep(poll_s)
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume_dir", type=str, help="Output directory to create or resume. Default: outputs/evaluate_all_<timestamp>")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)), help="Tasks to evaluate (default: 0-9)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to evaluate (default: 0, 1, 2)")
    parser.add_argument("--n_episodes", type=int, default=10, help="Episodes per run = init states 0..n-1 (default: 10)")
    parser.add_argument("--policy_path", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--episode_length", type=int, default=520)
    parser.add_argument("--observation_size", type=int, default=256)
    parser.add_argument("--lerobot_eval", type=str, default="lerobot-eval", help="lerobot-eval executable")
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel (default: 1)")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids assigned round-robin, e.g. 0,1")
    parser.add_argument("--launch_deadline_min", type=float, default=None, help="Start no new run after this many minutes")
    parser.add_argument("--hard_stop_min", type=float, default=None, help="Kill unfinished runs after this many minutes")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["PYTHONFAULTHANDLER"] = "1"

    cfg_path = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))) / "config.yaml"
    if not cfg_path.exists():
        print(f"{cfg_path} not found. Write it first (benchmark_root, bddl_files, init_states, datasets, assets); "
              "the Kaggle and Colab notebooks show how.")
        sys.exit(1)

    output_root = Path(args.resume_dir or f"outputs/evaluate_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")

    sys.path.insert(0, str(REPO_ROOT))
    from vla_core.provenance import append_jsonl, collect

    append_jsonl(str(output_root / "provenance.jsonl"), collect(str(REPO_ROOT), vars(args)))

    interrupted = []

    def on_sigint(sig, frame):
        print("\n[!] Ctrl+C: no new runs will start; waiting for the running ones to finish...")
        interrupted.append(True)

    signal.signal(signal.SIGINT, on_sigint)

    command_kwargs = {"lerobot_eval": args.lerobot_eval, "policy_path": args.policy_path, "n_episodes": args.n_episodes,
                      "episode_length": args.episode_length, "observation_size": args.observation_size}
    run_all(
        output_root, args.tasks, args.seeds, command_kwargs, jobs=args.jobs,
        gpus=[g for g in args.gpus.split(",") if g.strip()],
        launch_deadline_s=None if args.launch_deadline_min is None else 60 * args.launch_deadline_min,
        hard_stop_s=None if args.hard_stop_min is None else 60 * args.hard_stop_min,
        should_stop=lambda: bool(interrupted), log=lambda m: print(m, flush=True),
    )

    missing = generate_report(output_root, args.tasks, args.seeds, args.n_episodes)
    print(f"\nReport: {output_root / 'consolidated_summary.txt'}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} runs missing; run again with --resume_dir {output_root} to finish them.")
        sys.exit(2)


if __name__ == "__main__":
    main()
