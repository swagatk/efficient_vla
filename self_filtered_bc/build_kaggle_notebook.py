"""Build self_filtered_bc/kaggle_self_filtered_bc.ipynb; the hardware/code/environment/resume cells are taken
verbatim from rl_based_tuning/kaggle_sac_residual.ipynb so both notebooks install the same stack."""
import json
import sys

REPO = "/home/swagat/GIT/efficient_vla"
sac = json.load(open(f"{REPO}/rl_based_tuning/kaggle_sac_residual.ipynb"))
src = lambda i: "".join(sac["cells"][i]["source"])  # noqa: E731

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md('''
# Self-filtered BC on SmolVLA, LIBERO-10 task 6 (Kaggle, 2 x T4)

Pilot of `self_filtered_bc/` (see its README for the hypothesis): a LoRA adapter on SmolVLA's action expert,
trained with SmolVLA's own flow-matching loss on the base policy's **successful** rollouts from training init
states 10-49, evaluated on held-out init states 0-9 x seeds 0, 1, 2, paired with the base episodes of the
residual-SAC harness check (`week6/sac_residual_task6_pilot1/eval_base/`, 15/30).

Set `STAGE` in step 1, then **Save Version > Save & Run All (Commit)** (GPU T4 x2, Internet on):

1. `"verify"` (about 1 h): unit tests; a smoke run of all three stages (1 short episode, 3 training steps at
   the real batch size, which reports peak GPU memory and seconds per step, 1 short evaluation); and a
   **harness check**: the base policy through this evaluation path on init states 0-9, seed 0, compared with
   the same 10 episodes of pilot 1's base evaluation. Stop if they disagree.
2. `"collect"` (about 4-6 h, an estimate from pilot timings): the base policy on init states 10-49 x `COLLECT_SEEDS` (120 episodes), 4
   processes in parallel. Saves about 5 GB of episodes in the output. If the session ends early, add this
   version's output as input and run again: finished episodes are skipped.
3. `"train"` (about 1-2 h per run): one LoRA adapter per entry of `TRAIN_RUNS`, one per GPU.
4. `"eval"`: each adapter against the base on the held-out episodes, with an exact McNemar test.

Each stage after the first needs the previous version's output as input (**Add Input > Your Work >** this
notebook), so the stages chain through versions.
''')

md("## 1. Settings")
code('''
import csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

STAGE = "verify"                   # "verify", "collect" (repeat until done), "train", "eval"

REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "main"
REPO = "/tmp/efficient_vla"        # code and venv live outside /kaggle/working, so they are not saved as output
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"

OUT_NAME = "self_filtered_bc_task6"
OUT_ROOT = f"/kaggle/working/{OUT_NAME}"  # episodes, adapters and results, kept as the version's output
DATA = f"{OUT_ROOT}/data"

TASK_ID = 6
COLLECT_SEEDS = "0-2"              # episode seeds per training init state
INIT_STATE_PARTS = ["10-19", "20-29", "30-39", "40-49"]   # one collection process per part
VAL_INIT_STATES = "45-49"          # held out of training for the validation loss (hyperparameter choice)
TRAIN_RUNS = [("success", 0)]      # (selection, seed); later: ("success", 1), ("success", 2), ("all_matched", 0), ...
TRAIN_STEPS = 2000
BATCH_SIZE = 16
TRAIN_EXTRA = ""                   # e.g. "--learning_rate 5e-5 --lora_r 8"
EVAL_SEEDS = [0, 1, 2]
EVAL_INIT_STATES = "0-9"
GB_PER_JOB = 6                     # base policy + one episode in memory per collection/evaluation process

# Base policy: lerobot/smolvla_libero at the revision evaluated in week6/smolvla_libero_lerobot_eval_kaggle.
POLICY_REPO, POLICY_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def sh(cmd, check=True, env=None):
    # Capture and print: notebooks do not show output that a subprocess writes directly.
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {r.returncode}: {cmd}")
    return r
''')

md("## 2. Hardware")
code(src(4))
md("## 3. Code")
code(src(6))
md(src(7))
code(src(8))

md("## 5. Sanity checks, tests and the base-policy snapshot")
code('''
RUN_ENV = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", PYTHONPATH=REPO, PYTHONFAULTHANDLER="1",
               WANDB_MODE="offline")

sh(f"""{PY} -c "
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
" """, env=RUN_ENV)
egl = sh(f"{PY} -c \\"import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('EGL context ok')\\"", check=False, env=RUN_ENV)
assert "EGL context ok" in egl.stdout, "EGL rendering failed; this notebook needs GPU rendering"
sh(f"cd {REPO} && {PY} -m pytest tests/ -q -p no:cacheprovider", env=RUN_ENV)

# Pinned base-policy snapshot, downloaded once; every process loads it from this folder.
r = sh(f"""{PY} -c "from huggingface_hub import snapshot_download; print(snapshot_download('{POLICY_REPO}', revision='{POLICY_REVISION}'))" """, env=RUN_ENV)
POLICY_PATH = r.stdout.strip().splitlines()[-1]
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"
BASE_ARGS = f"--task_id {TASK_ID} --policy_path {POLICY_PATH}"   # n_action_steps=1 and the rename map are defaults
# Base episodes of the residual-SAC harness check: same snapshot, settings, episode seeds and T4 GPUs.
BASE_CSV = f"{REPO}/week6/sac_residual_task6_pilot1/eval_base/per_episode_base.csv"
os.makedirs(OUT_ROOT, exist_ok=True)
print("base policy:", POLICY_PATH)
''')

md("## 6. Helpers")
code('''
def run_jobs(jobs, max_parallel, poll_s=60, log_every_s=600):
    """Run {name: (command, gpu)} with at most max_parallel at once. Exit code 3 means 'call again' (more
    episodes to run in a fresh process) and the command is restarted; returns {name: final exit code}."""
    pending, running, codes, last_log = list(jobs), {}, {}, time.time()
    while pending or running:
        while pending and len(running) < max_parallel:
            name = pending.pop(0)
            cmd, gpu = jobs[name]
            log = open(f"{OUT_ROOT}/logs/{name}.log", "a")
            env = dict(RUN_ENV, CUDA_VISIBLE_DEVICES=str(gpu))
            running[name] = (subprocess.Popen(cmd, shell=True, executable="/bin/bash", env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True), log)
            print(time.strftime("%H:%M"), "started", name, "on GPU", gpu, flush=True)
            time.sleep(5)  # staggered EGL start-up (FAIL-006)
        for name, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            if proc.returncode == 3:
                pending.insert(0, name)
                continue
            codes[name] = proc.returncode
            print(time.strftime("%H:%M"), name, "finished with exit code", proc.returncode, flush=True)
        if time.time() - last_log > log_every_s:
            last_log = time.time()
            for name in running:
                tail = [l for l in open(f"{OUT_ROOT}/logs/{name}.log").read().splitlines() if l.startswith("[")][-1:]
                print(time.strftime("%H:%M"), name, tail, flush=True)
        time.sleep(poll_s)
    return codes


FIELDS = ["task_id", "seed", "init_state", "success", "steps"]


def read_csv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def write_csv(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)
    return rows


def gpu_of(i):
    return i % max(1, len(gpus))


sys.path.insert(0, f"{REPO}/baseline_performance")
from compare_per_state import compare, format_report


def episodes(rows, task=TASK_ID):
    return {(task, int(r["seed"]), int(r["init_state"])): int(r["success"]) for r in rows if int(r["task_id"]) == task}


def in_ranges(i, spec):
    return any(int(a) <= i <= int(b) for a, b in (p.split("-") if "-" in p else (p, p) for p in spec.split(",")))


os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
''')

md("## 7. Resume from an earlier version's output")
code(src(14))

md("## 8. Stage `verify`: smoke run and harness check")
code('''
if STAGE == "verify":
    # 1. Smoke run: 1 short episode, 3 training steps at the real batch size, 1 short paired evaluation.
    smoke = f"{OUT_ROOT}/_smoke"
    sh(f"cd {REPO} && {PY} -u {MAIN} --stage collect {BASE_ARGS} --data_dir {smoke}/data --init_states 10 "
       f"--collect_seeds 0 --max_steps 60", env=RUN_ENV)
    r = sh(f"cd {REPO} && {PY} -u {MAIN} --stage train {BASE_ARGS} --data_dir {smoke}/data --selection all "
           f"--train_steps 3 --batch_size {BATCH_SIZE} --log_every 1 --output_dir {smoke}/run {TRAIN_EXTRA}", env=RUN_ENV)
    print("\\n".join(l for l in r.stdout.splitlines() if "sec_per_step" in l or "peak GPU" in l))
    for _ in range(4):
        r = sh(f"cd {REPO} && {PY} -u {MAIN} --stage eval {BASE_ARGS} --output_dir {smoke}/run --eval_seeds 0 "
               f"--eval_init_states 0 --max_steps 60 --max_episodes_per_process 1", check=False, env=RUN_ENV)
        if r.returncode != 3:
            break
    assert r.returncode == 0, "smoke evaluation failed"

    # 2. Harness check: base policy through this evaluation path on init states 0-9, seed 0, against the same
    #    10 episodes of pilot 1's base evaluation (identical snapshot, settings and episode seeds).
    hc = f"{OUT_ROOT}/harness_check"
    jobs = {f"harness_{k}": (f"cd {REPO} && {PY} -u {MAIN} --stage eval {BASE_ARGS} --eval_policies base "
                            f"--eval_seeds 0 --eval_init_states {part} --eval_output_dir {hc}/part{k} "
                            f"--max_episodes_per_process 3", gpu_of(k))
            for k, part in enumerate(["0-4", "5-9"])}
    codes = run_jobs(jobs, max_parallel=2)
    assert all(c == 0 for c in codes.values()), codes
    mine = write_csv(read_csv(f"{hc}/part0/per_episode_base.csv") + read_csv(f"{hc}/part1/per_episode_base.csv"),
                     f"{hc}/per_episode_base.csv")
    ref = [r for r in read_csv(BASE_CSV) if r["seed"] == "0"]
    report = format_report(compare(episodes(ref), episodes(mine)), "pilot-1 base (SAC harness)", "self_filtered_bc eval")
    open(f"{hc}/harness_check.txt", "w").write(report)
    print(report)
    same = sum(episodes(ref)[k] == v for k, v in episodes(mine).items())
    print(f"{same}/10 episodes with the same outcome. PASS if the success counts are close (same seeds and GPU "
          "type, but GPU arithmetic is not bit-reproducible, and task-6 outcomes flip under tiny perturbations).")
else:
    print("stage is", STAGE)
''')

md("## 9. Stage `collect`: base-policy rollouts on training init states")
code('''
if STAGE == "collect":
    import psutil
    parallel = max(1, min(len(INIT_STATE_PARTS), int((psutil.virtual_memory().available / 1e9 - 4) // GB_PER_JOB)))
    # One data folder per process (separate collection.csv files); episodes are moved into DATA/episodes after.
    jobs = {f"collect_{k}": (f"cd {REPO} && {PY} -u {MAIN} --stage collect {BASE_ARGS} --data_dir {DATA}/part{k} "
                             f"--init_states {part} --collect_seeds {COLLECT_SEEDS} --max_episodes_per_process 40",
                             gpu_of(k))
            for k, part in enumerate(INIT_STATE_PARTS)}
    print(f"{len(jobs)} collection processes, {parallel} at a time")
    codes = run_jobs(jobs, max_parallel=parallel)
    print(codes)
    os.makedirs(f"{DATA}/episodes", exist_ok=True)
    for f in glob.glob(f"{DATA}/part*/episodes/*.npz"):
        shutil.move(f, f"{DATA}/episodes/{os.path.basename(f)}")
    rows = write_csv([r for k in range(len(INIT_STATE_PARTS)) for r in read_csv(f"{DATA}/part{k}/collection.csv")],
                     f"{DATA}/collection.csv")
    n_files = len(glob.glob(f"{DATA}/episodes/*.npz"))
    succ = [r for r in rows if r["success"] == "1"]
    print(f"{len(rows)} episodes collected ({n_files} files), {len(succ)} successful")
    by_state = {}
    for r in rows:
        by_state.setdefault(int(r["init_state"]), []).append(int(r["success"]))
    print("successes per init state:", {i: sum(v) for i, v in sorted(by_state.items())})
    train_succ = [r for r in succ if not in_ranges(int(r["init_state"]), VAL_INIT_STATES)]
    print(f"successful episodes for training (excluding {VAL_INIT_STATES}): {len(train_succ)}; "
          f"mean steps {sum(int(r['steps']) for r in train_succ) / max(1, len(train_succ)):.0f}")
    sh(f"du -sh {DATA}/episodes")
    assert all(c == 0 for c in codes.values()), "some collection processes failed; run this stage again to resume"
else:
    print("stage is", STAGE)
''')

md("## 10. Stage `train`: one LoRA adapter per run")
code('''
if STAGE == "train":
    assert os.path.exists(f"{DATA}/collection.csv"), "run the collect stage first (and add its output as input)"
    jobs = {f"train_{sel}_seed{s}": (
        f"cd {REPO} && {PY} -u {MAIN} --stage train {BASE_ARGS} --data_dir {DATA} --selection {sel} --seed {s} "
        f"--val_init_states {VAL_INIT_STATES} --train_steps {TRAIN_STEPS} --batch_size {BATCH_SIZE} "
        f"--output_dir {OUT_ROOT}/runs/{sel}_seed{s} {TRAIN_EXTRA}", gpu_of(i))
        for i, (sel, s) in enumerate(TRAIN_RUNS)
        if not os.path.exists(f"{OUT_ROOT}/runs/{sel}_seed{s}/adapter/run_config.json")}
    print("to train:", list(jobs) or "nothing (all adapters exist)")
    codes = run_jobs(jobs, max_parallel=max(1, len(gpus)))
    print(codes)
    for name in jobs:
        lines = open(f"{OUT_ROOT}/logs/{name}.log").read().splitlines()
        print(name, [l for l in lines if "training episodes" in l or "frames" in l][:2])
        print("   ", [l for l in lines if "val_loss" in l][-3:], [l for l in lines if "peak GPU" in l])
    assert all(c == 0 for c in codes.values()), codes
else:
    print("stage is", STAGE)
''')

md("## 11. Stage `eval`: LoRA vs base on the same episodes")
code('''
if STAGE == "eval":
    import psutil
    base_rows = read_csv(BASE_CSV)
    assert len(base_rows) == 30, BASE_CSV
    jobs = {}
    for k, (sel, s) in enumerate(TRAIN_RUNS):
        run = f"{OUT_ROOT}/runs/{sel}_seed{s}"
        assert os.path.exists(f"{run}/adapter/run_config.json"), f"no adapter for {sel} seed {s}; run the train stage"
        for j, es in enumerate(EVAL_SEEDS):  # reuse the base episodes, so only the LoRA policy runs
            d = f"{run}/eval/seed{es}"
            if not os.path.exists(f"{d}/per_episode_base.csv"):
                write_csv([r for r in base_rows if int(r["seed"]) == es], f"{d}/per_episode_base.csv")
            jobs[f"eval_{sel}_seed{s}_es{es}"] = (
                f"cd {REPO} && {PY} -u {MAIN} --stage eval {BASE_ARGS} --output_dir {run} --eval_policies base,lora "
                f"--eval_seeds {es} --eval_init_states {EVAL_INIT_STATES} --eval_output_dir {d} "
                f"--max_episodes_per_process 3", gpu_of(k * len(EVAL_SEEDS) + j))
    parallel = max(1, min(4, int((psutil.virtual_memory().available / 1e9 - 4) // GB_PER_JOB)))
    codes = run_jobs(jobs, max_parallel=parallel)
    assert all(c == 0 for c in codes.values()), codes
    pooled_b, pooled_l = {}, {}
    for sel, s in TRAIN_RUNS:
        run = f"{OUT_ROOT}/runs/{sel}_seed{s}"
        lora = write_csv([r for es in EVAL_SEEDS for r in read_csv(f"{run}/eval/seed{es}/per_episode_lora.csv")],
                         f"{run}/eval/per_episode_lora.csv")
        b, l = episodes(base_rows), episodes(lora)
        report = format_report(compare(b, l), "base", f"LoRA ({sel}, training seed {s})")
        open(f"{run}/eval/comparison_base_vs_lora.txt", "w").write(report)
        print(report)
        if sel == "success":
            pooled_b.update({(t, 100 * s + es, i): v for (t, es, i), v in b.items()})
            pooled_l.update({(t, 100 * s + es, i): v for (t, es, i), v in l.items()})
    if pooled_l:
        report = format_report(compare(pooled_b, pooled_l), "base", "LoRA (success), all training seeds")
        open(f"{OUT_ROOT}/runs/comparison_pooled_success.txt", "w").write(report)
        print(report)
else:
    print("stage is", STAGE)
''')

md("## 12. Files for the repo")
code('''
# Small files for the repo (no episodes or adapter weights); the full folder stays in the Output tab.
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/{OUT_NAME}_results.zip . -x '*.npz' '*.safetensors' '*.pt' 'wandb/*'")
sh(f"du -h /kaggle/working/{OUT_NAME}_results.zip; du -sh {OUT_ROOT}")
''')

nb = {"cells": cells, "metadata": sac["metadata"], "nbformat": 4, "nbformat_minor": sac.get("nbformat_minor", 5)}
for c in nb["cells"]:  # store sources as lists of lines, as Jupyter does
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
