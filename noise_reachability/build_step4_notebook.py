"""Builds noise_reachability/kaggle_step4_weight_update.ipynb (run from the repo root).

The hardware, code and environment cells are taken verbatim from baseline_performance/kaggle_baseline_lerobot_eval.ipynb
(as the step-3b notebook), so every notebook installs the same stack. Edit this builder, not the notebook.
"""
import json
from pathlib import Path

REPO = Path.cwd()
base = json.load(open(REPO / "baseline_performance/kaggle_baseline_lerobot_eval.ipynb"))
src = lambda i: "".join(base["cells"][i]["source"])  # noqa: E731
HARDWARE, CODE, ENVIRONMENT = src(4), src(6), src(8)
assert "JOBS = max(1" in HARDWARE and "git clone" in CODE and "uv venv" in ENVIRONMENT

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 4: weight updates on task 0's systematic init states (Kaggle, 2 x T4)

Step 4 of `noise_reachability/README.md` (design fixed 2026-10-29). Base `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, against three weight updates, on task 0, init states 0-9 x evaluation seeds 16-31:

- **A**: LoRA on the base's own successful rollouts (init states 10-49 x seeds 0-2), training seeds 0-2;
- **C**: the same LoRA training on the 33 task-0 demonstrations of `lerobot/libero`, training seeds 0-2;
- **B**: the continued-training checkpoint (`continued_training/`), as it is.

**Before the first version:** add the continued-training checkpoint (`smolvla_libero10_continued_final/` from
Google Drive, uploaded as a private Kaggle dataset; the continued-training evaluation used the same dataset) as
input (**Add Input > Your Datasets**). Step 7 finds it by the job name in its `train_config.json`, whatever the
folder is called, and refuses to run without it.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. Each version does
whatever is not done yet, in this order, and stops before the session limit:

1. checks (once): demonstration conversion and its checks, arm B's checkpoint, a fresh LoRA reproducing the
   base, and the **harness check** (the base on task 0, init states 0-9 x seeds 0-2, against step 0's 7/30;
   the run stops if the gap is 7 or more episodes);
2. collection: 120 base episodes on init states 10-49;
3. training: 6 LoRA adapters (A and C x seeds 0-2);
4. evaluation: 8 policies x 160 episodes;
5. the analysis (`analyze_step4.py`), once every episode is done.

**More than one session is needed** (estimate: 10-14 h in all). When step 11 says work remains, add this
version's output as input (**Add Input > Your Work > this notebook's output**, keep the checkpoint dataset)
and Save & Run All again: step 6 copies the earlier output in and everything resumes. Every version must clone
the same commit (step 6 checks).

Downloads (Output tab): `step4_results.zip` (CSVs, logs, reports, provenance; small). The episodes and adapters
stay in the version's output folder.
""")

md("## 1. Settings")
code('''
import csv, glob, hashlib, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

T_SESSION = time.time()
REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "9318dd507a3cc3413577e64b514a694aa47c0ffd"  # the reviewed step-4 commit
REPO = "/tmp/efficient_vla"        # code and venv live outside /kaggle/working, so they are not saved as output
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"

# The protocol of README.md step 4; nothing here is meant to be edited.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
B_JOB_NAME = "smolvla_libero10_continued"     # arm B: the dataset-input folder whose train_config.json names this job
TASK = 0
N_ACTION_STEPS = 10
COLLECT_PARTS = ["10-19", "20-29", "30-39", "40-49"]   # init states, one collection process each
COLLECT_SEEDS = "0-2"
VAL_INIT_STATES = "45-49"          # arm A's validation episodes
VAL_DEMO_EVERY = 5                 # arm C's validation demonstrations (demo_index 4, 9, ...)
TRAIN_SEEDS = [0, 1, 2]
TRAIN_STEPS, BATCH_SIZE = 2000, 16 # the task-6 settings, unchanged
EVAL_SEED_PARTS = ["16-19", "20-23", "24-27", "28-31"]  # 4 x 4 seeds = 16 per init state
EVAL_INIT_STATES = "0-9"
HARNESS_SEEDS = "0-2"
OUT_NAME = "step4"
OUT_ROOT = f"/kaggle/working/{OUT_NAME}"
STOP_AT_H = 10.5                   # no new work after this many hours of the session; running jobs are stopped
TRAIN_NEEDS_H = 1.5                # before the timing check: no training run starts with less time left (it saves
                                   # only at the end); step 9 replaces it with 1.25 x the measured time plus 10 min
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
POLICIES = ["base"] + [f"A{s}" for s in TRAIN_SEEDS] + [f"C{s}" for s in TRAIN_SEEDS] + ["B"]
print(f"step 4: task {TASK}, n_action_steps {N_ACTION_STEPS}, policies {POLICIES}, eval seeds 16-31 x init "
      f"states {EVAL_INIT_STATES}, output {OUT_ROOT}")


def sh(cmd, check=True, env=None):
    # Capture and print: notebooks do not show output that a subprocess writes directly.
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {r.returncode}: {cmd}")
    return r


def hours_left():
    return STOP_AT_H - (time.time() - T_SESSION) / 3600
''')

md("## 2. Hardware")
code(HARDWARE + '''
assert len(gpus) == 2, "Set Accelerator to GPU T4 x2"
assert JOBS >= 4, f"RAM for {JOBS} SmolVLA processes; the notebook runs 4 at once"
''')

md("## 3. Code")
code(CODE)

md("""
## 4. Environment

The package versions of the local `lerobot_env` (as the step-0/1/3b Kaggle notebooks). Takes several minutes.
""")
code(ENVIRONMENT)

md("""
## 5. Sanity checks, tests and the base checkpoint
""")
code('''
RUN_ENV = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MPLBACKEND="Agg", WANDB_MODE="disabled", OMP_NUM_THREADS="1", PYTHONPATH=REPO,
               PYTHONFAULTHANDLER="1")
sh(f"""{PY} -c "
import torch, av
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], '| av', av.__version__)
" """, env=RUN_ENV)
egl = sh(f"{PY} -c \\"import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('EGL context ok')\\"", check=False, env=RUN_ENV)
assert "EGL context ok" in egl.stdout, "EGL rendering failed"
sh(f"cd {REPO} && {PY} -m pytest tests/ -q -p no:cacheprovider", env=RUN_ENV)
r = sh(f"""{PY} -c "from huggingface_hub import snapshot_download; print(snapshot_download('{CHECKPOINT_REPO}', revision='{CHECKPOINT_REVISION}'))" """, env=RUN_ENV)
POLICY_PATH = r.stdout.strip().splitlines()[-1]
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"
BASE_ARGS = f"--task_id {TASK} --n_action_steps {N_ACTION_STEPS}"
print("base checkpoint", POLICY_PATH)
''')

md("""
## 6. Resume from an earlier version's output

Copies an earlier version's output in, and refuses to resume on code other than the commit the first version
cloned. Nothing happens on a first run.
""")
code('''
os.makedirs(OUT_ROOT, exist_ok=True)
for prev in sorted(glob.glob(f"/kaggle/input/**/{OUT_NAME}", recursive=True)):
    if os.path.isdir(prev) and os.path.exists(f"{prev}/repo_commit.txt"):
        print("copying", prev)
        shutil.copytree(prev, OUT_ROOT, dirs_exist_ok=True)
head = sh(f"git -C {REPO} log -1 --format='%H'").stdout.strip()
first = Path(OUT_ROOT, "repo_commit.txt")
started_with = first.read_text().split()[0] if first.exists() and first.read_text().strip() else None
if started_with and started_with != head:
    raise RuntimeError(f"the run started at commit {started_with}, this version cloned {head}: set REPO_REF in "
                       f"step 1 to {started_with}")
with open(first, "a") as f:
    f.write(head + "\\n")
for d in ("logs", "demos", "data", "runs", "eval", "harness_check"):
    os.makedirs(f"{OUT_ROOT}/{d}", exist_ok=True)
''')

md("""
## 7. Arm B's checkpoint (Kaggle dataset input)
""")
code('''
# As baseline_performance/kaggle_baseline_lerobot_eval.ipynb: found by its training job, whatever the folder is called.
found = [str(Path(f).parent) for f in glob.glob("/kaggle/input/**/train_config.json", recursive=True)
         if json.load(open(f)).get("job_name") == B_JOB_NAME and Path(f).with_name("model.safetensors").exists()]
assert len(found) == 1, (f"need exactly one input folder whose train_config.json has job_name {B_JOB_NAME} and a "
                         f"model.safetensors, found {found}: add the continued-training checkpoint as a dataset input")
B_PATH = found[0]
h = hashlib.sha256()
with open(f"{B_PATH}/model.safetensors", "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
B_INFO = {"path": B_PATH, "model_sha256": h.hexdigest(), "files": sorted(os.listdir(B_PATH))}
prev = Path(OUT_ROOT, "arm_b_checkpoint.json")
if prev.exists():
    assert json.loads(prev.read_text())["model_sha256"] == B_INFO["model_sha256"], "arm B's checkpoint changed between versions"
prev.write_text(json.dumps(B_INFO, indent=1) + "\\n")
print(json.dumps(B_INFO, indent=1))
''')

md("""
## 8. Helpers: parallel jobs with resume and a deadline
""")
code('''
def run_jobs(jobs, max_parallel, poll_s=30, log_every_s=600):
    """Run {name: (command, gpu)} with at most max_parallel at once; exit code 3 means 'call again' (more
    episodes in a fresh process). No job starts after the deadline and running ones are stopped at it.
    Returns {name: exit code}; a job stopped at the deadline gets None."""
    pending, running, codes, last_log = list(jobs), {}, {}, time.time()
    while pending or running:
        while pending and len(running) < max_parallel and hours_left() > 0:
            name = pending.pop(0)
            cmd, gpu = jobs[name]
            log = open(f"{OUT_ROOT}/logs/{name}.log", "a")
            env = dict(RUN_ENV, CUDA_VISIBLE_DEVICES=str(gpu))
            running[name] = (subprocess.Popen(cmd, shell=True, executable="/bin/bash", env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True), log)
            time.sleep(5)  # staggered EGL start-up (FAIL-006)
        if hours_left() <= 0 and running:
            print(time.strftime("%H:%M"), "deadline: stopping", list(running), flush=True)
            for name, (proc, log) in running.items():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(20)
            for name, (proc, log) in running.items():
                log.close()
                codes[name] = None
            for name in pending:
                codes[name] = None
            return codes
        if hours_left() <= 0 and not running:
            for name in pending:
                codes[name] = None
            return codes
        for name, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            if proc.returncode == 3:
                pending.insert(0, name)
                continue
            codes[name] = proc.returncode
            print(time.strftime("%H:%M"), name, "exit", proc.returncode, flush=True)
        if time.time() - last_log > log_every_s:
            last_log = time.time()
            print(time.strftime("%H:%M"), f"{hours_left():.1f} h left; running {list(running)}; "
                  f"{len(pending)} waiting", flush=True)
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


def n_range(spec):
    a, b = (spec.split("-") + [spec])[:2]
    return int(b) - int(a) + 1


def failed(codes):
    return {k: c for k, c in codes.items() if c not in (0, None)}
''')

md("""
## 9. Checks (once): demonstrations, arm B, fresh LoRA, harness

All 33 task-0 demonstrations are converted (`demo_episodes.py`, pyav decoding), then `check_step4.py` checks
the base's loss on them (below 0.3; rotated frames at least 2x), arm B's checkpoint, and that a fresh LoRA
reproduces the base. Then the base on task 0, init states 0-9 x seeds 0-2 through this evaluation path, against
step 0's 7/30 at `n_action_steps=10`: a gap of 7 or more episodes stops the run (README, step 4).
""")
code('''
CHECKS_OK = Path(OUT_ROOT, "checks_passed.json")
if not CHECKS_OK.exists():
    demo_dir = f"{OUT_ROOT}/demos"
    if not Path(demo_dir, "demo_manifest.json").exists():
        sh(f"cd {REPO} && {PY} -u noise_reachability/demo_episodes.py --task_id {TASK} --root /tmp/lerobot_libero "
           f"--out_dir {demo_dir}", env=RUN_ENV)
    manifest = json.loads(Path(demo_dir, "demo_manifest.json").read_text())
    assert len(manifest["episodes"]) == 33 and manifest["task_index"] == 5, manifest
    assert not manifest["provenance"]["git_dirty_step4"], "conversion ran on a dirty tree"
    print(f"{len(manifest['episodes'])} demonstrations, {manifest['total_frames']} frames")
    r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_step4.py --demo_dir {demo_dir} --task_id {TASK} "
           f"--policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS} --b_policy_path {B_PATH}",
           check=False, env=dict(RUN_ENV, CUDA_VISIBLE_DEVICES="0"))
    Path(OUT_ROOT, "check_step4.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0 and "[check] all passed" in r.stdout, "a step-4 check failed; see the output above"

    hc = f"{OUT_ROOT}/harness_check"
    jobs = {f"harness_{k}": (f"cd {REPO} && {PY} -u {MAIN} --stage eval {BASE_ARGS} --policy_path {POLICY_PATH} "
                            f"--eval_policies base --eval_seeds {HARNESS_SEEDS} --eval_init_states {part} "
                            f"--eval_output_dir {hc}/part{k} --max_episodes_per_process 8", k % 2)
            for k, part in enumerate(["0-2", "3-5", "6-7", "8-9"])}
    codes = run_jobs(jobs, max_parallel=4)
    assert not failed(codes) and all(c == 0 for c in codes.values()), codes
    rows = write_csv([r for k in range(4) for r in read_csv(f"{hc}/part{k}/per_episode_base.csv")],
                     f"{hc}/per_episode_base.csv")
    n = sum(int(r["success"]) for r in rows)
    assert len(rows) == 30, len(rows)
    print(f"harness check: base {n}/30 against step 0's 7/30 (gap {n - 7:+d}, allowed 6)")
    assert abs(n - 7) <= 6, "harness check FAILED: investigate the evaluation path before anything else runs"

    # Training on a T4 (bf16 autocast, forward and backward): 20 steps at the real batch size, not kept.
    r = sh(f"cd {REPO} && {PY} -u {MAIN} --stage train {BASE_ARGS} --policy_path {POLICY_PATH} --seed 0 "
           f"--selection success --data_dir {demo_dir} --val_demo_every {VAL_DEMO_EVERY} --train_steps 20 "
           f"--batch_size {BATCH_SIZE} --log_every 5 --val_every 1000 --output_dir /tmp/step4_train_timing",
           env=dict(RUN_ENV, CUDA_VISIBLE_DEVICES="0"))
    sps = float(re.findall(r"sec_per_step=([0-9.]+)", r.stdout)[-1])
    print(f"training: {sps:.2f} s per step on a T4 -> {TRAIN_STEPS * sps / 3600:.2f} h per adapter")
    CHECKS_OK.write_text(json.dumps({"harness_base": n, "demos": len(manifest["episodes"]), "train_sec_per_step": sps,
                                     "commit": head, "date": time.strftime("%Y-%m-%d %H:%M")}) + "\\n")
else:
    print("checks passed in an earlier version:", CHECKS_OK.read_text())
''')

md("""
## 10. Collection (arm A's data): the base on init states 10-49 x seeds 0-2
""")
code('''
DATA = f"{OUT_ROOT}/data"
need = 40 * n_range(COLLECT_SEEDS)
merged = read_csv(f"{DATA}/collection.csv")
if len(merged) < need and hours_left() > 0:
    jobs = {f"collect_{k}": (f"cd {REPO} && {PY} -u {MAIN} --stage collect {BASE_ARGS} --policy_path {POLICY_PATH} "
                             f"--data_dir {DATA}/part{k} --init_states {part} --collect_seeds {COLLECT_SEEDS} "
                             f"--max_episodes_per_process 15", k % 2)
            for k, part in enumerate(COLLECT_PARTS)}
    codes = run_jobs(jobs, max_parallel=4)
    assert not failed(codes), f"collection failed: {codes}"
    rows = [r for k in range(len(COLLECT_PARTS)) for r in read_csv(f"{DATA}/part{k}/collection.csv")]
    if len(rows) == need:
        os.makedirs(f"{DATA}/episodes", exist_ok=True)
        for f in glob.glob(f"{DATA}/part*/episodes/*.npz"):
            shutil.move(f, f"{DATA}/episodes/{os.path.basename(f)}")
        merged = write_csv(rows, f"{DATA}/collection.csv")
if len(merged) == need:
    succ = [r for r in merged if r["success"] == "1"]
    train = [r for r in succ if not 45 <= int(r["init_state"]) <= 49]
    print(f"collection complete: {len(succ)}/{need} successful; {len(train)} for training, "
          f"{len(succ) - len(train)} for validation; files {len(glob.glob(f'{DATA}/episodes/*.npz'))}")
    assert len(glob.glob(f"{DATA}/episodes/*.npz")) == need
    assert len(train) > 0, "no successful training episode for arm A"
COLLECTED = len(merged) == need
''')

md("""
## 11. Training: 6 LoRA adapters (arms A and C, training seeds 0-2)
""")
code('''
def adapter_done(name):
    return Path(OUT_ROOT, "runs", name, "adapter", "run_config.json").exists()


TRAIN_NEEDS_H = 1.25 * TRAIN_STEPS * json.loads(CHECKS_OK.read_text())["train_sec_per_step"] / 3600 + 10 / 60
print(f"a training run starts only with more than {TRAIN_NEEDS_H:.2f} h left")
TRAIN = {}
for s in TRAIN_SEEDS:
    common = (f"cd {REPO} && {PY} -u {MAIN} --stage train {BASE_ARGS} --policy_path {POLICY_PATH} --seed {s} "
              f"--selection success --train_steps {TRAIN_STEPS} --batch_size {BATCH_SIZE}")
    if COLLECTED:
        TRAIN[f"A{s}"] = f"{common} --data_dir {OUT_ROOT}/data --val_init_states {VAL_INIT_STATES} --output_dir {OUT_ROOT}/runs/A{s}"
    TRAIN[f"C{s}"] = f"{common} --data_dir {OUT_ROOT}/demos --val_demo_every {VAL_DEMO_EVERY} --output_dir {OUT_ROOT}/runs/C{s}"
todo = [n for n in TRAIN if not adapter_done(n)]
print("adapters to train:", todo or "none")
while todo and hours_left() > TRAIN_NEEDS_H:
    batch, todo = todo[:2], todo[2:]
    for n in batch:
        shutil.rmtree(f"{OUT_ROOT}/runs/{n}", ignore_errors=True)   # an unfinished run restarts from scratch
    codes = run_jobs({f"train_{n}": (TRAIN[n], k) for k, n in enumerate(batch)}, max_parallel=2)
    assert not failed(codes), f"training failed: {codes}"
    for n in batch:
        lines = open(f"{OUT_ROOT}/logs/train_{n}.log").read().splitlines()
        print(n, [l for l in lines if "training episodes" in l or "training frames" in l or "peak GPU" in l],
              [l.split("val_loss=")[-1] for l in lines if "val_loss" in l][-3:])
DONE_ADAPTERS = [n for n in [f"A{s}" for s in TRAIN_SEEDS] + [f"C{s}" for s in TRAIN_SEEDS] if adapter_done(n)]
print("adapters done:", DONE_ADAPTERS)
''')

md("""
## 12. Evaluation: 8 policies x 160 episodes (init states 0-9 x seeds 16-31)
""")
code('''
def eval_cmd(policy, part, out):
    common = (f"cd {REPO} && {PY} -u {MAIN} --stage eval {BASE_ARGS} --eval_seeds {part} "
              f"--eval_init_states {EVAL_INIT_STATES} --eval_output_dir {out} --max_episodes_per_process 10")
    if policy == "base":
        return f"{common} --policy_path {POLICY_PATH} --eval_policies base"
    if policy == "B":
        return f"{common} --policy_path {B_PATH} --eval_policies base"
    return f"{common} --policy_path {POLICY_PATH} --eval_policies lora --output_dir {OUT_ROOT}/runs/{policy}"


def eval_rows(policy):
    name = "lora" if policy[0] in "AC" else "base"
    return [r for k in range(len(EVAL_SEED_PARTS))
            for r in read_csv(f"{OUT_ROOT}/eval/{policy}/part{k}/per_episode_{name}.csv")]


ready = ["base", "B"] + DONE_ADAPTERS
jobs, k = {}, 0
for policy in ready:
    for j, part in enumerate(EVAL_SEED_PARTS):
        out = f"{OUT_ROOT}/eval/{policy}/part{j}"
        name = "lora" if policy[0] in "AC" else "base"
        if len(read_csv(f"{out}/per_episode_{name}.csv")) < 10 * n_range(part):
            jobs[f"eval_{policy}_part{j}"] = (eval_cmd(policy, part, out), k % 2)
            k += 1
print(f"{len(jobs)} evaluation jobs to run; {hours_left():.1f} h left")
if jobs and hours_left() > 0:
    codes = run_jobs(jobs, max_parallel=4)
    assert not failed(codes), f"evaluation failed: {codes}"
for policy in POLICIES:
    rows = eval_rows(policy)
    print(f"{policy}: {len(rows)}/160 episodes")
ALL_DONE = all(len(eval_rows(p)) == 160 for p in POLICIES)
print("all episodes done" if ALL_DONE else "WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("""
## 13. Analysis (when every episode is done)

`analyze_step4.py`: every arm against the base on the systematic and the reachable episodes, A against C,
and the reading fixed in the README, with the harness check.
""")
code('''
if ALL_DONE:
    for p in POLICIES:
        write_csv(eval_rows(p), f"{OUT_ROOT}/eval/per_episode_{p}.csv")
    arms = " ".join(f"{p}={OUT_ROOT}/eval/per_episode_{p}.csv" for p in POLICIES if p != "base")
    sh(f"cd {REPO} && {PY} -S noise_reachability/analyze_step4.py --base {OUT_ROOT}/eval/per_episode_base.csv "
       f"--arm {arms} --harness {OUT_ROOT}/harness_check/per_episode_base.csv --out {OUT_ROOT}/step4_report.txt",
       env=RUN_ENV)
else:
    print("evaluation not finished; analysis skipped")
''')

md("## 14. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step4_results.zip . -x '*.npz' '*.safetensors' '*.pt' 'data/part*/episodes/*'")
sh(f"du -h /kaggle/working/step4_results.zip; du -sh {OUT_ROOT}")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
''')

nb = {"cells": cells, "metadata": base["metadata"], "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
out = REPO / "noise_reachability" / "kaggle_step4_weight_update.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out, len(cells), "cells")
