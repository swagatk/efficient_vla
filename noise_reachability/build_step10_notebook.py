"""Builds noise_reachability/kaggle_step10_learn_from_noise.ipynb (run from the repo root).

    python noise_reachability/build_step10_notebook.py <commit> [output path]

The settings, hardware, code, environment, sanity, resume, job, wrapper-check and harness cells come from step 8's
notebook, so step 10 installs and runs as step 8 did; collection, training and evaluation follow step 4's notebook.
The replays come from analyze_step10.py --replay_jobs, so their list is written in one place. Edit this builder, not
the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step10_learn_from_noise.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + '''# The protocol of README.md step 10; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = [0, 6, 7]
N_ACTION_STEPS = 10
COLLECT_PARTS = ["10-19", "20-29", "30-39", "40-49"]   # arm A's data, as step 4: task 0, one process each
COLLECT_SEEDS = "0-2"
VAL_INIT_STATES = "45-49"          # arm A's validation episodes, as step 4
TRAIN_SEEDS = [0, 1, 2]
TRAIN_STEPS, BATCH_SIZE = 2000, 16 # step 4's settings, unchanged
EVAL_STATES = {0: "0-9,11,14,27", 6: "0,8,12,14,20", 7: "4,33,38,47"}
EVAL_SEED_PARTS = {0: ["16-19", "20-23", "24-27", "28-31"],
                   6: ["16-19", "20-23", "24-27", "28-31", "32-35", "36-39", "40-43", "44-47"],
                   7: ["16-19", "20-23", "24-27", "28-31", "32-35", "36-39", "40-43", "44-47"]}
OUT_NAME = "step10"
OUT_ROOT = f"/kaggle/working/{OUT_NAME}"
STOP_AT_H = 9.0                    # no new work after this many hours of the session; running jobs are stopped
TRAIN_NEEDS_H = 3.5                # before the timing check: no training run starts with less time left
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + '''# Adapters in training order: arm A first (its data is cheap), then arm N one task after another per seed.
ADAPTERS = [f"A{s}" for s in TRAIN_SEEDS] + [f"N{t}_{s}" for s in TRAIN_SEEDS for t in TASKS]
# Policies per task under analyze_step10.py's names (N<s> is adapter N<t>_<s> on task t).
POLICIES = {t: ["base"] + [f"N{s}" for s in TRAIN_SEEDS] + ([f"A{s}" for s in TRAIN_SEEDS] if t == 0 else [])
            for t in TASKS}


def adapter_of(t, policy):
    return policy if policy.startswith("A") else f"N{t}_{policy[1:]}"
print(f"step 10 at {REPO_REF}: tasks {TASKS}, n_action_steps {N_ACTION_STEPS}, adapters {ADAPTERS}, "
      f"at most {90 + 122 + 120 + 2608} episodes (90 harness, 122 replays, 120 collection, 2608 evaluation), "
      f"stop after {STOP_AT_H} h, output {OUT_ROOT}")


''' + SETTINGS[end:]
HARDWARE = step8_cell("JOBS = max(1")
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
RESUME = step8_cell("repo_commit.txt")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
JOBS = swap(JOBS, "to the zip in step 15", "to the zip in step 16")
CHECK = step8_cell("CHECK_DONE = True")
HARNESS = step8_cell("HARNESS_DONE = CHECK_DONE and")
HARNESS = swap(HARNESS, 'run_stage("harness", jobs, 12)', 'run_stage("harness", jobs, 3 * len(TASKS))')
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 10: can constant-noise successes be learned into the weights? (Kaggle, 2 x T4)

Step 10 of `noise_reachability/README.md` (design fixed 2026-11-03). The base `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, tasks 0, 6, 7: replays of earlier constant-noise successes are recorded, LoRA adapters are trained
on them (arm N) and on the base's own successes (arm A, task 0), and every policy is evaluated with sampled noise.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The wrapper check and the
harness run first; a failure of either stops the run. Each session stops starting work after 9 h and then writes the zip.

**This run needs about four sessions.** When step 15 says work remains, add this version's output as input (**Add Input
> Your Work**), and only that version, and Save & Run All again: finished episodes and adapters are kept. Step 6 must
then print `RESUMING`.

Download from the Output tab: `step10_results.zip` (results without recordings or adapter weights, which stay in the
version's output for the next session).
""")

md("## 1. Settings")
code(SETTINGS)

md("## 2. Hardware")
code(HARDWARE + '''
assert len(gpus) == 2, "Set Accelerator to GPU T4 x2"
''')

md("## 3. Code")
code(CODE)

md("""
## 4. Environment

The package versions of the local `lerobot_env` (as the earlier Kaggle notebooks). Takes several minutes.
""")
code(ENVIRONMENT)

md("## 5. Sanity checks, tests and the base checkpoint")
code(SANITY)

md("## 6. Resume from an earlier version's output")
code(RESUME)

md("## 7. Helpers: parallel jobs with resume and a deadline")
code(JOBS + '''
SEARCH = f"cd {REPO} && {PY} -u noise_reachability/deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step10.py --root {OUT_ROOT}"
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


def sfbc(task):
    return f"cd {REPO} && {PY} -u {MAIN} --task_id {task} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}"


def run_stage(name, jobs, total, max_parallel=4):
    """True when the stage has no work left: a job that failed stops the run, the deadline leaves work for a next version."""
    print(f"{name}: {len(jobs)} of {total} jobs to run; {hours_left():.1f} h left", flush=True)
    if not jobs:
        return True
    if hours_left() <= 0:
        return False
    codes = run_jobs(jobs, max_parallel=max_parallel)
    if failed(codes):
        fail(f"{name} failed: {failed(codes)}")
        return False
    return all(c == 0 for c in codes.values()) and len(codes) == len(jobs)


def run_gate(which):
    if sh(f"{ANALYZE} --gate {which}", check=False, env=RUN_ENV).returncode != 0:
        fail(f"the {which} gate failed: investigate before spending more")
        return False
    return True


def adapter_done(name):
    return Path(OUT_ROOT, "runs", name, "adapter", "run_config.json").exists()
''')

md("## 8. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code(CHECK)

md("## 9. Harness: the base on seeds 0-2, per task; stops the run if it fails")
code(HARNESS)

md("""
## 10. Arm N's data: replay and record every constant-noise success on a resistant state

The list comes from `analyze_step10.py --replay_jobs` (from the committed result files); recordings go to
`task_<t>/replay_data/episodes/`.
""")
code('''
REPLAYS_DONE = False
if HARNESS_DONE:
    r = sh(f"cd {REPO} && {PY} -S noise_reachability/analyze_step10.py --replay_jobs", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail("the replay list failed")
    else:
        plan = [line.split() for line in r.stdout.strip().splitlines()]
        jobs = {}
        for t, f, s, ids, js in plan:
            out = f"{task_dir(t)}/replay"
            n = len((ids if js == "-" else js).split(","))
            csv_name = f"per_episode_{f}_state{s}.csv" if js == "-" else f"confirm_{f}_n{ids}_state{s}.csv"
            if len(read_csv(f"{out}/{csv_name}")) < n:
                jitter = "" if js == "-" else f" --jitter_seeds {js}"
                jobs[f"replay_task{t}_{f}_state{s}_{ids if js != '-' else 'plain'}"] = (
                    f"{SEARCH} --task_id {t} --family {f} --init_state {s} --noise_ids {ids}{jitter} --output_dir {out} "
                    f"--record_dir {task_dir(t)}/replay_data", len(jobs) % 2)
        print(f"{len(plan)} replay jobs, {sum(len((i if j == '-' else j).split(',')) for _, _, _, i, j in plan)} episodes",
              flush=True)
        REPLAYS_DONE = run_stage("replays", jobs, len(plan))
        for t in TASKS:
            rows = [r for p in glob.glob(f"{task_dir(t)}/replay/*.csv") for r in read_csv(p)]
            print(f"task {t}: {sum(r['success'] == '1' for r in rows)} of {len(rows)} replays succeeded; "
                  f"{len([f for f in glob.glob(f'{task_dir(t)}/replay_data/episodes/*.npz') if not f.endswith('.tmp.npz')])} "
                  f"recordings", flush=True)
''')

md("## 11. Arm A's data: the base on task 0, init states 10-49 x seeds 0-2 (as step 4)")
code('''
DATA_A = f"{task_dir(0)}/collect_data"
need = 40 * n_range(COLLECT_SEEDS)
COLLECTED = len(read_csv(f"{DATA_A}/collection.csv")) == need
if HARNESS_DONE and not COLLECTED:
    jobs = {}
    for k, part in enumerate(COLLECT_PARTS):
        if len(read_csv(f"{task_dir(0)}/collect/part{k}/collection.csv")) < 10 * n_range(COLLECT_SEEDS):
            jobs[f"collect_{k}"] = (f"{sfbc(0)} --stage collect --data_dir {task_dir(0)}/collect/part{k} --init_states {part} "
                                    f"--collect_seeds {COLLECT_SEEDS} --max_episodes_per_process 15", k % 2)
    if run_stage("collection", jobs, len(COLLECT_PARTS)):
        rows = [r for k in range(len(COLLECT_PARTS)) for r in read_csv(f"{task_dir(0)}/collect/part{k}/collection.csv")]
        if len(rows) == need:
            os.makedirs(f"{DATA_A}/episodes", exist_ok=True)
            for f in glob.glob(f"{task_dir(0)}/collect/part*/episodes/*.npz"):
                if f.endswith(".tmp.npz"):     # a write cut off by a stopped process
                    continue
                shutil.move(f, f"{DATA_A}/episodes/{os.path.basename(f)}")
            write_csv(rows, f"{DATA_A}/collection.csv")
            COLLECTED = len([f for f in glob.glob(f"{DATA_A}/episodes/*.npz") if not f.endswith(".tmp.npz")]) == need
if COLLECTED:
    rows = read_csv(f"{DATA_A}/collection.csv")
    print(f"arm A's data: {sum(r['success'] == '1' for r in rows)} of {len(rows)} successful")
''')

md("""
## 12. Training: arm A (task 0) and arm N (tasks 0, 6, 7), training seeds 0-2

Two adapters at a time, one per T4. A run starts only if the session can finish it (1.25 x the measured time plus 10
minutes); an unfinished run restarts from scratch in the next session.
""")
code('''
TIMING = Path(OUT_ROOT, "train_timing.json")
if (REPLAYS_DONE or COLLECTED) and not TIMING.exists() and hours_left() > 0.5:
    data = DATA_A if COLLECTED else f"{task_dir(7)}/replay_data"
    task = 0 if COLLECTED else 7
    r = sh(f"{sfbc(task)} --stage train --seed 0 --selection success --data_dir {data} --train_steps 20 "
           f"--batch_size {BATCH_SIZE} --log_every 5 --val_every 1000 --output_dir /tmp/step10_train_timing",
           check=False, env=dict(RUN_ENV, CUDA_VISIBLE_DEVICES="0"))
    found = re.findall(r"sec_per_step=([0-9.]+)", r.stdout)
    if r.returncode != 0 or not found:
        fail("the training timing run failed")
    else:
        TIMING.write_text(json.dumps({"sec_per_step": float(found[-1]), "date": time.strftime("%Y-%m-%d %H:%M")}) + "\\n")
if TIMING.exists():
    TRAIN_NEEDS_H = 1.25 * TRAIN_STEPS * json.loads(TIMING.read_text())["sec_per_step"] / 3600 + 10 / 60
print(f"a training run starts only with more than {TRAIN_NEEDS_H:.2f} h left")


def train_cmd(name):
    common = f"--stage train --selection success --train_steps {TRAIN_STEPS} --batch_size {BATCH_SIZE}"
    if name.startswith("A"):
        return (f"{sfbc(0)} {common} --seed {name[1:]} --data_dir {DATA_A} --val_init_states {VAL_INIT_STATES} "
                f"--output_dir {OUT_ROOT}/runs/{name}")
    t, s = name[1:].split("_")
    return f"{sfbc(t)} {common} --seed {s} --data_dir {task_dir(t)}/replay_data --output_dir {OUT_ROOT}/runs/{name}"


ready = [n for n in ADAPTERS if not adapter_done(n) and (COLLECTED if n.startswith("A") else REPLAYS_DONE)]
print("adapters to train now:", ready or "none", flush=True)
while ready and hours_left() > TRAIN_NEEDS_H and not FAILURE:
    batch, ready = ready[:2], ready[2:]
    for n in batch:
        shutil.rmtree(f"{OUT_ROOT}/runs/{n}", ignore_errors=True)   # an unfinished run restarts from scratch
    codes = run_jobs({f"train_{n}": (train_cmd(n), k) for k, n in enumerate(batch)}, max_parallel=2)
    if failed(codes):
        fail(f"training failed: {failed(codes)}")
    for n in batch:
        lines = open(f"{OUT_ROOT}/logs/train_{n}.log").read().splitlines()
        print(n, [l for l in lines if "training episodes" in l or "training frames" in l or "peak GPU" in l])
DONE_ADAPTERS = [n for n in ADAPTERS if adapter_done(n)]
print("adapters done:", DONE_ADAPTERS)
''')

md("## 13. Evaluation with sampled noise: every policy whose adapter is ready")
code('''
def eval_out(t, policy, j):
    return f"{task_dir(t)}/eval/{policy}/part{j}"


def eval_cmd(t, policy, j):
    common = (f"{sfbc(t)} --stage eval --eval_seeds {EVAL_SEED_PARTS[t][j]} --eval_init_states {EVAL_STATES[t]} "
              f"--eval_output_dir {eval_out(t, policy, j)} --max_episodes_per_process 10")
    if policy == "base":
        return f"{common} --eval_policies base"
    return f"{common} --eval_policies lora --output_dir {OUT_ROOT}/runs/{adapter_of(t, policy)}"


def n_states(spec):
    return sum(n_range(p) for p in spec.split(","))


def part_size(t, j):
    return n_range(EVAL_SEED_PARTS[t][j]) * n_states(EVAL_STATES[t])


def eval_rows(t, policy):
    name = "base" if policy == "base" else "lora"
    return [r for j in range(len(EVAL_SEED_PARTS[t])) for r in read_csv(f"{eval_out(t, policy, j)}/per_episode_{name}.csv")]


jobs, k = {}, 0
if HARNESS_DONE and not FAILURE:
    for t in TASKS:
        for policy in POLICIES[t]:
            if policy != "base" and not adapter_done(adapter_of(t, policy)):
                continue
            name = "base" if policy == "base" else "lora"
            for j in range(len(EVAL_SEED_PARTS[t])):
                if len(read_csv(f"{eval_out(t, policy, j)}/per_episode_{name}.csv")) < part_size(t, j):
                    jobs[f"eval_task{t}_{policy}_part{j}"] = (eval_cmd(t, policy, j), k % 2)
                    k += 1
    run_stage("evaluation", jobs, len(jobs))
for t in TASKS:
    print(f"task {t}:", {p: len(eval_rows(t, p)) for p in POLICIES[t]},
          f"of {sum(part_size(t, j) for j in range(len(EVAL_SEED_PARTS[t])))} each")
ALL_DONE = (len(DONE_ADAPTERS) == len(ADAPTERS) and all(
    len(eval_rows(t, p)) == sum(part_size(t, j) for j in range(len(EVAL_SEED_PARTS[t]))) for t in TASKS for p in POLICIES[t]))
''')

md("## 14. Analysis (when every episode is done)")
code('''
if ALL_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step10_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
''')

md("## 15. Work left")
code('''
if not ALL_DONE and not FAILURE:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 16. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step10_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step10_results.zip; du -sh {OUT_ROOT}")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": step8["metadata"], "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, len(cells), "cells")
