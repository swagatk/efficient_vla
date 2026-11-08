"""Builds noise_reachability/kaggle_step12_expert_iteration.ipynb (run from the repo root).

    python noise_reachability/build_step12_notebook.py <commit> [output path]

The settings, hardware, code, environment, sanity, resume, job and wrapper-check cells come from step 8's notebook, as
step 10's builder takes them, so step 12 installs and runs as steps 8 and 10 did; training and evaluation follow step
10's cells. The states, noise ids and evaluation episodes are analyze_step12.py's constants, written into the settings
cell here, so they are fixed in one place. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step12 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step12_expert_iteration.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_HARNESS = len(a.TASKS) * len(a.HARNESS_SEEDS)
N_SEARCH = sum(len(v) for v in a.SEARCHED.values()) * len(a.SEARCH_IDS)
N_EVAL = len(a.POLICIES) * sum(len(a.EVAL_SEEDS[t]) * len(a.eval_states(t)) for t in a.TASKS)

SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 12; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {list(a.TASKS)}
N_ACTION_STEPS = 10
SEARCHED = {dict(a.SEARCHED)}       # states round 1 left weak: the round-1 explorer searches them
HELD_OUT = {dict(a.HELD_OUT)}       # selected like them, never searched: transfer
CHECK = {dict(a.CHECK)}             # round 1's best trained states: kept?
EXPLORER = {a.EXPLORER}                        # step 10's adapter N<t>_{a.EXPLORER} explores
SEARCH_IDS = "{a.SEARCH_IDS[0]}-{a.SEARCH_IDS[-1]}"         # fixed repeat noises, fresh ids
HARNESS_STATE = {dict(a.HARNESS_STATE)}   # step 10's N<t>_0 episodes re-run, seeds 16-19
HARNESS_SEEDS = "{a.HARNESS_SEEDS[0]}-{a.HARNESS_SEEDS[-1]}"
TRAIN_SEEDS = {list(a.TRAIN_SEEDS)}
TRAIN_STEPS, BATCH_SIZE = 2000, 16 # step 4's and step 10's settings, unchanged
EVAL_STATES = {dict(a.EVAL_STATE_SPECS)}
EVAL_SEED_PARTS = {dict(a.EVAL_SEED_PARTS)}
OUT_NAME = "step12"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
ROUND2_DATA = "/tmp/step12"        # round 1's recordings and round 2's, linked per session (not saved as output)
STOP_AT_H = 11.0                   # no new work after this many hours of the session; running jobs are stopped
TRAIN_NEEDS_H = 3.5                # before the timing check: no training run starts with less time left
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''# Round-2 adapters in training order: one task after another per seed. Policies under analyze_step12.py's names:
# N<s> is step 10's adapter N<t>_<s>, R<s> is round 2's R<t>_<s>.
ADAPTERS = [f"R{{t}}_{{s}}" for s in TRAIN_SEEDS for t in TASKS]
POLICIES = {a.POLICIES}


def adapter_of(t, policy):
    return f"{{policy[0]}}{{t}}_{{policy[1:]}}"
print(f"step 12 at {{REPO_REF}}: tasks {{TASKS}}, searched {{SEARCHED}}, held out {{HELD_OUT}}, check {{CHECK}}, "
      f"explorer N<t>_{{EXPLORER}}, repeat noise ids {{SEARCH_IDS}}, adapters {{ADAPTERS}}, at most "
      f"{N_HARNESS + N_SEARCH + N_EVAL} episodes ({N_HARNESS} harness, {N_SEARCH} search, {N_EVAL} evaluation), "
      f"stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


''' + SETTINGS[end:]
HARDWARE = step8_cell("JOBS = max(1")
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
RESUME = step8_cell("repo_commit.txt")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
JOBS = swap(JOBS, "to the zip in step 15", "to the zip in step 17")
CHECK = step8_cell("CHECK_DONE = True")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# noise_reachability step 12: a second round of constant-noise search and training (Kaggle, 2 x T4)

Step 12 of `noise_reachability/README.md` (expert iteration, round 2). Step 10's adapter N<t>_0 searches the states
step 10 left weak with fixed `repeat` noises; LoRA adapters R are trained from the base on step 10's recordings plus the
new successes; N0-N2 and R0-R2 are evaluated with sampled noise on fresh episodes.

**Needs step 10's output as input:** Add Input > Your Work > the step 10 notebook, its **last** version (it holds the
adapters and recordings). **Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The
wrapper check and the harness run first; a failure of either stops the run.

**This run needs about three sessions.** When step 16 says work remains, add this version's output as input as well
(**Add Input > Your Work**, only the latest step 12 version, and keep step 10's), and Save & Run All again. Step 6 must
then print `RESUMING`.

Download from the Output tab: `step12_results.zip` (results without recordings or adapter weights, which stay in the
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

md("## 7. Step 10's output: round 1's adapters and recordings")
code('''
found = [p for p in sorted(glob.glob("/kaggle/input/**/step10", recursive=True))
         if os.path.exists(f"{p}/runs/N0_0/adapter/run_config.json")]
assert len(found) == 1, f"attach step 10's last version as input (found {found})"
STEP10_IN = found[0]
print("step 10 output:", STEP10_IN, "| its commits:", open(f"{STEP10_IN}/repo_commit.txt").read().split())
for t in TASKS:
    for s in TRAIN_SEEDS:
        assert os.path.exists(f"{STEP10_IN}/runs/N{t}_{s}/adapter/run_config.json"), f"step 10 has no adapter N{t}_{s}"
    eps = [f for f in glob.glob(f"{STEP10_IN}/task_{t}/replay_data/episodes/*.npz") if not f.endswith(".tmp.npz")]
    assert eps, f"step 10 has no recordings for task {t}"
    print(f"task {t}: {len(eps)} round-1 recordings")


def n_adapter(t, s):
    return f"{STEP10_IN}/runs/N{t}_{s}/adapter"
''')

md("## 8. Helpers: parallel jobs with resume and a deadline")
code(JOBS + '''
SEARCH = f"cd {REPO} && {PY} -u noise_reachability/deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step12.py --root {OUT_ROOT}"
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


def n_states(spec):
    return len(spec.split(","))


def recordings(folder):
    return [f for f in glob.glob(f"{folder}/episodes/*.npz") if not f.endswith(".tmp.npz")]
''')

md("## 9. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code(CHECK)

md("""
## 10. Harness: step 10's N<t>_0 on four of its own evaluation episodes per task; stops the run if it fails

At least 11 of the 12 episodes must reproduce step 10's rows exactly (success and number of steps).
""")
code('''
HARNESS_DONE = False
if CHECK_DONE and not FAILURE:
    jobs = {}
    for t in TASKS:
        out = f"{task_dir(t)}/harness"
        if len(read_csv(f"{out}/per_episode_lora.csv")) < n_range(HARNESS_SEEDS):
            jobs[f"harness_task{t}"] = (f"{sfbc(t)} --stage eval --eval_policies lora --lora_dir {n_adapter(t, EXPLORER)} "
                                        f"--eval_seeds {HARNESS_SEEDS} --eval_init_states {HARNESS_STATE[t]} "
                                        f"--eval_output_dir {out} --max_episodes_per_process 10", t % 2)
    if run_stage("harness", jobs, len(TASKS)):
        HARNESS_DONE = run_gate("harness")
''')

md("""
## 11. Round 2's search: the round-1 explorer with fixed `repeat` noises on the searched states

One process per state; every successful episode is recorded to `task_<t>/search_data/episodes/` for training (failures
are not saved: training uses only successes, and 400 recordings of 30-50 MB would fill the disk).
""")
code('''
SEARCH_DONE = False
if HARNESS_DONE and not FAILURE:
    jobs, k = {}, 0
    for t in TASKS:
        for s in SEARCHED[t]:
            if len(read_csv(f"{task_dir(t)}/search/per_episode_repeat_state{s}.csv")) < n_range(SEARCH_IDS):
                jobs[f"search_task{t}_state{s}"] = (
                    f"{SEARCH} --task_id {t} --family repeat --init_state {s} --noise_ids {SEARCH_IDS} "
                    f"--lora_dir {n_adapter(t, EXPLORER)} --output_dir {task_dir(t)}/search "
                    f"--record_dir {task_dir(t)}/search_data --record_successes_only 1", k % 2)
                k += 1
    SEARCH_DONE = run_stage("search", jobs, sum(len(v) for v in SEARCHED.values()))
for t in TASKS:
    for s in SEARCHED[t]:
        rows = read_csv(f"{task_dir(t)}/search/per_episode_repeat_state{s}.csv")
        print(f"search {t}/{s}: {sum(r['success'] == '1' for r in rows)} of {len(rows)} succeeded")
sh(f"du -sh {OUT_ROOT}; df -h /kaggle/working | tail -1")
''')

md("""
## 12. Round 2's training data: step 10's recordings plus the search's, per task

Linked (not copied) under `/tmp/step12/task_<t>/round2_data/episodes/` in every session; training keeps the successes.
""")
code('''
def round2_data(t):
    return f"{ROUND2_DATA}/task_{t}/round2_data"


if SEARCH_DONE and not FAILURE:
    for t in TASKS:             # every search success must have its recording before anything trains on them
        for s in SEARCHED[t]:
            for r in read_csv(f"{task_dir(t)}/search/per_episode_repeat_state{s}.csv"):
                k = int(r["seed"]) - 1000
                name = f"repeat_n{k}_init{s:02d}_seed{r['seed']}.npz"
                if r["success"] == "1" and not os.path.exists(f"{task_dir(t)}/search_data/episodes/{name}"):
                    fail(f"search success {name} of task {t} has no recording")
if SEARCH_DONE and not FAILURE:
    for t in TASKS:
        dest = f"{round2_data(t)}/episodes"
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        for f in recordings(f"{STEP10_IN}/task_{t}/replay_data") + recordings(f"{task_dir(t)}/search_data"):
            target = f"{dest}/{os.path.basename(f)}"
            assert not os.path.exists(target), f"two recordings named {os.path.basename(f)}"
            os.symlink(f, target)
        print(f"task {t}: {len(recordings(f'{STEP10_IN}/task_{t}/replay_data'))} round-1 and "
              f"{len(recordings(f'{task_dir(t)}/search_data'))} round-2 recordings (successes and failures)")
''')

md("""
## 13. Training: R<t>_<s> from the base, tasks 0, 6, 7 x training seeds 0-2

Two adapters at a time, one per T4. A run starts only if the session can finish it (1.25 x the measured time plus 10
minutes); an unfinished run restarts from scratch in the next session.
""")
code('''
TIMING = Path(OUT_ROOT, "train_timing.json")
if SEARCH_DONE and not FAILURE and not TIMING.exists() and hours_left() > 0.5:
    r = sh(f"{sfbc(7)} --stage train --seed 0 --selection success --data_dir {round2_data(7)} --train_steps 20 "
           f"--batch_size {BATCH_SIZE} --log_every 5 --val_every 1000 --output_dir /tmp/step12_train_timing",
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
    t, s = name[1:].split("_")
    return (f"{sfbc(t)} --stage train --selection success --train_steps {TRAIN_STEPS} --batch_size {BATCH_SIZE} "
            f"--seed {s} --data_dir {round2_data(t)} --output_dir {OUT_ROOT}/runs/{name}")


ready = [n for n in ADAPTERS if not adapter_done(n)] if SEARCH_DONE and not FAILURE else []
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

md("## 14. Evaluation with sampled noise on fresh episodes: N0-N2 (step 10's) and every R whose adapter is ready")
code('''
def eval_out(t, policy, j):
    return f"{task_dir(t)}/eval/{policy}/part{j}"


def lora_of(t, policy):
    return n_adapter(t, policy[1:]) if policy.startswith("N") else f"{OUT_ROOT}/runs/{adapter_of(t, policy)}/adapter"


def eval_cmd(t, policy, j):
    return (f"{sfbc(t)} --stage eval --eval_policies lora --lora_dir {lora_of(t, policy)} "
            f"--eval_seeds {EVAL_SEED_PARTS[t][j]} --eval_init_states {EVAL_STATES[t]} "
            f"--eval_output_dir {eval_out(t, policy, j)} --max_episodes_per_process 10")


def part_size(t, j):
    return n_range(EVAL_SEED_PARTS[t][j]) * n_states(EVAL_STATES[t])


def eval_rows(t, policy):
    return [r for j in range(len(EVAL_SEED_PARTS[t])) for r in read_csv(f"{eval_out(t, policy, j)}/per_episode_lora.csv")]


jobs, k = {}, 0
if HARNESS_DONE and not FAILURE:
    for t in TASKS:
        for policy in POLICIES:
            if policy.startswith("R") and not adapter_done(adapter_of(t, policy)):
                continue
            for j in range(len(EVAL_SEED_PARTS[t])):
                if len(read_csv(f"{eval_out(t, policy, j)}/per_episode_lora.csv")) < part_size(t, j):
                    jobs[f"eval_task{t}_{policy}_part{j}"] = (eval_cmd(t, policy, j), k % 2)
                    k += 1
    run_stage("evaluation", jobs, len(jobs))
for t in TASKS:
    print(f"task {t}:", {p: len(eval_rows(t, p)) for p in POLICIES},
          f"of {sum(part_size(t, j) for j in range(len(EVAL_SEED_PARTS[t])))} each")
ALL_DONE = (len(DONE_ADAPTERS) == len(ADAPTERS) and all(
    len(eval_rows(t, p)) == sum(part_size(t, j) for j in range(len(EVAL_SEED_PARTS[t]))) for t in TASKS for p in POLICIES))
''')

md("## 15. Analysis (when every episode is done)")
code('''
if ALL_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step12_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
''')

md("## 16. Work left")
code('''
if not ALL_DONE and not FAILURE:
    print("WORK REMAINS: add this version's output as input (keep step 10's) and Save & Run All again")
''')

md("## 17. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step12_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step12_results.zip; du -sh {OUT_ROOT}")
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
