"""Builds noise_reachability/kaggle_step8_resistant_states.ipynb (run from the repo root).

    python noise_reachability/build_step8_notebook.py <commit> [output path]

The settings, hardware, code, environment, sanity, resume and job cells come from step 7's notebook, so step 8 installs
and runs as step 7 did. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step8_resistant_states.ipynb"
step7 = json.load(open(REPO / "noise_reachability/kaggle_step7_deep_noise.ipynb"))


def step7_cell(marker):
    found = [c for c in step7["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


SETTINGS = step7_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 7")] + '''# The protocol of README.md step 8; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = [0, 4, 6, 7]
CONTROL = {0: 9, 4: 5, 6: 0, 7: 4}     # task: control init state
FAMILIES = ["sampled", "repeat"]
N_ACTION_STEPS = 10
CONTROL_IDS = "35-69"
SCREEN_TASKS = [0, 4, 6]
SCREEN_STATES = range(10, 30)
DIRECT = {7: [33, 38, 47]}             # step 3b's only task-7 states with 0 hits in 10-49
FIRST_IDS = "400-407"
SECOND_IDS = "408-431"
ALL_SCREEN_IDS = "400-431"
TEST1 = (6, 8)
JITTER_SEEDS = "5000-5003"
OUT_NAME = "step8"
OUT_ROOT = f"/kaggle/working/{OUT_NAME}"
STOP_AT_H = 10.5                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 7 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + '''print(f"step 8 at {REPO_REF}: tasks {TASKS}, control {CONTROL}, test 1 state {TEST1}, n_action_steps {N_ACTION_STEPS}, families {FAMILIES}, "
      f"at most {120 + 280 + 600 + 1056 + 1200 + 60} episodes (120 harness, 280 control, 600 test 1, up to 1056 screen, up to 1200 test 2, "
      f"up to 60 confirmation), output {OUT_ROOT}")


''' + SETTINGS[end:]
HARDWARE = step7_cell("JOBS = max(1")
CODE = step7_cell("log -1 --format='%H %s'")
ENVIRONMENT = step7_cell("uv venv")
SANITY = step7_cell("RUN_ENV = dict(os.environ")
RESUME = swap(step7_cell("repo_commit.txt"), """for prev in sorted(glob.glob(f"/kaggle/input/**/{OUT_NAME}", recursive=True)):
    if os.path.isdir(prev) and os.path.exists(f"{prev}/repo_commit.txt"):
        print("copying", prev)
        shutil.copytree(prev, OUT_ROOT, dirs_exist_ok=True)
""", """previous = [p for p in sorted(glob.glob(f"/kaggle/input/**/{OUT_NAME}", recursive=True))
            if os.path.isdir(p) and os.path.exists(f"{p}/repo_commit.txt")]
assert len(previous) <= 1, f"attach only the latest version's output, found {previous}"
for prev in previous:
    print("RESUMING from", prev)
    shutil.copytree(prev, OUT_ROOT, dirs_exist_ok=True)
if not previous:
    print("FRESH START: no earlier output attached. If this is the second session, stop and add it as input.", flush=True)
""")
JOBS = step7_cell("def run_jobs(jobs")
JOBS = swap(JOBS[:JOBS.index("SEARCH = f")], "to the zip in step 13", "to the zip in step 15")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 8: does fixed noise beat sampling on states that resist sampling? (Kaggle, 2 x T4)

Step 8 of `noise_reachability/README.md` (design fixed 2026-11-02). The base `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, tasks 0, 4, 6, 7: task 6 state 8 on fresh noises, and states screened from 10-49. No training and no
dataset input.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The wrapper check runs first, then
the harness and the controls: a failure of any stops the run. Then the screen, the selection, the two tests, the confirmation and
the analysis. If anything stops the run, step 15 still writes the zip, then reports the failure as an error.

**This run needs two sessions.** When step 14 says work remains, add this version's output as input (**Add Input > Your
Work**), and only that version, and Save & Run All again: finished episodes are skipped. Step 6 must print `RESUMING`.

Download from the Output tab: `step8_results.zip`.
""")

md("## 1. Settings")
code(SETTINGS)

md("## 2. Hardware")
code(HARDWARE)

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
SCENE = f"cd {REPO} && {PY} -u noise_reachability/check_scene.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step8.py --root {OUT_ROOT}"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


def episode_job(jobs, name, task, family, state, ids, folder, wanted):
    """Adds one (family, state) job unless its file already has the episodes wanted."""
    out = f"{task_dir(task)}/{folder}"
    if len(read_csv(f"{out}/per_episode_{family}_state{state}.csv")) < wanted:
        jobs[name] = (f"{SEARCH} --task_id {task} --family {family} --init_state {state} --noise_ids {ids} --output_dir {out}",
                      len(jobs) % 2)


def run_stage(name, jobs, total):
    """True when the stage has no work left: a job that failed stops the run, the deadline leaves work for a next version."""
    print(f"{name}: {len(jobs)} of {total} jobs to run; {hours_left():.1f} h left", flush=True)
    if not jobs:
        return True
    if hours_left() <= 0:
        return False
    codes = run_jobs(jobs, max_parallel=4)
    if failed(codes):
        fail(f"{name} failed: {failed(codes)}")
        return False
    return all(c == 0 for c in codes.values()) and len(codes) == len(jobs)


def run_gate(which):
    if sh(f"{ANALYZE} --gate {which}", check=False, env=RUN_ENV).returncode != 0:
        fail(f"the {which} gate failed: investigate before spending more")
        return False
    return True


def listed(flag):
    """The lines of an analysis listing as lists of words, or None after failing the run."""
    r = sh(f"{ANALYZE} {flag}", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"the listing {flag} failed")
        return None
    return [line.split() for line in r.stdout.strip().splitlines()]


def scene_checked(task, state):
    """One state's scene check, in its own process, one at a time."""
    if hours_left() <= 0:
        return False
    r = sh(f"{SCENE} --task_id {task} --init_state {state} --output_dir {task_dir(task)}/scene", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"the scene check failed to run on task {task} state {state}")
        return False
    return True
''')

md("## 8. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code('''
CHECK_DONE = True
for t in TASKS:
    r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
           check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"check_step5a failed on task {t}")
        CHECK_DONE = False
''')

md("## 9. Harness: the base on seeds 0-2, per task; stops the run if it fails")
code('''
jobs = {}
for t in TASKS:
    for s in (0, 1, 2):
        out = f"{task_dir(t)}/harness/seed{s}"
        if len(read_csv(f"{out}/per_episode_base.csv")) < 10:
            jobs[f"harness_task{t}_seed{s}"] = (
                f"cd {REPO} && {PY} -u noise_reachability/eval_noise_selection.py --task_id {t} --policies base "
                f"--policy_path {POLICY_PATH} --seeds {s} --init_states 0-9 --n_action_steps {N_ACTION_STEPS} --output_dir {out}",
                len(jobs) % 2)
HARNESS_DONE = CHECK_DONE and run_stage("harness", jobs, 12) and run_gate("harness")
''')

md("## 10. Control states; stops the run if sampled noise fails on one")
code('''
jobs = {}
for t, s in CONTROL.items():
    for f in FAMILIES:
        episode_job(jobs, f"control_{f}_task{t}", t, f, s, CONTROL_IDS, "control", n_range(CONTROL_IDS))
CONTROL_DONE = HARNESS_DONE and run_stage("control", jobs, 8) and run_gate("control")
''')

md("""
## 11. Screen

Tasks whose `repeat` control failed are left out of the second part and of the tests by the analysis, so the first part is
simply run on all. Task 7's three states run all 32 episodes at once.
""")
code('''
SCREEN_DONE = False
if CONTROL_DONE:
    jobs = {}
    for t in SCREEN_TASKS:
        for s in SCREEN_STATES:
            episode_job(jobs, f"screen1_task{t}_state{s}", t, "sampled", s, FIRST_IDS, "screen", n_range(FIRST_IDS))
    for t, states in DIRECT.items():
        for s in states:
            episode_job(jobs, f"screen_task{t}_state{s}", t, "sampled", s, ALL_SCREEN_IDS, "screen", n_range(ALL_SCREEN_IDS))
    if run_stage("screen, first part", jobs, 63):
        second = listed("--second")
        if second is not None:
            jobs = {}
            for t, s in second:
                episode_job(jobs, f"screen2_task{t}_state{s}", int(t), "sampled", int(s), SECOND_IDS, "screen", n_range(ALL_SCREEN_IDS))
            SCREEN_DONE = run_stage("screen, second part", jobs, len(second))
''')

md("""
## 12. Scene checks and selection

Each state is checked before it is tested, in the selection's order, until 4 pass. A state that fails is passed over: it is
written to `scene_state<i>.json`, and the report lists it.
""")
code('''
SELECT_DONE = False
if SCREEN_DONE:
    SELECT_DONE = scene_checked(*TEST1)
    order = listed("--order") if SELECT_DONE else None
    if order is None:
        SELECT_DONE = False
    else:
        print(f"{len(order)} eligible states: {order}", flush=True)
        passed = 0
        for t, s in order:
            if passed == 4:
                break
            if not scene_checked(int(t), int(s)):
                SELECT_DONE = False
                break
            scene = json.load(open(f"{task_dir(int(t))}/scene/scene_state{s}.json"))
            passed += bool(scene["same"])
''')

md("## 13. The two tests, then confirmation of hits by jitter")
code('''
TESTS_DONE = CONFIRM_DONE = False
plan = listed("--tests") if SELECT_DONE else None
if plan is not None:
    print("tests:", plan, flush=True)
    jobs = {}
    for t, s, ids in plan:
        for f in FAMILIES:
            episode_job(jobs, f"test_{f}_task{t}_state{s}", int(t), f, int(s), ids, "test", n_range(ids))
    TESTS_DONE = run_stage("tests", jobs, 2 * len(plan))
if TESTS_DONE:
    hits = listed("--hits")
    if hits is not None:
        jobs = {}
        for t, s, k in hits:
            out = f"{task_dir(t)}/test"
            if len(read_csv(f"{out}/confirm_repeat_n{k}_state{s}.csv")) < 4:
                jobs[f"confirm_task{t}_state{s}_n{k}"] = (
                    f"{SEARCH} --task_id {t} --family repeat --init_state {s} --noise_ids {k} --jitter_seeds {JITTER_SEEDS} "
                    f"--output_dir {out}", len(jobs) % 2)
        print(f"{len(hits)} hits to confirm", flush=True)
        CONFIRM_DONE = run_stage("confirm", jobs, len(hits))
''')

md("## 14. Analysis (when every episode is done)")
code('''
if CONFIRM_DONE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step8_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
elif not FAILURE:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 15. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step8_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step8_results.zip; du -sh {OUT_ROOT}")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": step7["metadata"], "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, len(cells), "cells")
