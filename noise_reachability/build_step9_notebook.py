"""Builds noise_reachability/kaggle_step9_retrieval.ipynb (run from the repo root).

    python noise_reachability/build_step9_notebook.py <commit> [output path]

The settings, hardware, code, environment, sanity, resume, job and wrapper-check cells come from step 8's notebook, so
step 9 installs and runs as step 8 did. The noise ids of every job come from analyze_step9.py --jobs, so the library is
written in one place. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step9_retrieval.ipynb"
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
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + '''# The protocol of README.md step 9; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = [0, 4, 6, 7]
HELD_OUT = {0: [14, 27], 4: [18, 26], 6: [14, 20], 7: [38, 47]}   # step 8's eligible states it did not test
N_ACTION_STEPS = 10
OUT_NAME = "step9"
OUT_ROOT = f"/kaggle/working/{OUT_NAME}"
STOP_AT_H = 10.5                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + '''print(f"step 9 at {REPO_REF}: tasks {TASKS}, held-out states {HELD_OUT}, n_action_steps {N_ACTION_STEPS}, family repeat, "
      f"at most {120 + 542} episodes (120 harness, 142 library, 400 random), output {OUT_ROOT}")


''' + SETTINGS[end:]
HARDWARE = step8_cell("JOBS = max(1")
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
RESUME = step8_cell("repo_commit.txt")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
JOBS = swap(JOBS, "to the zip in step 15", "to the zip in step 13")
CHECK = step8_cell("CHECK_DONE = True")
HARNESS = step8_cell("HARNESS_DONE = CHECK_DONE and")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 9: do noises that solved one resistant state help on other states of their task? (Kaggle, 2 x T4)

Step 9 of `noise_reachability/README.md` (design fixed 2026-11-03). The base `lerobot/smolvla_libero` at `a24f1f3804`,
`n_action_steps=10`, family `repeat`, tasks 0, 4, 6, 7: a library of earlier hit noises against fresh noises on eight
held-out resistant states. No training and no dataset input.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The wrapper check runs first, then
the harness: a failure of either stops the run. Then the scene checks, the library and random episodes, and the analysis.
If anything stops the run, step 13 still writes the zip, then reports the failure as an error.

**One session should do (about 3.4 h, an estimate).** If step 12 says work remains, add this version's output as input
(**Add Input > Your Work**), and only that version, and Save & Run All again: finished episodes are skipped. Step 6 must
then print `RESUMING`.

Download from the Output tab: `step9_results.zip`.
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
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step9.py --root {OUT_ROOT}"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


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
code(CHECK)

md("## 9. Harness: the base on seeds 0-2, per task; stops the run if it fails")
code(HARNESS)

md("""
## 10. Scene checks of the eight held-out states

A state that fails is dropped, not replaced: it is written to `scene_state<i>.json`, and the report lists it.
""")
code('''
SCENES_DONE = HARNESS_DONE
if HARNESS_DONE:
    for t in TASKS:
        for s in HELD_OUT[t]:
            if SCENES_DONE and not scene_checked(t, s):
                SCENES_DONE = False
''')

md("## 11. Library and random noises on every held-out state read")
code('''
TESTS_DONE = False
plan = listed("--jobs") if SCENES_DONE else None
if plan is not None:
    jobs = {}
    for t, s, arm, ids in plan:
        out = f"{task_dir(t)}/test/{arm}"
        if len(read_csv(f"{out}/per_episode_repeat_state{s}.csv")) < len(ids.split(",")):
            jobs[f"test_{arm}_task{t}_state{s}"] = (
                f"{SEARCH} --task_id {t} --family repeat --init_state {s} --noise_ids {ids} --output_dir {out}", len(jobs) % 2)
    print(f"{len(plan)} jobs planned: {[(t, s, arm, len(ids.split(','))) for t, s, arm, ids in plan]}", flush=True)
    TESTS_DONE = run_stage("tests", jobs, len(plan))
''')

md("## 12. Analysis (when every episode is done)")
code('''
if TESTS_DONE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step9_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
elif not FAILURE:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 13. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step9_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step9_results.zip; du -sh {OUT_ROOT}")
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
