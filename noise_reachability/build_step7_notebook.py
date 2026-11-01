"""Builds noise_reachability/kaggle_step7_deep_noise.ipynb (run from the repo root).

    python noise_reachability/build_step7_notebook.py <commit> [output path]

The settings, hardware, code, environment, sanity, resume and job cells come from step 6's notebook, so step 7 installs
and runs as step 6 did. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step7_deep_noise.ipynb"
step6 = json.load(open(REPO / "noise_reachability/kaggle_step6_replication.ipynb"))


def step6_cell(marker):
    found = [c for c in step6["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


SETTINGS = step6_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, step6_cell("T_SESSION = time.time()").split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 6")] + f'''# The protocol of README.md step 7; nothing here is meant to be edited. Step 12's analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
SYSTEMATIC = {{4: 2, 6: 8, 8: 4}}       # task: systematic init state
CONTROL = {{4: 5, 6: 0, 8: 8}}          # task: control init state
FAMILIES = ["sampled", "full", "repeat"]
N_ACTION_STEPS = 10
NOISE_IDS = "0-69"
CONTROL_IDS = "0-34"
JITTER_SEEDS = "5000-5003"
OUT_NAME = "step7"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 10.5                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 6 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + '''print(f"step 7 at {REPO_REF}: systematic {SYSTEMATIC}, control {CONTROL}, n_action_steps {N_ACTION_STEPS}, families {FAMILIES}, "
      f"at most {90 + 315 + 630 + 72} episodes (90 harness, 315 control, 630 search, up to 72 confirmation), output {OUT_ROOT}")


''' + SETTINGS[end:]
HARDWARE = step6_cell("JOBS = max(1")
CODE = step6_cell("log -1 --format='%H %s'")
ENVIRONMENT = step6_cell("uv venv")
SANITY = swap(step6_cell("RUN_ENV = dict(os.environ"), 'MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"\n', "")
RESUME = step6_cell("repo_commit.txt")
JOBS = step6_cell("def run_jobs(jobs")
JOBS = swap(JOBS[:JOBS.index("LABEL_FILES = (")], "to the zip in step 12", "to the zip in step 13")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 7: does deep noise reach systematic states beyond task 0? (Kaggle, 2 x T4)

Step 7 of `noise_reachability/README.md` (design fixed 2026-11-01). Step 5a's search with the base `lerobot/smolvla_libero`
at `a24f1f3804`, `n_action_steps=10`, on task 4 state 2, task 6 state 8 and task 8 state 4. No training and no dataset input.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The wrapper check runs first, then
the harness and the controls: a failure of any stops the run. Then the search, the confirmation and the analysis. If anything stops
the run, step 13 still writes the zip, then reports the failure as an error.

**If step 12 says work remains**, add this version's output as input (**Add Input > Your Work**) and Save & Run All
again: finished episodes are skipped.

Download from the Output tab: `step7_results.zip`.
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
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step7.py --root {OUT_ROOT}"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


def search_jobs(state_of, ids):
    """One job per (task, family); each covers one init state."""
    jobs = {}
    for t, s in state_of.items():
        for f in FAMILIES:
            if len(read_csv(f"{task_dir(t)}/per_episode_{f}_state{s}.csv")) < n_range(ids):
                jobs[f"{f}_task{t}_state{s}"] = (f"{SEARCH} --task_id {t} --family {f} --init_state {s} --noise_ids {ids} "
                                                 f"--output_dir {task_dir(t)}", len(jobs) % 2)
    return jobs


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
''')

md("## 8. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code('''
CHECK_DONE = True
for t in SYSTEMATIC:
    r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
           + (" --scene_varies" if t == 8 else ""), check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"check_step5a failed on task {t}")
        CHECK_DONE = False
''')

md("## 9. Harness: the base on seeds 0-2, per task; stops the run if it fails")
code('''
jobs = {}
for t in SYSTEMATIC:
    for s in (0, 1, 2):
        out = f"{task_dir(t)}/harness/seed{s}"
        if len(read_csv(f"{out}/per_episode_base.csv")) < 10:
            jobs[f"harness_task{t}_seed{s}"] = (
                f"cd {REPO} && {PY} -u noise_reachability/eval_noise_selection.py --task_id {t} --policies base "
                f"--policy_path {POLICY_PATH} --seeds {s} --init_states 0-9 --n_action_steps {N_ACTION_STEPS} --output_dir {out}",
                len(jobs) % 2)
HARNESS_DONE = CHECK_DONE and run_stage("harness", jobs, 9) and run_gate("harness")
''')

md("## 10. Control states; stops the run if sampled noise fails on one")
code('''
CONTROL_DONE = HARNESS_DONE and run_stage("control", search_jobs(CONTROL, CONTROL_IDS), 9) and run_gate("control")
''')

md("## 11. Search, then confirmation of hits by jitter")
code('''
SEARCH_DONE = CONTROL_DONE and run_stage("search", search_jobs(SYSTEMATIC, NOISE_IDS), 9)
CONFIRM_DONE = False
if SEARCH_DONE:
    r = sh(ANALYZE + " --hits", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail("the hit list failed")
    else:
        jobs = {}
        hits = [line.split() for line in r.stdout.strip().splitlines()]
        for t, f, s, k in hits:
            if len(read_csv(f"{task_dir(t)}/confirm_{f}_n{k}_state{s}.csv")) < 4:
                jobs[f"confirm_{f}_task{t}_n{k}"] = (f"{SEARCH} --task_id {t} --family {f} --init_state {s} --noise_ids {k} "
                                                     f"--jitter_seeds {JITTER_SEEDS} --output_dir {task_dir(t)}", len(jobs) % 2)
        print(f"{len(hits)} hits to confirm", flush=True)
        CONFIRM_DONE = run_stage("confirm", jobs, len(hits))
''')

md("## 12. Analysis (when every episode is done)")
code('''
if CONFIRM_DONE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step7_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
elif not FAILURE:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 13. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step7_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step7_results.zip; du -sh {OUT_ROOT}")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": step6["metadata"], "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, len(cells), "cells")
