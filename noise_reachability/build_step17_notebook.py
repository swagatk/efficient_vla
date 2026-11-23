"""Builds noise_reachability/kaggle_step17_features.ipynb (run from the repo root).

    python noise_reachability/build_step17_notebook.py <commit> [output path]

Step 17 of README.md: SmolVLA's prefix features at the first inference of every map state of step 16 (tasks 0, 4, 6
and 7, states 0-29), from one episode of id 10000 per state (and id 10001 on four states), with the step-16 stack. The
episodes are the run's harness: each must reproduce step 16's committed row. The settings, hardware, code,
environment, sanity, resume, job and wrapper-check cells come from step 8's notebook, as step 16's builder takes them.
The tasks, states and ids are analyze_step17.py's constants. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step17 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step17_features.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_STATES = len(a.TASKS) * len(a.MAP_STATES)
N_EPISODES = N_STATES + len(a.DUPLICATE_STATES)
IDS = {f"{t}/{s}": a.ids_spec(a.ids_of(t, s)) for t in a.TASKS for s in a.MAP_STATES}

SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 17; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {list(a.TASKS)}                 # the wrapper check runs on each
N_ACTION_STEPS = 10
MAP_STATES = {list(a.MAP_STATES)}
IDS = {json.dumps(IDS)}   # "task/state": repeat ids (id {a.DUPLICATE_ID} also on {len(a.DUPLICATE_STATES)} states)
OUT_NAME = "step17"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 4.0                    # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"step 17 at {{REPO_REF}}: first-inference features, tasks {{TASKS}} x states 0-29, {N_EPISODES} episodes "
      f"(id {a.FEATURE_ID} on every state, id {a.DUPLICATE_ID} on {len(a.DUPLICATE_STATES)}); stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


''' + SETTINGS[end:]
HARDWARE = step8_cell("JOBS = max(1")
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
RESUME = step8_cell("repo_commit.txt")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
CHECK = step8_cell("CHECK_DONE = True")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md(f"""
# noise_reachability step 17: first-inference features of the step-16 map states (Kaggle, 2 x T4)

Step 17 of `noise_reachability/README.md`. No training and no new map: one episode of the base with `repeat` id
{a.FEATURE_ID} on each of the {N_STATES} map states (and id {a.DUPLICATE_ID} on {len(a.DUPLICATE_STATES)} of them), which stores SmolVLA's prefix
features at the episode's first inference. Each episode must reproduce step 16's committed row: at least {a.MIN_SAME} of
{N_STATES} states, and the duplicate id must give the same features bit for bit; otherwise the run stops and is not read.

**Needs no input. Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. One session
(about 1.5 h, an estimate). If step 10 says work remains, add this version's output as input (**Add Input > Your
Work**, only the latest step 17 version) and Save & Run All again; step 6 must then print `RESUMING`.

Download from the Output tab: `step17_results.zip`.
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
SEARCH = f"cd {REPO} && {PY} -u noise_reachability/deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS} --family repeat"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step17.py --root {OUT_ROOT}"


def feature_dir(task):
    return f"{OUT_ROOT}/task_{task}/features"


def n_ids(spec):
    return sum(int(p.split("-")[1]) - int(p.split("-")[0]) + 1 if "-" in p else 1 for p in spec.split(","))


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
''')

md("## 8. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code(CHECK)

md(f"""
## 9. Feature episodes: one job per map state ({N_STATES} jobs), four at a time

Each job runs `deep_noise_search.py --family repeat --feature_dir ...` on one state; the features of the first
inference go to `task_<t>/features/json/`, the episode's row to `task_<t>/features/per_episode_repeat_state<i>.csv`.
""")
code('''
FEATURES_DONE = False
if CHECK_DONE and not FAILURE:
    jobs = {}
    for t in TASKS:
        out = feature_dir(t)
        for s in MAP_STATES:
            ids = IDS[f"{t}/{s}"]
            if len(read_csv(f"{out}/per_episode_repeat_state{s}.csv")) < n_ids(ids):
                jobs[f"features_task{t}_state{s}"] = (f"{SEARCH} --task_id {t} --init_state {s} --noise_ids {ids} "
                                                     f"--output_dir {out} --feature_dir {out}/json", len(jobs) % 2)
    FEATURES_DONE = run_stage("feature episodes", jobs, len(TASKS) * len(MAP_STATES))
''')

md(f"""
## 10. Feature check, report and features file

The gate: at least {a.MIN_SAME} of {N_STATES} episodes reproduce step 16's committed rows, and the duplicate id's features are
identical on all {len(a.DUPLICATE_STATES)} states. The test U1 itself (`step17_g1b.py`) runs locally on `step17_features.json`.
""")
code('''
GATE_DONE = False
if FEATURES_DONE and not FAILURE:
    if sh(f"{ANALYZE} --gate features", check=False, env=RUN_ENV).returncode != 0:
        fail("the feature check failed: investigate before reading anything")
    elif sh(f"{ANALYZE} --out {OUT_ROOT}/step17_report.txt --features {OUT_ROOT}/step17_features.json",
            check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
    else:
        GATE_DONE = True
if FAILURE:
    print("FAILED:", FAILURE)
elif GATE_DONE:
    print("STEP 17 DONE: download step17_results.zip")
else:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 11. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step17_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step17_results.zip; du -sh {OUT_ROOT}")
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
