"""Builds noise_reachability/kaggle_step16_crossed_map.ipynb (run from the repo root).

    python noise_reachability/build_step16_notebook.py <commit> [output path]

Step 16 of README.md: the crossed constant-noise map. The base with the same fresh `repeat` noise ids on every state
0-29 of tasks 0, 4, 6 and 7, in five parts of 10 ids; parts 2-5 run only once part 1 is complete and its gate passes.
The settings, hardware, code, environment, sanity, resume, job and wrapper-check cells come from step 8's notebook, as
the builders of steps 11-15 take them, so step 16 installs and runs as they did. Tasks, states, ids and the harness are
analyze_step16.py's constants, written into the settings cell here, so they are fixed in one place. Edit this builder,
not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step16 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step16_crossed_map.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_STATES = len(a.TASKS) * len(a.MAP_STATES)
N_IDS = len(a.parse_ids(a.PARTS[0]))
assert all(len(a.parse_ids(p)) == a.IDS_PER_PROCESS == N_IDS for p in a.PARTS), "one process per job"
N_PART = N_STATES * N_IDS
N_HARNESS = len(a.HARNESS_IDS)

SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 16; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {list(a.TASKS)}                 # the wrapper check runs on each
N_ACTION_STEPS = 10
MAP_STATES = {list(a.MAP_STATES)}
PARTS = {list(a.PARTS)}   # fresh repeat ids; parts 2-5 only after part 1's gate passes
HARNESS_TASK, HARNESS_STATE = {a.HARNESS_TASK}, {a.HARNESS_STATE}
HARNESS_IDS = "{a.ids_spec(a.HARNESS_IDS)}"   # 12 of step 15's committed 7/38 episodes
OUT_NAME = "step16"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 11.0                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"step 16 at {{REPO_REF}}: crossed map, tasks {{TASKS}} x states 0-29 x repeat ids {{PARTS}}, {N_PART} episodes "
      f"per part ({N_STATES} states x {N_IDS} ids), {N_HARNESS} harness episodes, {N_STATES} scene checks; parts 2-5 only after "
      f"part 1's gate; stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


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
# noise_reachability step 16: the crossed constant-noise map (Kaggle, 2 x T4)

Step 16 of `noise_reachability/README.md`. No training. The base policy with the same fresh `repeat` noises on every
state 0-29 of tasks {", ".join(map(str, a.TASKS))}: five parts of {N_IDS} ids ({N_PART} episodes each). Parts 2-5 run only once
part 1 is complete and its pre-registered gate passes.

**Needs no input on the first session. Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing
needs editing. The wrapper check, the scene checks and the harness run first; a failure stops the run.

**This run needs several sessions (part 1 about 9 h, each later part about 7.5 h; estimates).** When step 13 says
work remains, add this version's output as input (**Add Input > Your Work**, only the latest step 16 version) and Save &
Run All again. Step 6 must then print `RESUMING`.

Download from the Output tab: `step16_results.zip`.
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
SCENE = f"cd {REPO} && {PY} -u noise_reachability/check_scene.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step16.py --root {OUT_ROOT}"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


def search_cmd(task, state, ids, out):
    return f"{SEARCH} --task_id {task} --init_state {state} --noise_ids {ids} --output_dir {out}"


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


def run_gate(which):
    if sh(f"{ANALYZE} --gate {which}", check=False, env=RUN_ENV).returncode != 0:
        fail(f"the {which} gate failed: investigate before spending more")
        return False
    return True


def scene_file(task, state):
    return Path(task_dir(task), "scene", f"scene_state{state}.json")


def scene_passed(task, state):
    p = scene_file(task, state)
    return p.exists() and bool(json.loads(p.read_text())["same"])
''')

md("## 8. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code(CHECK)

md(f"""
## 9. Scene checks of the {N_STATES} map states, four at a time

A state that fails is dropped, not replaced: it is written to `scene_state<i>.json`, and the report lists it.
""")
code('''
SCENES_DONE = False
if CHECK_DONE and not FAILURE:
    jobs = {}
    for t in TASKS:
        for s in MAP_STATES:
            if not scene_file(t, s).exists():
                jobs[f"scene_task{t}_state{s}"] = (f"{SCENE} --task_id {t} --init_state {s} --output_dir {task_dir(t)}/scene",
                                                  len(jobs) % 2)
    SCENES_DONE = run_stage("scene checks", jobs, len(TASKS) * len(MAP_STATES))
    if SCENES_DONE:
        dropped = [f"{t}/{s}" for t in TASKS for s in MAP_STATES if not scene_passed(t, s)]
        print("scene checks done; dropped:", dropped or "none", flush=True)
''')

md(f"""
## 10. Harness: the base on 12 of step 15's 7/38 episodes; stops the run if it fails

At least {a.HARNESS_MIN_SAME} of the {N_HARNESS} episodes must reproduce the committed rows exactly (success and number of steps).
""")
code('''
HARNESS_DONE = False
if SCENES_DONE and not FAILURE:
    jobs = {}
    if len(read_csv(f"{OUT_ROOT}/harness/base/per_episode_repeat_state{HARNESS_STATE}.csv")) < n_ids(HARNESS_IDS):
        jobs["harness_base"] = (search_cmd(HARNESS_TASK, HARNESS_STATE, HARNESS_IDS, f"{OUT_ROOT}/harness/base"), 0)
    if run_stage("harness", jobs, 1):
        HARNESS_DONE = run_gate("harness")
''')

md("""
## 11. The map, part by part: one job per state and part (10 episodes, one process), four at a time

Part 1 first. Parts 2-5 start only when part 1 is complete and its gate passes (at least 10% of the 120 states with
both a success and a failure).
""")
code('''
def part_jobs(j):
    jobs = {}
    for t in TASKS:
        for s in MAP_STATES:
            if not scene_passed(t, s):
                continue
            out = f"{task_dir(t)}/map/part{j}"
            if len(read_csv(f"{out}/per_episode_repeat_state{s}.csv")) < n_ids(PARTS[j]):
                jobs[f"map_part{j}_task{t}_state{s}"] = (search_cmd(t, s, PARTS[j], out), len(jobs) % 2)
    return jobs


def part1_gate_passes():
    r = sh(f"{ANALYZE} --parts 1 --out {OUT_ROOT}/step16_part1_report.txt", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail("the part-1 analysis failed")
        return False
    return json.loads(Path(f"{OUT_ROOT}/step16_part1_report.txt.json").read_text())["part1_gate"]["passes"]


PARTS_DONE = 0
if HARNESS_DONE and not FAILURE:
    for j in range(len(PARTS)):
        if j == 1 and not part1_gate_passes():
            print("part 1's gate fails: parts 2-5 are not run (README step 16)", flush=True)
            break
        if not run_stage(f"map part {j + 1}", part_jobs(j), len(TASKS) * len(MAP_STATES)):
            break
        PARTS_DONE = j + 1
print(f"map parts complete: {PARTS_DONE} of {len(PARTS)}", flush=True)
''')

md("## 12. Report and table of the complete parts")
code('''
if PARTS_DONE and not FAILURE:
    # one name per number of parts, so a table of fewer parts is never taken for the final one (step16_g1.py refuses it)
    if sh(f"{ANALYZE} --parts {PARTS_DONE} --out {OUT_ROOT}/step16_report_parts{PARTS_DONE}.txt "
          f"--table {OUT_ROOT}/step16_table_parts{PARTS_DONE}.csv",
          check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
''')

md("## 13. Work left?")
code('''
gate_file = Path(f"{OUT_ROOT}/step16_part1_report.txt.json")
stopped_by_gate = gate_file.exists() and not json.loads(gate_file.read_text())["part1_gate"]["passes"]
if FAILURE:
    print("FAILED:", FAILURE)
elif PARTS_DONE == len(PARTS) or stopped_by_gate:
    print("STEP 16 DONE: download step16_results.zip")
else:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 14. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step16_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step16_results.zip; du -sh {OUT_ROOT}")
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
