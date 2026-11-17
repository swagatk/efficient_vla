"""Builds noise_reachability/kaggle_step15_unreachable.ipynb (run from the repo root).

    python noise_reachability/build_step15_notebook.py <commit> [output path]

Step 15 of README.md: are 6/14 and 6/20 unreachable, and did distillation lose 7/38's constant-noise solutions? The
settings, hardware, code, environment, sanity, resume, job and wrapper-check cells come from step 8's notebook, as the
builders of steps 11-14 take them, so step 15 installs and runs as they did. States, ids, folders and the adapters'
sha256 are analyze_step15.py's constants, written into the settings cell here, so they are fixed in one place. Edit this
builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step15 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step15_unreachable.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_HARNESS = len(a.HARNESS_BASE_IDS) + 4
N_A = len(a.STATES_A) * len(a.A_IDS)
N_B = len(a.B_POLICIES) * len(a.B_IDS)

SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 15; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = [{a.TASK_A}, {a.TASK_B}]                    # the wrapper check runs on both
N_ACTION_STEPS = 10
TASK_A, STATES_A = {a.TASK_A}, {list(a.STATES_A)}        # test A: never solved by any noise
A_PARTS = {list(a.A_PARTS)}   # fresh repeat ids, two jobs per state
TASK_B, STATE_B = {a.TASK_B}, {a.STATE_B}                # test B: the base against N7_0-N7_2 on the same noises
B_IDS = "{a.ids_spec(a.B_IDS)}"
B_POLICIES = {a.B_POLICIES}
SCENE_STATES = {dict(a.SCENE_STATES)}
HARNESS_BASE_IDS = "{a.ids_spec(a.HARNESS_BASE_IDS)}"   # step 9's 7/38 episodes
HARNESS_N_STATE, HARNESS_N_SEEDS = {a.HARNESS_N_STATE}, "{a.HARNESS_N_SEEDS}"   # step 10's N7_0 episodes
N7_SHA256 = {dict(a.N7_SHA256)}
OUT_NAME = "step15"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 11.0                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"step 15 at {{REPO_REF}}: test A {{TASK_A}}/{{STATES_A}} ids {{A_PARTS}}, test B {{TASK_B}}/{{STATE_B}} ids "
      f"{{B_IDS}} for {{B_POLICIES}}, at most {N_HARNESS + N_A + N_B} episodes ({N_HARNESS} harness, {N_A} test A, {N_B} "
      f"test B) plus replays, stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


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


md("""
# noise_reachability step 15: are 6/14 and 6/20 unreachable, and did distillation lose 7/38's solutions? (Kaggle, 2 x T4)

Step 15 of `noise_reachability/README.md`. No training. *Test A:* the base with 400 fresh `repeat` noises on each of 6/14
and 6/20, every success replayed once. *Test B:* the base and step 10's N7_0-N7_2 on the same 200 fresh `repeat` noises
on 7/38.

**Needs one input:** Add Input > Your Work > the step 10 notebook, its **last** version (adapters N7_0-N7_2; their sha256
are checked). **Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The wrapper
check, the scene checks and the harness run first; a failure stops the run.

**This run needs two sessions (about 11 h and 1.5 h, an estimate).** When step 14 says work remains, add this version's output as input
as well (**Add Input > Your Work**, only the latest step 15 version; keep step 10's), and Save & Run All again. Step 6
must then print `RESUMING`.

Download from the Output tab: `step15_results.zip`.
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

md("## 7. Input: step 10's adapters N7_0-N7_2 (sha256 against step 10's archive manifest)")
code('''
found = [p for p in sorted(glob.glob("/kaggle/input/**/step10", recursive=True))
         if os.path.exists(f"{p}/runs/N{TASK_B}_0/adapter/run_config.json")]
assert len(found) == 1, f"attach step 10's last version as input (found {found})"
STEP10_IN = found[0]
print("step 10 output:", STEP10_IN, "| its commits:", open(f"{STEP10_IN}/repo_commit.txt").read().split())


def n_adapter(policy):
    return f"{STEP10_IN}/runs/N{TASK_B}_{policy[1:]}/adapter"


for policy, files in N7_SHA256.items():
    for name, want in files.items():
        got = hashlib.sha256(Path(n_adapter(policy), name).read_bytes()).hexdigest()
        assert got == want, f"{n_adapter(policy)}/{name}: sha256 {got}, expected step 10's {want}"
print(f"{len(N7_SHA256)} adapters: sha256 equal to step 10's archive manifest")
''')

md("## 8. Helpers: parallel jobs with resume and a deadline")
code(JOBS + '''
SEARCH = f"cd {REPO} && {PY} -u noise_reachability/deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS} --family repeat"
SCENE = f"cd {REPO} && {PY} -u noise_reachability/check_scene.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step15.py --root {OUT_ROOT}"
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"


def task_dir(task):
    return f"{OUT_ROOT}/task_{task}"


def search_cmd(task, state, ids, out, policy="base"):
    lora = "" if policy == "base" else f" --lora_dir {n_adapter(policy)}"
    return f"{SEARCH} --task_id {task} --init_state {state} --noise_ids {ids} --output_dir {out}{lora}"


def harness_n_cmd():
    return (f"cd {REPO} && {PY} -u {MAIN} --task_id {TASK_B} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH} "
            f"--stage eval --eval_policies lora --lora_dir {n_adapter('N0')} --eval_seeds {HARNESS_N_SEEDS} "
            f"--eval_init_states {HARNESS_N_STATE} --eval_output_dir {OUT_ROOT}/harness/N0 --max_episodes_per_process 10")


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


def scene_checked(task, state):
    """One state's scene check, in its own process, one at a time."""
    if hours_left() <= 0:
        return False
    r = sh(f"{SCENE} --task_id {task} --init_state {state} --output_dir {task_dir(task)}/scene", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"the scene check failed to run on task {task} state {state}")
        return False
    return True


def scene_passed(task, state):
    p = Path(task_dir(task), "scene", f"scene_state{state}.json")
    return p.exists() and bool(json.loads(p.read_text())["same"])
''')

md("## 9. Wrapper check on each task's real model (a few minutes); stops the run if it fails")
code(CHECK)

md("""
## 10. Scene checks of 6/14, 6/20 and 7/38

A state that fails is dropped, not replaced: it is written to `scene_state<i>.json`, and the report lists it.
""")
code('''
SCENES_DONE = CHECK_DONE and not FAILURE
for t, states in SCENE_STATES.items():
    for s in states:
        if SCENES_DONE and not scene_checked(t, s):
            SCENES_DONE = False
print({f"{t}/{s}": scene_passed(t, s) for t, states in SCENE_STATES.items() for s in states})
''')

md("""
## 11. Harness: the base on 10 of step 9's 7/38 episodes, N7_0 on 4 of step 10's 7/33 episodes; stops the run if it fails

At least 13 of the 14 episodes must reproduce the committed rows exactly (success and number of steps).
""")
code('''
HARNESS_DONE = False
if SCENES_DONE and not FAILURE:
    jobs = {}
    if len(read_csv(f"{OUT_ROOT}/harness/base/per_episode_repeat_state{STATE_B}.csv")) < n_ids(HARNESS_BASE_IDS):
        jobs["harness_base"] = (search_cmd(TASK_B, STATE_B, HARNESS_BASE_IDS, f"{OUT_ROOT}/harness/base"), 0)
    if len(read_csv(f"{OUT_ROOT}/harness/N0/per_episode_lora.csv")) < n_ids(HARNESS_N_SEEDS):
        jobs["harness_N0"] = (harness_n_cmd(), 1)
    if run_stage("harness", jobs, 2):
        HARNESS_DONE = run_gate("harness")
''')

md("""
## 12. Tests A and B: the base on 6/14 and 6/20 (400 noises each, in two parts), the base and N7_0-N7_2 on 7/38 (200 each)

Eight jobs of 200 episodes, four at a time; states that failed the scene check are skipped.
""")
code('''
def test_jobs():
    jobs = {}
    for s in STATES_A:
        if not scene_passed(TASK_A, s):
            continue
        for j, ids in enumerate(A_PARTS):
            out = f"{task_dir(TASK_A)}/search/part{j}"
            if len(read_csv(f"{out}/per_episode_repeat_state{s}.csv")) < n_ids(ids):
                jobs[f"testA_state{s}_part{j}"] = (search_cmd(TASK_A, s, ids, out), len(jobs) % 2)
    if scene_passed(TASK_B, STATE_B):
        for policy in B_POLICIES:
            out = f"{task_dir(TASK_B)}/search/{policy}"
            if len(read_csv(f"{out}/per_episode_repeat_state{STATE_B}.csv")) < n_ids(B_IDS):
                jobs[f"testB_{policy}"] = (search_cmd(TASK_B, STATE_B, B_IDS, out, policy), len(jobs) % 2)
    return jobs


TESTS_DONE = False
if HARNESS_DONE and not FAILURE:
    TESTS_DONE = run_stage("tests A and B", test_jobs(), len(STATES_A) * len(A_PARTS) + len(B_POLICIES))
''')

md("## 13. Replays: every success of test A re-run once with its noise id")
code('''
def listed(flag):
    """The lines of an analysis listing as lists of words, or None after failing the run."""
    r = sh(f"{ANALYZE} {flag}", check=False, env=RUN_ENV)
    if r.returncode != 0:
        fail(f"the listing {flag} failed")
        return None
    return [line.split() for line in r.stdout.strip().splitlines()]


REPLAYS_DONE = False
plan = listed("--replay_jobs") if TESTS_DONE and not FAILURE else None
if plan is not None:
    jobs = {}
    for t, s, ids in plan:
        out = f"{task_dir(int(t))}/replay"
        if len(read_csv(f"{out}/per_episode_repeat_state{s}.csv")) < n_ids(ids):
            jobs[f"replay_state{s}"] = (search_cmd(int(t), int(s), ids, out), len(jobs) % 2)
    print("successes to replay:", {s: ids for _, s, ids in plan} or "none", flush=True)
    REPLAYS_DONE = run_stage("replays", jobs, len(plan))
''')

md("## 14. Analysis (when every episode is done)")
code('''
if REPLAYS_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step15_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
elif not FAILURE:
    print("WORK REMAINS: add this version's output as input (keep step 10's) and Save & Run All again")
''')

md("## 15. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step15_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step15_results.zip; du -sh {OUT_ROOT}")
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
