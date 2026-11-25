"""Builds noise_reachability/kaggle_step18_mechanism.ipynb (run from the repo root).

    python noise_reachability/build_step18_notebook.py <commit> [output path]

Step 18 of README.md (mechanism M1): constant noise against a slowly drifting noise, lower-temperature sampling and
the reported processes, on step 18's eight resistant states. The settings, hardware, code, environment, sanity,
resume, job and wrapper-check cells come from step 8's notebook, as the builders of steps 11-17 take them, so step 18
installs and runs as they did. States, ids, conditions and the harness are analyze_step18.py's constants, written into
the settings cell here, so they are fixed in one place. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step18 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step18_mechanism.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


TASKS = sorted({t for t, _ in a.STATES})
# {label: [family, rho, sigma]} and {"t/s": [[ids of block 0], ...]} per label, from analyze_step18
CONDITIONS = {c: list(v[:3]) for c, v in a.CONDITIONS.items()}
BLOCKS = {c: {f"{t}/{s}": [a.ids_spec(b) for b in a.blocks(t, s, c)] for t, s in a.STATES} for c in a.CONDITIONS}
HARNESS_IDS = {f"{t}/{s}": a.ids_spec(a.harness_ids(t, s)) for t, s in a.STATES}
N_EPISODES = sum(len(a.condition_ids(t, s, c)) for t, s in a.STATES for c in a.CONDITIONS)
N_JOBS = sum(len(a.blocks(t, s, c)) for t, s in a.STATES for c in a.CONDITIONS)
N_HARNESS = sum(len(a.harness_ids(t, s)) for t, s in a.STATES)
N_BLOCKS = max(len(a.blocks(t, s, c)) for t, s in a.STATES for c in a.CONDITIONS)


def flags(family, rho, sigma):
    return f"--family {family}" + (f" --rho {rho:g}" if rho is not None else "") + (
        f" --sigma {sigma:g}" if sigma is not None else "")


SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 18; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {TASKS}                 # the wrapper checks run on each
N_ACTION_STEPS = 10
STATES = {[list(x) for x in a.STATES]}
CONDITIONS = {CONDITIONS!r}   # label: [family, rho, sigma]
FLAGS = {({c: flags(*v) for c, v in CONDITIONS.items()})!r}
BLOCKS = {BLOCKS!r}   # label: {{"task/state": [ids of each block]}}
HARNESS_LABEL, HARNESS_FLAGS = "{a.HARNESS_LABEL}", "--family ar_repeat --rho 1"
HARNESS_IDS = {HARNESS_IDS!r}   # one committed success and one failure per state
OUT_NAME = "step18"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 11.0                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"step 18 at {{REPO_REF}}: mechanism M1, {len(a.STATES)} states x {len(a.CONDITIONS)} conditions, {N_EPISODES} episodes "
      f"in {N_JOBS} jobs, {N_HARNESS} harness episodes; stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


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
# noise_reachability step 18: constancy or lower noise diversity? (mechanism M1; Kaggle, 2 x T4)

Step 18 of `noise_reachability/README.md`. No training. The base policy on {len(a.STATES)} resistant states with
constant noise (`repeat`), a slowly drifting noise (rho 0.9), half-temperature sampling (sigma 0.5) and the reported
processes: {N_EPISODES} episodes in {N_JOBS} jobs of at most {a.BLOCK}, after a {N_HARNESS}-episode harness.

**Needs no input on the first session. Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing
needs editing. The wrapper checks and the harness run first; a failure stops the run.

**This run needs two sessions (about 18 h of episodes; an estimate).** When step 12 says work remains, add this
version's output as input (**Add Input > Your Work**, only the latest step 18 version) and Save & Run All again. Step
6 must then print `RESUMING`.

Download from the Output tab: `step18_results.zip`.
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
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step18.py --root {OUT_ROOT}"


def n_ids(spec):
    return sum(int(p.split("-")[1]) - int(p.split("-")[0]) + 1 if "-" in p else 1 for p in spec.split(","))


def block_dir(task, state, label, b):
    return f"{OUT_ROOT}/task_{task}/{label}/state{state}_block{b}"


# this session's index: step 6 appends one line to repo_commit.txt per session (0 = the first)
SESSION = len(Path(OUT_ROOT, "repo_commit.txt").read_text().split()) - 1
print("session", SESSION, flush=True)


def done(path, spec):
    return len(read_csv(path)) >= n_ids(spec)


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

md("## 8. Wrapper checks on each task's real model (a few minutes); stop the run if they fail")
code(CHECK + '''
for t in TASKS:
    if not CHECK_DONE:
        break
    r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_noise_processes.py --task_id {t} --policy_path {POLICY_PATH} "
           f"--n_action_steps {N_ACTION_STEPS}", check=False, env=RUN_ENV)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode != 0:
        fail(f"check_noise_processes failed on task {t}")
        CHECK_DONE = False
''')

md(f"""
## 9. Harness, every session: two committed episodes per state, rerun through the new noise-process code; stops the run if it fails

Each state's two episodes must reproduce the committed rows exactly (success and number of steps). In the first session
a state that does not is dropped from every condition, and at least {a.MIN_HARNESS_STATES} of {len(a.STATES)} states must
reproduce. Every later session reruns the harness in its own folder before any episode, and must reproduce every kept
state again (and its package set must equal the first session's), or the run stops: the stack changed.
""")
code('''
HARNESS_DONE, KEPT = False, []
if CHECK_DONE and not FAILURE:
    jobs = {}
    for t, s in STATES:
        out = f"{OUT_ROOT}/harness/session{SESSION}/task{t}_state{s}"
        spec = HARNESS_IDS[f"{t}/{s}"]
        if not done(f"{out}/per_episode_{HARNESS_LABEL}_state{s}.csv", spec):
            jobs[f"harness_task{t}_state{s}"] = (f"{SEARCH} {HARNESS_FLAGS} --task_id {t} --init_state {s} "
                                                 f"--noise_ids {spec} --output_dir {out}", len(jobs) % 2)
    if run_stage("harness", jobs, len(STATES)):
        r = sh(f"{ANALYZE} --gate harness --out {OUT_ROOT}/step18_harness.json", check=False, env=RUN_ENV)
        print(r.stdout, r.stderr[-2000:])
        if r.returncode != 0:
            fail("the harness gate failed: investigate before spending more")
        else:
            KEPT = [tuple(k) for k in json.loads(Path(f"{OUT_ROOT}/step18_harness.json").read_text())["kept"]]
            HARNESS_DONE = True
            print("kept states:", KEPT, flush=True)
''')

md(f"""
## 10. The episodes: block by block ({a.BLOCK} ids, one process each), every kept state and condition, four at a time

Blocks run in order, so a session stopped at the deadline leaves the conditions with nearly equal counts. Block 0
holds every condition; blocks 2-{N_BLOCKS - 1} only `repeat`, rho 0.9 and sigma 0.5.
""")
code('''
EPISODES_DONE = False
if HARNESS_DONE and not FAILURE:
    jobs, total = {}, 0
    for b in range(''' + str(N_BLOCKS) + '''):
        for t, s in KEPT:
            for label in CONDITIONS:
                specs = BLOCKS[label][f"{t}/{s}"]
                if b >= len(specs):
                    continue
                total += 1
                out = block_dir(t, s, label, b)
                if not done(f"{out}/per_episode_{label}_state{s}.csv", specs[b]):
                    jobs[f"{label}_task{t}_state{s}_block{b}"] = (
                        f"{SEARCH} {FLAGS[label]} --task_id {t} --init_state {s} --noise_ids {specs[b]} --output_dir {out}",
                        len(jobs) % 2)
    EPISODES_DONE = run_stage("episodes", jobs, total)
''')

md("## 11. Report (only when every episode is in)")
code('''
if EPISODES_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step18_report.txt --table {OUT_ROOT}/step18_table.csv",
          check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
''')

md("## 12. Work left?")
code('''
if FAILURE:
    print("FAILED:", FAILURE)
elif EPISODES_DONE:
    print("STEP 18 DONE: download step18_results.zip")
else:
    print("WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("## 13. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step18_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step18_results.zip; du -sh {OUT_ROOT}")
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
