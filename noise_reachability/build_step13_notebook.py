"""Builds noise_reachability/colab_step13_unseen_states.ipynb (run from the repo root).

    python noise_reachability/build_step13_notebook.py <commit> [output path]

Step 13 of README.md on Colab T4: the base, N0-N2 and R0-R2 on initial states 30-49 of tasks 0, 6, 7 that no adapter
trained on. The code, environment, sanity and job cells come from step 8's Kaggle notebook unchanged (as the Colab T4
check's do), so the install is the one the Kaggle runs used; the settings, GPU/Drive, session, adapter, harness and job
cells are Colab's. States, seeds, policies and the adapters' sha256 are analyze_step13.py's constants. Every session
re-runs the harness first and stops if it fails. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step13 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "colab_step13_unseen_states.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_RUN = sum(len(a.STATES[t]) * len(a.SEEDS) * len(a.run_policies(t)) for t in a.TASKS)
N_HARNESS = len(a.TASKS) * len(a.HARNESS_SEEDS)
SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 13; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {list(a.TASKS)}
N_ACTION_STEPS = 10
STATES = {({t: ",".join(map(str, v)) for t, v in a.STATES.items()})}   # 30-49 minus trained states (7/33)
PARTS = {a.PARTS}                 # seeds 200-207 in two parts per policy and task
POLICIES = {a.POLICIES}
RUN_POLICIES = {({t: a.run_policies(t) for t in a.TASKS})}   # R7_s's weights are N7_s's: not run
HARNESS_STATE = {dict(a.HARNESS_STATE)}   # every session: N<t>_0 on the Colab T4 check's step-10 episodes
HARNESS_SEEDS = "{a.HARNESS_SEEDS[0]}-{a.HARNESS_SEEDS[-1]}"
ARCHIVE = "/content/drive/MyDrive/Alienware_Research_archives/efficient_vla"   # tools/archive_kaggle_output.sh folders
RUNS_IN = {{"N": f"{{ARCHIVE}}/noise-reachability-step-10/output/step10/runs",
           "R": f"{{ARCHIVE}}/expert-iteration-step-12/output/step12/runs"}}
ADAPTER_SHA256 = {dict(a.ADAPTER_SHA256)}
DRIVE_DIR = "/content/drive/MyDrive/efficient_vla_runs/step13"
OUT_ROOT = f"{{DRIVE_DIR}}/run"
STOP_AT_H = 20.0                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"step 13 at {{REPO_REF}}: tasks {{TASKS}}, {sum(len(v) for v in a.STATES.values())} states, seeds 200-207, policies "
      f"{{POLICIES}}, {N_RUN} evaluation episodes to run plus {N_HARNESS} harness episodes per session, "
      f"stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


''' + SETTINGS[end:]
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
JOBS = swap(JOBS, "to the zip in step 15", "to the zip in step 11")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md(f"""
# noise_reachability step 13: do the distilled adapters change success on initial states they never trained on? (Colab T4)

Step 13 of `noise_reachability/README.md`. The base, step 10's N<t>_0-2 and step 12's R<t>_0-2 on initial states 30-49 of
tasks 0, 6 and 7 that no adapter trained on, seeds 200-207, sampled noise ({N_RUN} episodes; R is not run on task 7,
whose R weights are N's).

**Inputs:** the archive on Google Drive, `MyDrive/Alienware_Research_archives/efficient_vla/` with
`noise-reachability-step-10/` and `expert-iteration-step-12/` (as `tools/archive_kaggle_output.sh` wrote them). Step 6
checks every adapter's sha256 against the archive manifests.

**Runtime: T4 GPU.** **Run all.** Nothing needs editing. Every session first re-runs 12 of step 10's Kaggle T4 episodes
(harness) and stops if fewer than 11 reproduce. Keep the tab open; about 45 h of runtime in all (estimate), so several
sessions: after a disconnect or the end of a session, **Run all** again. Finished episodes are kept on Drive and skipped.

When step 10 says every episode is done, download `step13_results.zip`.

At the end of every session, step 11 downloads `step13_results.zip` and step 12 syncs Drive and, after 90 s, releases
the runtime so it stops using compute units (set `RELEASE_RUNTIME = False` there to keep it).
""")

md("## 1. Settings")
code(SETTINGS)

md("## 2. GPU, RAM and Drive")
code('''
smi = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
gpus = [line.split(", ") for line in smi.stdout.strip().splitlines()]
assert gpus and all("T4" in g[0] for g in gpus), f"GPU {gpus}: set the runtime type to T4 GPU (step 13 runs on T4 only)"
CUDA_DRIVER = float(re.search(r"CUDA Version: ([0-9.]+)", sh("nvidia-smi").stdout).group(1))
avail_gb = int(re.search(r"MemAvailable:\\s+(\\d+)", open("/proc/meminfo").read()).group(1)) / 1e6
JOBS = max(1, min(4, int((avail_gb - 4) // GB_PER_JOB), os.cpu_count()))
print(f"{len(gpus)} GPU(s) {gpus}, driver supports CUDA {CUDA_DRIVER}; {os.cpu_count()} CPU cores; "
      f"{avail_gb:.0f} GB RAM available -> {JOBS} evaluation(s) at a time")

from google.colab import drive
drive.mount("/content/drive")
os.makedirs(f"{OUT_ROOT}/logs", exist_ok=True)
''')

md("## 3. Code")
code(CODE)

md("## 4. Environment (the package versions of the Kaggle runs; several minutes)")
code(ENVIRONMENT)

md("## 5. Sanity checks, tests and the base checkpoint")
code(SANITY)

md("## 6. This session, its commit, and the adapters (sha256 against the archive manifests)")
code('''
head = sh(f"git -C {REPO} log -1 --format='%H'").stdout.strip()
first = Path(OUT_ROOT, "repo_commit.txt")
earlier = first.read_text().split() if first.exists() else []
if earlier and earlier[0] != head:
    raise RuntimeError(f"step 13 started at commit {earlier[0]}, this session cloned {head}")
with open(first, "a") as f:
    f.write(head + "\\n")
k = len(earlier) + 1
while any(Path(OUT_ROOT, d, f"session{k}").exists() for d in ("harness", "harness_pending", "harness_failed")):
    k += 1                                   # a session whose repo_commit.txt line did not reach Drive
SESSION = f"session{k}"
Path(OUT_ROOT, f"environment_{SESSION}.json").write_text(json.dumps(
    {"gpus": gpus, "cuda_driver": CUDA_DRIVER, "torch_build": TORCH_BUILD, "ram_available_gb": round(avail_gb, 1),
     "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\\n")
print("this is", SESSION)


def adapter_dir(t, policy):
    return f"{RUNS_IN[policy[0]]}/{policy[0]}{t}_{policy[1:]}/adapter"


for name, files in ADAPTER_SHA256.items():
    folder = adapter_dir(int(name[1]), name[0] + name[3:])
    for fname, want in files.items():
        p = Path(folder, fname)
        assert p.exists(), f"{p} is missing: upload the archive folder to Drive"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == want, f"{p}: sha256 {got}, expected the archive's {want}"
    trained_with = json.loads(Path(folder, "run_config.json").read_text())["policy_path"]
    assert trained_with == POLICY_PATH, f"{name} was trained with {trained_with}, this session has {POLICY_PATH}"
print(f"{len(ADAPTER_SHA256)} adapters: sha256 equal to the archive's, trained with this checkpoint path")
''')

md("## 7. Helpers: jobs with resume and a deadline")
code(JOBS + '''
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"
ANALYZE = f"cd {REPO} && python3 -S noise_reachability/analyze_step13.py --root {OUT_ROOT}"


def sfbc(task):
    return f"cd {REPO} && {PY} -u {MAIN} --task_id {task} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}"


def suffix(policy):
    return "base" if policy == "base" else "lora"


def n_done(path):
    """Distinct episodes in a per-episode file (a row count would hide duplicates)."""
    return len({(r["seed"], r["init_state"]) for r in read_csv(path)})


def stop_leftovers():
    """Jobs run in their own process groups, so an interrupted cell can leave them writing to the same files."""
    sh("pkill -f 'self_filtered_bc/self_filtered_bc[.]py'; sleep 5", check=False)


def eval_out(policy, t, j):
    return f"{OUT_ROOT}/eval/{policy}/task_{t}/part{j}"


def eval_cmd(policy, t, j):
    which = "--eval_policies base" if policy == "base" else f"--eval_policies lora --lora_dir {adapter_dir(t, policy)}"
    return (f"{sfbc(t)} --stage eval {which} --eval_seeds {PARTS[j]} --eval_init_states {STATES[t]} "
            f"--eval_output_dir {eval_out(policy, t, j)} --max_episodes_per_process 10")


def part_size(t, j):
    return n_range(PARTS[j]) * len(STATES[t].split(","))


def harness_cmd(t):
    return (f"{sfbc(t)} --stage eval --eval_policies lora --lora_dir {adapter_dir(t, 'N0')} --eval_seeds {HARNESS_SEEDS} "
            f"--eval_init_states {HARNESS_STATE[t]} --eval_output_dir {OUT_ROOT}/harness_pending/{SESSION}/task_{t} "
            f"--max_episodes_per_process 10")
''')

md("## 8. Harness of this session: N<t>_0 on 12 of step 10's Kaggle T4 episodes; stops the session if it fails")
code('''
stop_leftovers()
PENDING = f"{OUT_ROOT}/harness_pending/{SESSION}"
jobs = {f"harness_{SESSION}_task{t}": (harness_cmd(t), 0) for t in TASKS
        if n_done(f"{PENDING}/task_{t}/per_episode_lora.csv") < n_range(HARNESS_SEEDS)}
codes = run_jobs(jobs, max_parallel=JOBS)
HARNESS_OK = False
if failed(codes) or any(c is None for c in codes.values()):
    fail(f"harness jobs failed: {codes}")      # left in harness_pending/, which the analysis does not read
elif (gate_rc := sh(f"{ANALYZE} --gate harness_pending/{SESSION}", check=False).returncode) == 1:
    os.makedirs(f"{OUT_ROOT}/harness_failed", exist_ok=True)
    os.rename(PENDING, f"{OUT_ROOT}/harness_failed/{SESSION}")
    fail("the harness failed: this session does not reproduce step 10 or has other package versions; no episodes added")
elif gate_rc != 0:
    fail("the harness could not be read (left in harness_pending/): see the message above")
else:
    os.makedirs(f"{OUT_ROOT}/harness", exist_ok=True)
    os.rename(PENDING, f"{OUT_ROOT}/harness/{SESSION}")
    HARNESS_OK = True
''')

md(f"## 9. Evaluation: {N_RUN} episodes, one part (4 seeds x one task's states) per job, in an order that keeps policies even")
code('''
def eval_done(policy, t):
    return sum(n_done(f"{eval_out(policy, t, j)}/per_episode_{suffix(policy)}.csv") for j in range(len(PARTS)))


if HARNESS_OK and not FAILURE:
    stop_leftovers()
    jobs = {}
    for j in range(len(PARTS)):
        for t in TASKS:
            for policy in RUN_POLICIES[t]:
                if n_done(f"{eval_out(policy, t, j)}/per_episode_{suffix(policy)}.csv") < part_size(t, j):
                    jobs[f"eval_{policy}_task{t}_part{j}"] = (eval_cmd(policy, t, j), 0)
    total = sum(len(RUN_POLICIES[t]) * len(PARTS) for t in TASKS)
    print(f"evaluation: {len(jobs)} of {total} parts to run; {hours_left():.1f} h left", flush=True)
    codes = run_jobs(jobs, max_parallel=JOBS)
    if failed(codes):
        fail(f"evaluation failed: {failed(codes)}; see {OUT_ROOT}/logs")
done = {t: {p: eval_done(p, t) for p in RUN_POLICIES[t]} for t in TASKS}
print("episodes done:", done)
ALL_DONE = all(done[t][p] == sum(part_size(t, j) for j in range(len(PARTS))) for t in TASKS for p in RUN_POLICIES[t])
''')

md("## 10. Analysis (when every episode is done)")
code('''
if ALL_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step13_report.txt", check=False).returncode != 0:
        fail("the analysis failed: the episodes are on Drive and in the zip")
elif not FAILURE:
    print("WORK REMAINS: Run all again in a new session (finished episodes are kept on Drive)")
''')

md("## 11. Download")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /content/step13_results.zip . -x '*.pyc' -x '*.tmp*'")
from google.colab import files
files.download("/content/step13_results.zip")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
''')

md("## 12. Release the runtime (stops the compute-unit charge; the results are on Drive and in the zip)")
code('''
RELEASE_RUNTIME = True             # False keeps the GPU assigned after the session's work
if FAILURE:
    print("THIS SESSION FAILED:", FAILURE)
drive.flush_and_unmount()          # every episode file written to Drive before the machine goes
print("Drive synced and unmounted")
if RELEASE_RUNTIME:
    print("releasing the runtime in 90 s (time for the browser to save step13_results.zip)", flush=True)
    time.sleep(90)
    from google.colab import runtime
    runtime.unassign()
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, "at", REF)
