"""Builds noise_reachability/kaggle_step14_dilution.ipynb (run from the repo root).

    python noise_reachability/build_step14_notebook.py <commit> [output path]

Step 14 of README.md: is round 2's loss on 0/8 real, and is it dilution? The settings, hardware, code, environment,
sanity, resume and job cells come from step 8's notebook, as step 12's builder takes them, so step 14 installs and runs
as steps 8-12 did. The episodes, folders and settings are analyze_step14.py's constants, written into the settings cell
here, so they are fixed in one place. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_step14 as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step14_dilution.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_HARNESS = sum(len(a.seeds_of(seeds)) for seeds, _, _ in a.HARNESS.values())
N_CHECK = len(a.CHECK_SEEDS) * len(a.CHECK_POLICIES)
N_SEARCHED = len(a.SEARCHED_SEEDS) * len(a.SEARCHED) * len(a.SEARCHED_POLICIES)

SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The protocol of README.md step 14; nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASK = {a.TASK}
N_ACTION_STEPS = 10
CHECK_STATE = {a.CHECK_STATE}                    # round 1's best trained state, where round 2 lost ground
SEARCHED = {list(a.SEARCHED)}             # step 12's searched task-0 states
TRAIN_SEEDS = {list(a.TRAIN_SEEDS)}
TRAIN_STEPS, BATCH_SIZE = {a.L_STEPS}, 16       # 2,000 x 7,309 / 5,473 frames: round 1's samples per frame on R's set
HARNESS = {({p: v[0] for p, v in a.HARNESS.items()})}   # policy: seeds of its committed 0/8 episodes
CHECK_PARTS = {a.CHECK_PARTS}
SEARCHED_PARTS = {a.SEARCHED_PARTS}
CHECK_POLICIES = {a.CHECK_POLICIES}
SEARCHED_POLICIES = {a.SEARCHED_POLICIES}
OUT_NAME = "step14"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
ROUND2_DATA = "/tmp/step14"        # R's recordings (step 10's and step 12's), linked per session (not saved as output)
STOP_AT_H = 11.0                   # no new work after this many hours of the session; running jobs are stopped
TRAIN_NEEDS_H = 5.0                # before the timing check: no training run starts with less time left
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''ADAPTERS = [f"L{{TASK}}_{{s}}" for s in TRAIN_SEEDS]
print(f"step 14 at {{REPO_REF}}: task {{TASK}}, check state {{CHECK_STATE}}, searched {{SEARCHED}}, adapters {{ADAPTERS}} "
      f"({{TRAIN_STEPS}} steps), at most {N_HARNESS + N_CHECK + N_SEARCHED} episodes ({N_HARNESS} harness, {N_CHECK} on "
      f"0/{a.CHECK_STATE}, {N_SEARCHED} on the searched states), stop after {{STOP_AT_H}} h, output {{OUT_ROOT}}")


''' + SETTINGS[end:]
HARDWARE = step8_cell("JOBS = max(1")
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
RESUME = step8_cell("repo_commit.txt")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# noise_reachability step 14: is round 2's loss on 0/8 real, and is it dilution? (Kaggle, 2 x T4)

Step 14 of `noise_reachability/README.md`. Task 0 only. Round 1's adapters N0_s (step 10), round 2's R0_s (step 12) and
L0_s (R0_s's training set, trained for 2,671 steps instead of 2,000) are evaluated with sampled noise on fresh episodes
of 0/8 (all three) and of step 12's searched task-0 states (R and L).

**Needs two inputs:** Add Input > Your Work > the step 10 notebook, its **last** version (adapters N, round 1's
recordings), and the step 12 notebook (`expert-iteration-step-12`), its **last** version (adapters R, the search's
recordings). **Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The harness runs
first; a failure stops the run.

**This run needs about two sessions.** When step 14 says work remains, add this version's output as input as well
(**Add Input > Your Work**, only the latest step 14 version; keep step 10's and step 12's), and Save & Run All again.
Step 6 must then print `RESUMING`.

Download from the Output tab: `step14_results.zip` (results without recordings or adapter weights, which stay in the
version's output).
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

md("## 7. Inputs: step 10's output (N, round 1's recordings) and step 12's (R, the search's recordings)")
code('''
def find_input(name, marker):
    found = [p for p in sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) if os.path.exists(f"{p}/{marker}")]
    assert len(found) == 1, f"attach {name}'s last version as input (found {found})"
    return found[0]


STEP10_IN = find_input("step10", f"runs/N{TASK}_0/adapter/run_config.json")
STEP12_IN = find_input("step12", f"runs/R{TASK}_0/adapter/run_config.json")
for name, path in (("step 10", STEP10_IN), ("step 12", STEP12_IN)):
    print(name, "output:", path, "| its commits:", open(f"{path}/repo_commit.txt").read().split())
for s in TRAIN_SEEDS:
    assert os.path.exists(f"{STEP10_IN}/runs/N{TASK}_{s}/adapter/run_config.json"), f"step 10 has no adapter N{TASK}_{s}"
    assert os.path.exists(f"{STEP12_IN}/runs/R{TASK}_{s}/adapter/run_config.json"), f"step 12 has no adapter R{TASK}_{s}"


def n_adapter(s):
    return f"{STEP10_IN}/runs/N{TASK}_{s}/adapter"


def r_adapter(s):
    return f"{STEP12_IN}/runs/R{TASK}_{s}/adapter"
''')

md("## 8. Helpers: parallel jobs with resume and a deadline")
code(JOBS + '''
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step14.py --root {OUT_ROOT}"
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"


def sfbc():
    return f"cd {REPO} && {PY} -u {MAIN} --task_id {TASK} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}"


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


def recordings(folder):
    return [f for f in glob.glob(f"{folder}/episodes/*.npz") if not f.endswith(".tmp.npz")]
''')

md("""
## 9. Harness: N0_0 and R0_0 on four of their own committed episodes of 0/8 each; stops the run if it fails

At least 7 of the 8 episodes must reproduce the committed rows exactly (success and number of steps).
""")
code('''
HARNESS_DONE = False
jobs = {}
for k, (policy, seeds) in enumerate(HARNESS.items()):
    out = f"{OUT_ROOT}/harness/{policy}"
    if len(read_csv(f"{out}/per_episode_lora.csv")) < n_range(seeds):
        lora = n_adapter(policy[1:]) if policy.startswith("N") else r_adapter(policy[1:])
        jobs[f"harness_{policy}"] = (f"{sfbc()} --stage eval --eval_policies lora --lora_dir {lora} --eval_seeds {seeds} "
                                     f"--eval_init_states {CHECK_STATE} --eval_output_dir {out} "
                                     f"--max_episodes_per_process 10", k % 2)
if run_stage("harness", jobs, len(HARNESS)):
    HARNESS_DONE = run_gate("harness")
''')

md("## 10. Evaluation helpers: 0/8 (N, R, L) and the searched states (R, L), sampled noise, fresh seeds")
code('''
SETS = {"eval_check": (str(CHECK_STATE), CHECK_PARTS, CHECK_POLICIES),
        "eval_searched": (",".join(map(str, SEARCHED)), SEARCHED_PARTS, SEARCHED_POLICIES)}


def eval_out(which, policy, j):
    return f"{OUT_ROOT}/{which}/{policy}/part{j}"


def lora_of(policy):
    s = policy[1:]
    return {"N": n_adapter(s), "R": r_adapter(s), "L": f"{OUT_ROOT}/runs/L{TASK}_{s}/adapter"}[policy[0]]


def eval_cmd(which, policy, j):
    states, parts, _ = SETS[which]
    return (f"{sfbc()} --stage eval --eval_policies lora --lora_dir {lora_of(policy)} --eval_seeds {parts[j]} "
            f"--eval_init_states {states} --eval_output_dir {eval_out(which, policy, j)} --max_episodes_per_process 10")


def part_size(which, j):
    states, parts, _ = SETS[which]
    return n_range(parts[j]) * len(states.split(","))


def eval_rows(which, policy):
    return [r for j in range(len(SETS[which][1])) for r in read_csv(f"{eval_out(which, policy, j)}/per_episode_lora.csv")]


def evaluate(letters):
    """Every unfinished part of the policies whose first letter is in ``letters`` and whose adapter is ready."""
    jobs, k = {}, 0
    for which, (_, parts, policies) in SETS.items():
        for policy in policies:
            if policy[0] not in letters or (policy[0] == "L" and not adapter_done(f"L{TASK}_{policy[1:]}")):
                continue
            for j in range(len(parts)):
                if len(read_csv(f"{eval_out(which, policy, j)}/per_episode_lora.csv")) < part_size(which, j):
                    jobs[f"{which}_{policy}_part{j}"] = (eval_cmd(which, policy, j), k % 2)
                    k += 1
    run_stage(f"evaluation of {letters}", jobs, len(jobs))
''')

md("## 11. Evaluation of N0-N2 and R0-R2 (their adapters exist already)")
code('''
if HARNESS_DONE and not FAILURE:
    evaluate("NR")
''')

md("""
## 12. L's training data: exactly R0_s's recordings (step 10's task-0 recordings and step 12's task-0 search successes)

Linked (not copied) under `/tmp/step14/task_0/round2_data/episodes/` in every session; training keeps the successes. The
analysis checks that each L0_s trained on exactly R0_s's episodes.
""")
code('''
def round2_data():
    return f"{ROUND2_DATA}/task_{TASK}/round2_data"


if HARNESS_DONE and not FAILURE:
    dest = f"{round2_data()}/episodes"
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    old, new = recordings(f"{STEP10_IN}/task_{TASK}/replay_data"), recordings(f"{STEP12_IN}/task_{TASK}/search_data")
    for f in old + new:
        target = f"{dest}/{os.path.basename(f)}"
        assert not os.path.exists(target), f"two recordings named {os.path.basename(f)}"
        os.symlink(f, target)
    print(f"task {TASK}: {len(old)} round-1 and {len(new)} round-2 recordings (successes and failures)")
    # before about 12 GPU-hours of training: the selection must give exactly R0_s's committed training episodes
    if sh(f"cd {REPO} && {PY} noise_reachability/check_step14_data.py {round2_data()}", check=False,
          env=RUN_ENV).returncode != 0:
        fail("L's training data is not R's: check the step 10 and step 12 inputs (last versions?)")
''')

md("""
## 13. Training: L0_s from the base for 2,671 steps, training seeds 0-2

Two adapters at a time, one per T4. A run starts only if the session can finish it (1.25 x the measured time plus 10
minutes); an unfinished run restarts from scratch in the next session.
""")
code('''
TIMING = Path(OUT_ROOT, "train_timing.json")
if HARNESS_DONE and not FAILURE and not TIMING.exists() and hours_left() > 0.5:
    r = sh(f"{sfbc()} --stage train --seed 0 --selection success --data_dir {round2_data()} --train_steps 20 "
           f"--batch_size {BATCH_SIZE} --log_every 5 --val_every 1000 --output_dir /tmp/step14_train_timing",
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
    s = name.split("_")[1]
    return (f"{sfbc()} --stage train --selection success --train_steps {TRAIN_STEPS} --batch_size {BATCH_SIZE} "
            f"--seed {s} --data_dir {round2_data()} --output_dir {OUT_ROOT}/runs/{name}")


ready = [n for n in ADAPTERS if not adapter_done(n)] if HARNESS_DONE and not FAILURE else []
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

md("## 14. Evaluation of every L whose adapter is ready; then what is left")
code('''
if HARNESS_DONE and not FAILURE:
    evaluate("L")
for which, (_, parts, policies) in SETS.items():
    print(which, {p: len(eval_rows(which, p)) for p in policies}, f"of {sum(part_size(which, j) for j in range(len(parts)))} each")
ALL_DONE = (len(DONE_ADAPTERS) == len(ADAPTERS) and all(
    len(eval_rows(w, p)) == sum(part_size(w, j) for j in range(len(SETS[w][1]))) for w in SETS for p in SETS[w][2]))
if ALL_DONE and not FAILURE:
    if sh(f"{ANALYZE} --out {OUT_ROOT}/step14_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
if not ALL_DONE and not FAILURE:
    print("WORK REMAINS: add this version's output as input (keep step 10's and step 12's) and Save & Run All again")
''')

md("## 15. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step14_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step14_results.zip; du -sh {OUT_ROOT}")
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
