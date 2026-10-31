"""Builds noise_reachability/kaggle_step6_replication.ipynb (run from the repo root).

    python noise_reachability/build_step6_notebook.py <commit> [output path]

The hardware, code and environment cells come from the baseline notebook and the sanity, resume, arm-B and job cells
from step 4's notebook, so step 6 installs and evaluates as step 4 did. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step6_replication.ipynb"
base = json.load(open(REPO / "baseline_performance/kaggle_baseline_lerobot_eval.ipynb"))
step4 = json.load(open(REPO / "noise_reachability/kaggle_step4_weight_update.ipynb"))
src = lambda nb, i: "".join(nb["cells"][i]["source"])  # noqa: E731
HARDWARE, CODE, ENVIRONMENT = src(base, 4), src(base, 6), src(base, 8)
assert "JOBS = max(1" in HARDWARE and "git clone" in CODE and "uv venv" in ENVIRONMENT


def step4_cell(marker):
    found = [c for c in step4["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


SANITY = swap(step4_cell('RUN_ENV = dict(os.environ'),
              'BASE_ARGS = f"--task_id {TASK} --n_action_steps {N_ACTION_STEPS}"\n', "")
RESUME = swap(step4_cell("repo_commit.txt"), '("logs", "demos", "data", "runs", "eval", "harness_check")', '("logs", "eval")')
ARM_B = step4_cell("found by its training job") + '''
assert B_INFO["model_sha256"] == B_SHA256, "this is not the checkpoint step 4 evaluated"
'''
JOBS = step4_cell("def run_jobs(jobs")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 6: do the labels and arm B's uneven gain hold beyond task 0? (Kaggle, 2 x T4)

Step 6 of `noise_reachability/README.md` (design fixed 2026-10-31). The base `lerobot/smolvla_libero` at `a24f1f3804`
and step 4's arm B checkpoint, `n_action_steps=10`, on tasks 4, 6, 7, 8 x init states 0-9 x seeds 16-31: 2 policies x
640 episodes. No training.

**Before the first version:** add the continued-training checkpoint as input (**Add Input > Your Datasets**), as in step
4. Step 7 finds it by the job name in its `train_config.json` and checks it is step 4's.

**Run with Save & Run All (Accelerator: GPU T4 x2, Internet on).** Nothing needs editing. The base runs first. Step 9
stops the run if a task's base differs from seeds 0-15 by more than 18 of 160. Then B, then the analysis. If anything
stops the run, step 12 still writes the zip, then reports the failure as an error.

**If step 10 says work remains**, add this version's output as input (**Add Input > Your Work**, keep the checkpoint
dataset) and Save & Run All again: finished episodes are skipped.

Download from the Output tab: `step6_results.zip`.
""")

md("## 1. Settings")
code(f'''
import csv, glob, hashlib, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

T_SESSION = time.time()
REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "{REF}"
assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"
REPO = "/tmp/efficient_vla"        # code and venv live outside /kaggle/working, so they are not saved as output
VENV = "/tmp/venv"
PY = f"{{VENV}}/bin/python"

# The protocol of README.md step 6; nothing here is meant to be edited.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
B_JOB_NAME = "smolvla_libero10_continued"     # the dataset-input folder whose train_config.json names this job
B_SHA256 = "b75872e158799eb4c824d2ac8d1f6e956bb0cc243d9033d302fa8e1868d6b336"  # step 4's arm_b_checkpoint.json
TASKS = [4, 6, 7, 8]
N_ACTION_STEPS = 10
EVAL_SEED_PARTS = ["16-19", "20-23", "24-27", "28-31"]  # 4 x 4 seeds = 16 per init state
EVAL_INIT_STATES = "0-9"
PART_EPISODES = 40
OUT_NAME = "step6"
OUT_ROOT = f"/kaggle/working/{{OUT_NAME}}"
STOP_AT_H = 10.5                   # no new work after this many hours of the session; running jobs are stopped
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
print(f"step 6 at {{REPO_REF}}: tasks {{TASKS}}, n_action_steps {{N_ACTION_STEPS}}, policies base and B, seeds 16-31 x init "
      f"states {{EVAL_INIT_STATES}}, {{2 * len(TASKS) * len(EVAL_SEED_PARTS)}} jobs, {{2 * len(TASKS) * 160}} episodes, output {{OUT_ROOT}}")


def sh(cmd, check=True, env=None):
    # Capture and print: notebooks do not show output that a subprocess writes directly.
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {{r.returncode}}: {{cmd}}")
    return r


def hours_left():
    return STOP_AT_H - (time.time() - T_SESSION) / 3600
''')

md("## 2. Hardware")
code(HARDWARE + '''
assert len(gpus) == 2, "Set Accelerator to GPU T4 x2"
assert JOBS >= 4, f"RAM for {JOBS} SmolVLA processes; the notebook runs 4 at once"
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

md("## 7. Arm B's checkpoint (Kaggle dataset input)")
code(ARM_B)

md("## 8. Helpers: parallel jobs with resume and a deadline")
code(JOBS + '''


FAILURE = None


def fail(message):
    """Keep going to the zip in step 12, which raises this."""
    global FAILURE
    FAILURE = message
    print("FAILED:", message, flush=True)


LABEL_FILES = (f"{REPO}/week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv "
               f"{REPO}/week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv")
ANALYZE = f"cd {REPO} && {PY} -S noise_reachability/analyze_step6.py --labels {LABEL_FILES}"


def part_dir(policy, task, j):
    return f"{OUT_ROOT}/eval/{policy}/task_{task}/part{j}"


def rows_of(policy):
    return [r for t in TASKS for j in range(len(EVAL_SEED_PARTS))
            for r in read_csv(f"{part_dir(policy, t, j)}/per_episode_base.csv")]


def evaluate(policy, path):
    """Both policies load with --eval_policies base, as step 4 ran arm B."""
    jobs = {}
    for t in TASKS:
        for j, part in enumerate(EVAL_SEED_PARTS):
            out = part_dir(policy, t, j)
            if len(read_csv(f"{out}/per_episode_base.csv")) < PART_EPISODES:
                jobs[f"eval_{policy}_task{t}_part{j}"] = (
                    f"cd {REPO} && {PY} -u {MAIN} --stage eval --task_id {t} --n_action_steps {N_ACTION_STEPS} "
                    f"--policy_path {path} --eval_policies base --eval_seeds {part} "
                    f"--eval_init_states {EVAL_INIT_STATES} --eval_output_dir {out} --max_episodes_per_process 10",
                    len(jobs) % 2)
    print(f"{policy}: {len(jobs)} jobs to run; {hours_left():.1f} h left")
    if jobs and hours_left() > 0:
        codes = run_jobs(jobs, max_parallel=4)
        if failed(codes):
            fail(f"evaluation failed: {failed(codes)}")
            return False
    rows = rows_of(policy)
    print(f"{policy}: {len(rows)}/{160 * len(TASKS)} episodes")
    if len(rows) == 160 * len(TASKS):
        write_csv(rows, f"{OUT_ROOT}/eval/per_episode_{policy}.csv")
        return True
    return False
''')

md("""
## 9. The base, then the harness

Stops the run if a task's base differs from its seeds 0-15 count by more than 18 of 160 (README, step 6).
""")
code('''
BASE_DONE = evaluate("base", POLICY_PATH)
if BASE_DONE and sh(f"{ANALYZE} --base {OUT_ROOT}/eval/per_episode_base.csv --harness_only", check=False,
                    env=RUN_ENV).returncode != 0:
    fail("the harness failed: investigate before running B")
    BASE_DONE = False
''')

md("## 10. Arm B")
code('''
B_DONE = evaluate("B", B_PATH) if BASE_DONE else False
if not FAILURE:
    print("all episodes done" if B_DONE else "WORK REMAINS: add this version's output as input and Save & Run All again")
''')

md("""
## 11. Analysis (when every episode is done)
""")
code('''
if B_DONE:
    STEP4 = f"{REPO}/week6/noise_reachability_step4_kaggle/run/eval"
    if sh(f"{ANALYZE} --base {OUT_ROOT}/eval/per_episode_base.csv --b {OUT_ROOT}/eval/per_episode_B.csv "
          f"--task0_base {STEP4}/per_episode_base.csv --task0_b {STEP4}/per_episode_B.csv "
          f"--out {OUT_ROOT}/step6_report.txt", check=False, env=RUN_ENV).returncode != 0:
        fail("the analysis failed: the episodes are in the zip")
else:
    print("evaluation not finished; analysis skipped")
''')

md("## 12. Files for the repo")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /kaggle/working/step6_results.zip . -x '*.npz' '*.safetensors' '*.pt'")
sh(f"du -h /kaggle/working/step6_results.zip; du -sh {OUT_ROOT}")
print(f"session time used: {(time.time() - T_SESSION) / 3600:.1f} h")
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": base["metadata"], "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, len(cells), "cells")
