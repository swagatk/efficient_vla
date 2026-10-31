"""Builds noise_reachability/colab_step5b_noise_awr.ipynb (run from the repo root).

The GPU, environment and sanity cells are taken from colab_step3_noise_sac.ipynb, so the notebooks install the same
stack. Edit this builder, not the notebook. Set REPO_REF to the reviewed commit before use.
"""
import json
import sys
from pathlib import Path

REPO_REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
REPO = Path.cwd()
base = json.load(open(REPO / "noise_reachability/colab_step3_noise_sac.ipynb"))
src = lambda i: "".join(base["cells"][i]["source"])  # noqa: E731
GPU, ENVIRONMENT, SANITY = src(4), src(8), src(10)
assert "drive.mount" in GPU and "uv venv" in ENVIRONMENT and "pytest tests/" in SANITY

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# noise_reachability step 5b: a learned noise actor on held-out states (Colab, A100)

Step 5b of `noise_reachability/README.md`. NSL-AWR on task 0: trained on init states 10-49, evaluated on 0-9, which
hold all five systematic states. The VLA's weights do not change. Nothing here is a setting to change.

**Before you start.** Runtime > Change runtime type > **A100 GPU** (High-RAM if offered).

**How to run.** Run the cells in order. Step 5 checks the wrapper on the real model. Step 6 times one training epoch
and fixes the number of epochs. Step 7 starts the run in the background, and step 8 keeps a cell busy until it ends,
so Colab does not treat the runtime as idle. Keep the tab open and the computer awake.

**If the runtime is lost:** finished work is on Google Drive. Run steps 1 to 7 again: the pilot is skipped once
`pilot.json` exists, and finished epochs and episodes are skipped. Step 7 refuses to resume on a different commit.

**Time.** Estimated at about 16 hours; step 6 prints the estimate from its measurement. The run stops by itself if
the harness or the gate after training seed 0 fails.
""")

md("## 1. Settings")
code(f'''
import csv, glob, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "{REPO_REF}"
assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"
REPO = "/content/efficient_vla"
VENV = "/content/venv"
PY = f"{{VENV}}/bin/python"
DRIVE_DIR = "/content/drive/MyDrive/efficient_vla_runs/step5b"
OUT_ROOT = f"{{DRIVE_DIR}}/run"
MAX_EPOCHS = 37
MAX_HOURS = 17
SEEDS_TRAINED = 3
EVAL_EPISODES = 160 + 160 + 480 + 30 + 10
POLICY_REPO, POLICY_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"

# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
print("step 5b at", REPO_REF, "output", OUT_ROOT)


def sh(cmd, check=True, env=None):
    # Capture and print: notebooks do not show output that a subprocess writes directly.
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {{r.returncode}}: {{cmd}}")
    return r
''')

md("## 2. GPU, RAM and Drive")
code(GPU)

md("## 3. Code")
code('''
if not Path(REPO, ".git").exists():
    sh(f"git clone {REPO_URL} {REPO}")
sh(f"git -C {REPO} fetch origin && git -C {REPO} checkout {REPO_REF}")
sh(f"git -C {REPO} log -1 --format='%H %s'")
''')

md("## 4. Environment\n\nThe same cell as the step-3 notebook. Takes several minutes.")
code(ENVIRONMENT)

md("## 4b. Sanity checks and tests")
code(SANITY)

md("## 5. Wrapper check on the real model (about 3 minutes)")
code('''
r = sh(f"""{PY} -c "from huggingface_hub import snapshot_download; print(snapshot_download('{POLICY_REPO}', revision='{POLICY_REVISION}'))" """, env=RUN_ENV)
POLICY_PATH = r.stdout.strip().splitlines()[-1]
RUN_ENV["POLICY_PATH"] = POLICY_PATH
r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_step5b.py --policy_path {POLICY_PATH}", check=False, env=RUN_ENV)
assert r.returncode == 0 and "[check] all passed" in r.stdout, "a check failed; see the output above"
''')

md("""
## 6. Pilot: one training epoch of seed 9 (about 15 minutes, once)

Not a result. It fixes the number of epochs: 37 if the estimate fits in 17 hours, otherwise the largest number that does.
""")
code('''
assert POLICY_REVISION in RUN_ENV.get("POLICY_PATH", ""), "run step 5 first: the policy snapshot is not set"
PILOT = Path(OUT_ROOT, "pilot.json")
if not PILOT.exists():
    shutil.rmtree(f"{OUT_ROOT}/pilot", ignore_errors=True)   # a pilot cut short starts again
    sh(f"bash {REPO}/noise_reachability/run_step5b_noise_awr.sh", env=dict(RUN_ENV, STAGE="pilot", OUT_ROOT=f"{OUT_ROOT}/pilot"))
    epoch_h = int(Path(f"{OUT_ROOT}/pilot/pilot_epoch_seconds.txt").read_text()) / 3600
    eval_h = EVAL_EPISODES * epoch_h / 40
    epochs = min(MAX_EPOCHS, math.floor((MAX_HOURS - eval_h) / (SEEDS_TRAINED * epoch_h)))
    assert epochs >= 10, f"only {epochs} epochs fit: stop and revisit the design"
    PILOT.write_text(json.dumps({"epoch_hours": epoch_h, "eval_hours": eval_h, "epochs": epochs}, indent=1))
pilot = json.loads(PILOT.read_text())
EPOCHS = pilot["epochs"]
print(f"one epoch took {pilot['epoch_hours'] * 60:.1f} min; {EPOCHS} epochs, estimated "
      f"{SEEDS_TRAINED * EPOCHS * pilot['epoch_hours'] + pilot['eval_hours']:.1f} h in all")
''')

md("## 7. Launch (background)\n\nRunning this cell while a run is active does nothing.")
code('''
assert POLICY_REVISION in RUN_ENV.get("POLICY_PATH", ""), "run step 5 first: the policy snapshot is not set"
running = subprocess.run(["pgrep", "-f", "run_step5b_noise_aw[r]"], capture_output=True, text=True).stdout.split()
head = sh(f"git -C {REPO} log -1 --format='%H'").stdout.strip()
first = Path(OUT_ROOT, "repo_commit.txt")
started_with = first.read_text().split()[0] if first.exists() and first.read_text().strip() else None
if running:
    print("A run is already active. Not starting another.")
elif started_with and started_with != head:
    raise RuntimeError(f"this run started at commit {started_with}, the checkout is at {head}")
else:
    run = f"bash {REPO}/noise_reachability/run_step5b_noise_awr.sh"
    analyse = (f"{PY} -I -S {REPO}/noise_reachability/analyze_step5b.py --root {OUT_ROOT} --epochs {EPOCHS} "
               f"--out {OUT_ROOT}/step5b_report.txt")
    cmd = f"STAGE=harness {run} && STAGE=prior {run} && STAGE=train {run} && STAGE=eval {run} && {analyse}"
    Path(OUT_ROOT, "step5b_report.txt").unlink(missing_ok=True)
    subprocess.Popen(f"nohup bash -c \\"{cmd}; echo runner exit \\\\$?\\" >> '{OUT_ROOT}/runner.log' 2>&1 &", shell=True,
                     executable="/bin/bash", env=dict(RUN_ENV, OUT_ROOT=OUT_ROOT, EPOCHS=str(EPOCHS)),
                     start_new_session=True)
    with open(first, "a") as f:
        f.write(head + "\\n")
    print("Started; logs in", f"{OUT_ROOT}/logs")
''')

md("## 8. Progress; waits until the run ends")
code('''
def count():
    try:
        done = len(glob.glob(f"{OUT_ROOT}/train/seed*/actor_e*.pt"))
        episodes = sum(sum(1 for _ in open(f)) - 1 for f in glob.glob(f"{OUT_ROOT}/eval/**/per_episode_*.csv", recursive=True))
    except OSError as e:     # Drive hiccup: this cell must outlive it, or the next cell never runs
        print("progress not read:", e, flush=True)
        return -1, -1
    return done, episodes


while subprocess.run(["pgrep", "-f", "run_step5b_noise_aw[r]"], capture_output=True).stdout:
    e, n = count()
    print(time.strftime("%H:%M"), f"{e}/{SEEDS_TRAINED * EPOCHS} epochs, {n}/{EVAL_EPISODES} evaluation episodes", flush=True)
    time.sleep(600)
print(count(), "no runner active")
sh(f"tail -n 3 '{OUT_ROOT}/runner.log'", check=False)
sh(f"cat '{OUT_ROOT}/step5b_report.txt'", check=False)
''')

md("""
## 9. Save the results to Drive and release the GPU

Queue this cell before you leave: it runs when step 8 ends, so the GPU is not left idle. Replay files and actors stay
in the Drive run folder and out of the zip.
""")
code('''
from google.colab import drive, runtime
sh(f"cd '{OUT_ROOT}' && zip -q -r '{DRIVE_DIR}/step5b_results.zip' . -x '*.pyc' -x '*.tmp*' -x '*.npz' -x '*.pt'", check=False)
sh(f"tail -n 3 '{OUT_ROOT}/runner.log'", check=False)
drive.flush_and_unmount()
runtime.unassign()
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("noise_reachability/colab_step5b_noise_awr.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
