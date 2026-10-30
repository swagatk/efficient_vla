"""Builds noise_reachability/colab_step5a_deep_noise.ipynb (run from the repo root).

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
# noise_reachability step 5a: throughput pilot and deep noise search (Colab, A100)

Step 5a of `noise_reachability/README.md`. The frozen `lerobot/smolvla_libero` (`n_action_steps=10`) on task 0: 70
noises in each of three families (sampled, a fixed full noise, a fixed repeated vector) on the five systematic init
states, 35 on the control state 9, then jittered copies of what succeeded with a fixed noise. No weights change. Nothing here
is a setting to change.

**Before you start.** Runtime > Change runtime type > **A100 GPU** (High-RAM if offered).

**How to run.** Run the cells in order. Step 5 checks the wrapper on the real model. Step 6 measures how many
processes the A100 serves and picks that number. Step 7 starts the run in the background, and step 8 keeps a cell
busy until it ends, so Colab does not treat the runtime as idle. Keep the tab open and the computer awake.

**If the runtime is lost:** every episode is saved to Google Drive. Run steps 1 to 7 again: the pilot is skipped once
`pilot.json` exists, and finished episodes are skipped. Step 7 refuses to resume on a different commit.

**Time.** The pilot is estimated at about 30 minutes, then at most 1,305 episodes; step 6 prints the estimate from its
measurement. The run stops by itself if the harness or the sampled control fails.
""")

md("## 1. Settings")
code(f'''
import csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "{REPO_REF}"
assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"
REPO = "/content/efficient_vla"
VENV = "/content/venv"
PY = f"{{VENV}}/bin/python"
DRIVE_DIR = "/content/drive/MyDrive/efficient_vla_runs/step5a"
OUT_ROOT = f"{{DRIVE_DIR}}/run"
PILOT_STREAMS = [2, 4, 6, 8]
GB_PER_STREAM = 6
MAX_EPISODES = 1305
POLICY_REPO, POLICY_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"

# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
print("step 5a at", REPO_REF, "output", OUT_ROOT)


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
r = sh(f"cd {REPO} && {PY} -u noise_reachability/check_step5a.py --policy_path {POLICY_PATH}", check=False, env=RUN_ENV)
assert r.returncode == 0 and "[check] all passed" in r.stdout, "a check failed; see the output above"
''')

md("""
## 6. Throughput pilot (about 30 minutes, once)

5 worst-case episodes per process at each process count, then the count where more processes stop paying. These
episodes are not results.
""")
code('''
PILOT = Path(OUT_ROOT, "pilot.json")
cap = min(os.cpu_count(), int(ram_gb // GB_PER_STREAM) + 1)
tried = [n for n in PILOT_STREAMS if n < cap]
assert tried, f"{os.cpu_count()} CPUs and {ram_gb:.0f} GB RAM cannot run two processes"
if not PILOT.exists():
    shutil.rmtree(f"{OUT_ROOT}/pilot", ignore_errors=True)   # a pilot cut short starts again
    sh(f"bash {REPO}/noise_reachability/run_step5a_deep_noise.sh",
       env=dict(RUN_ENV, STAGE="pilot", STREAMS=" ".join(map(str, tried)), OUT_ROOT=OUT_ROOT))
    sh(f"python3 -S {REPO}/noise_reachability/analyze_step5a.py --pilot '{OUT_ROOT}/pilot' --cpus {cap} --out '{PILOT}'")
pilot = json.loads(PILOT.read_text())
STREAMS = pilot["streams"]
print(f"{STREAMS} processes, {pilot['episodes_per_hour']:.0f} episodes an hour (worst-case episodes): "
      f"at most {MAX_EPISODES / pilot['episodes_per_hour']:.1f} h for this run (estimate)")
''')

md("## 7. Launch (background)\n\nRunning this cell while a run is active does nothing.")
code('''
running = subprocess.run(["pgrep", "-f", "run_step5a_deep_nois[e]"], capture_output=True, text=True).stdout.split()
head = sh(f"git -C {REPO} log -1 --format='%H'").stdout.strip()
first = Path(OUT_ROOT, "repo_commit.txt")
started_with = first.read_text().split()[0] if first.exists() and first.read_text().strip() else None
if running:
    print("A run is already active. Not starting another.")
elif started_with and started_with != head:
    raise RuntimeError(f"this run started at commit {started_with}, the checkout is at {head}")
else:
    run = f"bash {REPO}/noise_reachability/run_step5a_deep_noise.sh"
    analyse = f"{PY} -I -S {REPO}/noise_reachability/analyze_step5a.py --root {OUT_ROOT} --out {OUT_ROOT}/step5a_report.txt"
    cmd = f"STAGE=harness {run} && STAGE=control {run} && STAGE=search {run} && STAGE=confirm {run} && {analyse}"
    Path(OUT_ROOT, "step5a_report.txt").unlink(missing_ok=True)
    subprocess.Popen(f"nohup bash -c \\"{cmd}; echo runner exit \\\\$?\\" >> '{OUT_ROOT}/runner.log' 2>&1 &", shell=True,
                     executable="/bin/bash", env=dict(RUN_ENV, OUT_ROOT=OUT_ROOT, STREAMS=str(STREAMS)),
                     start_new_session=True)
    with open(first, "a") as f:
        f.write(head + "\\n")
    print("Started; logs in", f"{OUT_ROOT}/logs")
''')

md("## 8. Progress; waits until the run ends")
code('''
def count():
    n = 0
    for f in glob.glob(f"{OUT_ROOT}/harness/seed*/per_episode_base.csv") + glob.glob(f"{OUT_ROOT}/per_episode_*.csv") \\
            + glob.glob(f"{OUT_ROOT}/confirm_*.csv"):
        n += sum(1 for _ in open(f)) - 1
    return n


while subprocess.run(["pgrep", "-f", "run_step5a_deep_nois[e]"], capture_output=True).stdout:
    print(time.strftime("%H:%M"), f"{count()}/{MAX_EPISODES} episodes at most", flush=True)
    time.sleep(600)
print(count(), "episodes; no runner active")
sh(f"tail -n 3 '{OUT_ROOT}/runner.log'", check=False)
sh(f"cat '{OUT_ROOT}/step5a_report.txt'", check=False)
''')

md("## 9. Download the results for the repo")
code('''
name = "step5a_results"
sh(f"cd '{OUT_ROOT}' && zip -q -r /content/{name}.zip . -x '*.pyc' -x '*.tmp*'")
from google.colab import files
files.download(f"/content/{name}.zip")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("noise_reachability/colab_step5a_deep_noise.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
