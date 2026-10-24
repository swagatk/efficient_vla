"""Build continued_training/colab_continued_training.ipynb; the code and environment cells are taken verbatim from
baseline_performance/kaggle_baseline_lerobot_eval.ipynb, so training and evaluation install the same stack.

    python -S continued_training/build_colab_notebook.py continued_training/colab_continued_training.ipynb
"""
import json
import sys

REPO = "/home/swagat/GIT/efficient_vla"
base = json.load(open(f"{REPO}/baseline_performance/kaggle_baseline_lerobot_eval.ipynb"))
src = lambda i: "".join(base["cells"][i]["source"])  # noqa: E731
assert "git clone" in src(6) and "uv venv" in src(8), "baseline notebook cells moved; update the indices"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md('''
# Continued training of SmolVLA on LIBERO-10 (Colab, A100)

Continues training `lerobot/smolvla_libero` on the LIBERO-10 demonstrations of `lerobot/libero` with
`lerobot-train`, through `continued_training/run_continued_training.sh`. The hypothesis, settings and the
evaluation that follows are in `continued_training/README.md`. This notebook only trains; the evaluation runs on
Kaggle with `baseline_performance/kaggle_baseline_lerobot_eval.ipynb` (`CHECKPOINT = "continued"`).

**Before you start**

- Runtime > Change runtime type > **A100 GPU** (High-RAM if offered).
- At least **8 GB free on Google Drive**: the newest checkpoint with its optimizer state (about 2.4 GB) is
  mirrored there while training runs, and the final model (0.9 GB) is copied there at the end.

**How to run.** Run the cells in order. Step 7 measures the training speed over 100 steps and sets the number of
steps so that training takes about `TRAIN_HOURS`. Step 8 starts training in the background and step 9b keeps a
cell running until it ends, so Colab does not treat the runtime as idle. Keep the tab open and the computer awake:
closing the laptop lid ended the runtime once.

**If the runtime is lost:** run steps 1 to 8 again. Step 7 then reads the step budget from Drive instead of
measuring it, and step 8 restores the last mirrored checkpoint and resumes (at most `SAVE_FREQ` steps are
repeated).
''')

md("## 1. Settings")
code('''
import csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "main"
REPO = "/content/efficient_vla"
VENV = "/content/venv"
PY = f"{VENV}/bin/python"

DRIVE_DIR = "/content/drive/MyDrive/efficient_vla_runs/continued_training"
SYNC_DIR = f"{DRIVE_DIR}/run_seed0"           # mirror of the newest checkpoint, the log and provenance
FINAL_DIR = f"{DRIVE_DIR}/smolvla_libero10_continued_final"   # the model to upload to Kaggle
OUT_DIR = "/content/ct_run_seed0"             # lerobot-train output on the fast local disk
TRIAL_DIR = "/content/ct_trial"               # speed trial, not used afterwards

TRAIN_HOURS = 3.0       # target duration of the training loop; set from the remaining A100 budget
MAX_STEPS = 30_000      # upper limit whatever the speed (about 9.5 passes over LIBERO-10 at batch 32)
SAVE_FREQ = 2_500
BATCH_SIZE = 32         # as the checkpoint; the remaining settings are the defaults of run_continued_training.sh
SEED = 0

# Commits of the editable installs in the local lerobot_env that produced the existing results.
LEROBOT_COMMIT = "b6ec0060779550c0a157ae34feb89e0cf86012a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
POLICY_REPO, POLICY_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
DATASET_REPO, DATASET_REVISION = "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LIBERO10_EPISODES = (0, 378)   # first and last episode index of task_index 0-9 at this revision


def sh(cmd, check=True, env=None):
    # Capture and print: notebooks do not show output that a subprocess writes directly.
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {r.returncode}: {cmd}")
    return r
''')

md("## 2. GPU and Drive")
code('''
smi = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
gpus = [line.split(", ") for line in smi.stdout.strip().splitlines()]
assert gpus, "No GPU. Set the runtime type to A100 GPU."
if "A100" not in gpus[0][0]:
    print(f"WARNING: {gpus[0][0]}, not an A100; the step budget of step 7 still adapts to its speed")
CUDA_DRIVER = float(re.search(r"CUDA Version: ([0-9.]+)", sh("nvidia-smi").stdout).group(1))
print(f"driver supports CUDA {CUDA_DRIVER}; {os.cpu_count()} CPU cores")

from google.colab import drive
drive.mount("/content/drive")
os.makedirs(SYNC_DIR, exist_ok=True)
free_gb = shutil.disk_usage("/content/drive/MyDrive").free / 1e9
print(f"Drive: {free_gb:.1f} GB free; output folder {DRIVE_DIR}")
assert free_gb > 8, "need at least 8 GB free on Drive"
''')

md("## 3. Code")
code(src(6))
md(src(7))
code(src(8))

md('''
## 5. Sanity checks and tests

PyTorch sees the GPU and runs bfloat16, MuJoCo renders headless (needed for the rollout of step 10), and the
repo's CPU tests pass.
''')
code('''
RUN_ENV = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MPLBACKEND="Agg", WANDB_MODE="disabled", PYTHONPATH=REPO, PYTHON_BIN=PY)

sh(f"""{PY} -c "
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
print('bfloat16 matmul ok', float((x @ x).float().abs().mean()) > 0)
" """, env=RUN_ENV)
egl = sh(f"{PY} -c \\"import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); print('EGL context ok')\\"", check=False, env=RUN_ENV)
assert "EGL context ok" in egl.stdout, "EGL rendering failed"
sh(f"cd {REPO} && {PY} -m pytest tests/ -q -p no:cacheprovider", env=RUN_ENV)
''')

md('''
## 6. Data and checkpoint checks

Downloads the starting checkpoint and the dataset's metadata and state/action tables (about 20 MB; the videos
are downloaded by `lerobot-train`), then checks that

- episodes 0-378 are exactly the episodes of `task_index` 0-9, the LIBERO-10 tasks;
- the dataset's state and action stats, which `lerobot-train` uses for normalisation, equal the checkpoint's.
''')
code('''
check = f"""
import glob, json, sys
import pandas as pd, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
policy = snapshot_download("{POLICY_REPO}", revision="{POLICY_REVISION}")
ds = snapshot_download("{DATASET_REPO}", repo_type="dataset", revision="{DATASET_REVISION}", allow_patterns=["data/*", "meta/*"])
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in sorted(glob.glob(ds + "/data/*/*.parquet"))])
per_ep = df.groupby("episode_index").task_index.agg(["min", "max"])
assert (per_ep["min"] == per_ep["max"]).all(), "an episode has several tasks"
lib10 = sorted(per_ep.index[per_ep["min"] < 10])
assert lib10 == list(range({LIBERO10_EPISODES[0]}, {LIBERO10_EPISODES[1]} + 1)), (lib10[:3], lib10[-3:], len(lib10))
tasks = pd.read_parquet(ds + "/meta/tasks.parquet")
print(len(lib10), "LIBERO-10 episodes,", int(df.episode_index.isin(lib10).sum()), "frames; tasks:")
print("\\\\n".join(f"  {{i}}: {{t}}" for t, i in zip(tasks.index, tasks.task_index) if i < 10))
stats = json.load(open(ds + "/meta/stats.json"))
norm = load_file(policy + "/policy_preprocessor_step_5_normalizer_processor.safetensors")
for key in ("action", "observation.state"):
    for s in ("mean", "std"):
        diff = (torch.tensor(stats[key][s]) - norm[key + "." + s].float()).abs().max().item()
        assert diff < 1e-6, (key, s, diff)
print("state and action stats equal the checkpoint's")
print(policy)
"""
Path("/content/ct_checks.py").write_text(check)
r = sh(f"{PY} /content/ct_checks.py", env=RUN_ENV)
POLICY_PATH = r.stdout.strip().splitlines()[-1]
''')

md('''
## 7. Speed trial and step budget (skipped when resuming)

Trains 100 steps at the real batch size in a separate folder and measures seconds per step between steps 20 and
100, after the first batches have warmed up the data loader. `data_s` in the log is the time spent waiting for
data: if it is a large share of `step_s`, the video decoding is the bottleneck. The step budget is
`TRAIN_HOURS` divided by that speed, rounded down to a multiple of `SAVE_FREQ`, at most `MAX_STEPS`.
''')
code('''
RESUMING = Path(SYNC_DIR, "last_step").exists() or Path(OUT_DIR, "checkpoints", "last").exists()
plan_path = Path(SYNC_DIR, "plan.json")
if RESUMING:
    plan = json.load(open(plan_path))
    print("resuming; step budget from the first session:", plan)
else:
    shutil.rmtree(TRIAL_DIR, ignore_errors=True)
    t0 = time.time()
    sh(f"bash {REPO}/continued_training/run_continued_training.sh",
       env=dict(RUN_ENV, OUT_DIR=TRIAL_DIR, STEPS="100", SAVE_FREQ="100000", LOG_FREQ="10", BATCH_SIZE=str(BATCH_SIZE)))
    log = open(f"{TRIAL_DIR}.train.log").read()
    stamps = {int(m.group(2)): time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
              for m in re.finditer(r"INFO (\\S+ \\S+) \\S+ step:(\\d+) ", log)}
    sec_per_step = (stamps[100] - stamps[20]) / 80
    last = [l for l in log.splitlines() if "step:100 " in l][-1]
    print(last[last.index("step:100"):][:200])
    steps = int(min(MAX_STEPS, TRAIN_HOURS * 3600 / sec_per_step) // SAVE_FREQ * SAVE_FREQ)
    assert steps >= SAVE_FREQ, f"{sec_per_step:.2f} s/step leaves fewer than {SAVE_FREQ} steps in {TRAIN_HOURS} h"
    plan = dict(steps=steps, sec_per_step=round(sec_per_step, 3), train_hours=TRAIN_HOURS, batch_size=BATCH_SIZE,
                save_freq=SAVE_FREQ, seed=SEED, gpu=gpus[0][0], trial_minutes=round((time.time() - t0) / 60, 1))
    json.dump(plan, open(plan_path, "w"), indent=1)
    shutil.rmtree(TRIAL_DIR, ignore_errors=True)
    print(plan)
    print(f"-> {steps} steps, about {steps * sec_per_step / 3600:.1f} h of training "
          f"({steps * BATCH_SIZE / 101469:.1f} passes over LIBERO-10)")
''')

md('''
## 8. Train (background)

Starts `run_continued_training.sh` with the step budget of step 7 and returns immediately. Running it again while
training is active does nothing. On a new runtime it restores the last checkpoint from Drive and resumes.
''')
code('''
running = subprocess.run(["pgrep", "-f", "lerobot[-]train"], capture_output=True, text=True).stdout.split()
if running:
    print("training is already running (pids:", " ".join(running) + "); not starting another")
else:
    env = dict(RUN_ENV, OUT_DIR=OUT_DIR, SYNC_DIR=SYNC_DIR, STEPS=str(plan["steps"]), SAVE_FREQ=str(SAVE_FREQ),
               BATCH_SIZE=str(BATCH_SIZE), SEED=str(SEED), RESUME="1" if RESUMING else "0")
    sh(f"git -C {REPO} rev-parse HEAD > '{SYNC_DIR}/git_commit.txt'; "
       f"git -C {REPO} status --porcelain --untracked-files=no > '{SYNC_DIR}/git_dirty.txt'")
    Path("/content/ct_versions.py").write_text(
        "import importlib.metadata as m, torch\\n"
        "print({p: m.version(p) for p in ['lerobot', 'torch', 'transformers', 'av']}, torch.cuda.get_device_name(0))\\n")
    sh(f"{PY} /content/ct_versions.py >> '{SYNC_DIR}/versions.txt'", env=RUN_ENV)
    subprocess.Popen(f"nohup bash {REPO}/continued_training/run_continued_training.sh >> '{SYNC_DIR}/runner.log' 2>&1 &",
                     shell=True, executable="/bin/bash", env=env, start_new_session=True)
    print("started; log", f"{OUT_DIR}.train.log", "(mirrored to Drive with each checkpoint)")
''')

md("## 9. Progress (re-run any time)")
code('''
log_path = f"{OUT_DIR}.train.log"
lines = [l for l in open(log_path).read().splitlines() if " step:" in l] if os.path.exists(log_path) else []
if lines:
    m = re.search(r"step:(\\d+\\S*)", lines[-1])
    print(lines[-1][lines[-1].index("step:"):][:220])
    done = int(float(m.group(1).rstrip("K")) * (1000 if m.group(1).endswith("K") else 1))
    print(f"{done}/{plan['steps']} steps; about {(plan['steps'] - done) * plan['sec_per_step'] / 60:.0f} min left")
print("running:", bool(subprocess.run(["pgrep", "-f", "lerobot[-]train"], capture_output=True, text=True).stdout))
sh(f"cat '{SYNC_DIR}/last_step'", check=False)
sh(f"tail -n 3 '{SYNC_DIR}/runner.log'", check=False)
sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", check=False)
''')

md('''
## 9b. Wait until training ends

Prints progress every 5 minutes until the runner script exits (it waits on the script, not on `lerobot-train`, so
the checkpoint restore at the start of a resume counts). Interrupting this cell does not stop training.
''')
code('''
log_path = f"{OUT_DIR}.train.log"
while subprocess.run(["pgrep", "-f", "run_continued_[t]raining"], capture_output=True).stdout:
    lines = [l for l in open(log_path).read().splitlines() if " step:" in l] if os.path.exists(log_path) else []
    print(time.strftime("%H:%M"), lines[-1][lines[-1].index("step:"):][:120] if lines else "restoring / starting",
          flush=True)
    time.sleep(300)
print("runner finished")
sh(f"tail -n 3 '{SYNC_DIR}/runner.log'", check=False)
''')

md('''
## 10. Check the final checkpoint and copy it for Kaggle

Runs after training has finished (`lerobot-train exit code 0` in step 9). Checks that the final step was saved,
copies its `pretrained_model` to `FINAL_DIR` on Drive, and runs one rollout through `evaluate_all.py` (task 3,
seed 0, init state 0, the settings of the Kaggle evaluation) to check that it loads and acts. One episode says
nothing about success rates.
''')
code('''
assert open(f"{SYNC_DIR}/last_step").read().strip() == f"{plan['steps']:06d}", "training has not finished"
assert "lerobot-train exit code 0" in open(f"{SYNC_DIR}/runner.log").read()
shutil.rmtree(FINAL_DIR, ignore_errors=True)
shutil.copytree(f"{SYNC_DIR}/last/pretrained_model", FINAL_DIR)
sh(f"du -sh '{FINAL_DIR}'; ls '{FINAL_DIR}'")
rename = json.dumps({"observation.images.image": "observation.images.camera1",
                     "observation.images.image2": "observation.images.camera2"})
sh(f"cd {REPO} && {PY} baseline_performance/evaluate_all.py --resume_dir {DRIVE_DIR}/rollout_check --tasks 3 --seeds 0 "
   f"--n_episodes 1 --policy_path '{FINAL_DIR}' --lerobot_eval {VENV}/bin/lerobot-eval --n_action_steps 1 "
   f"--rename_map '{rename}' --hard_stop_min 30", env=RUN_ENV)
print(open(f"{DRIVE_DIR}/rollout_check/per_episode.csv").read())
''')

md('''
## 11. Files for the repo, and the model for Kaggle

- **Repo:** the zip below holds the training log, the plan, provenance and the final `train_config.json`. Unpack
  it into `week6/continued_training_colab/`.
- **Kaggle:** download `smolvla_libero10_continued_final/` from Drive (about 0.9 GB), upload it as a private
  Kaggle dataset (**Datasets > New Dataset**), add it as input to `kaggle_baseline_lerobot_eval.ipynb`, set
  `CHECKPOINT = "continued"` there and Save & Run All.
''')
code('''
shutil.copy(f"{OUT_DIR}.train.log", f"{SYNC_DIR}/train.log") if os.path.exists(f"{OUT_DIR}.train.log") else None
sh(f"cp '{FINAL_DIR}/train_config.json' '{SYNC_DIR}/final_train_config.json'")
sh(f"cd '{DRIVE_DIR}' && zip -q -r /content/continued_training_colab.zip run_seed0 rollout_check -x 'run_seed0/last/*'")
sh("du -h /content/continued_training_colab.zip")
from google.colab import files
files.download("/content/continued_training_colab.zip")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:  # store sources as lists of lines, as Jupyter does
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
