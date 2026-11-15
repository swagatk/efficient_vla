"""Builds noise_reachability/colab_t4_check.ipynb (run from the repo root).

    python noise_reachability/build_colab_t4_check_notebook.py <commit> [output path]

The Colab T4 reproduction check of README.md ("Colab T4 check"): the base and step 10's adapter N<t>_0 on step 12's
harness episodes (12 each), compared episode by episode with step 10's Kaggle T4 rows. The code, environment, sanity and
job cells come from step 8's Kaggle notebook unchanged, as step 12's do, so the install is the one the Kaggle runs used;
the settings, GPU/Drive, adapter and job cells are Colab's. The episodes and the adapters' sha256 are fixed here from
analyze_colab_t4_check.py and step 10's archive manifest. Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "noise_reachability"))
import analyze_colab_t4_check as a  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell pulls it by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "colab_t4_check.ipynb"
step8 = json.load(open(REPO / "noise_reachability/kaggle_step8_resistant_states.ipynb"))

# step 10's adapters N<t>_0 (week6/noise_reachability_step10_kaggle, archived output MANIFEST.sha256, 2026-11-10)
ADAPTER_SHA256 = {
    0: {"adapter_config.json": "4f512e8d34de097adadb1d1e76324556a58d5a486dd98169929a772801a146fd",
        "adapter_model.safetensors": "aa163591bfc41a54da6d711908d828f7fe5b9a8d7d9f8895f1392c07e3e2d48d",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01"},
    6: {"adapter_config.json": "6e3e0f49f27fc0fef5a0d76761a971b2c5c4feec91ea81ee8cd41de657c69cc3",
        "adapter_model.safetensors": "1eb3b7d5321c1a1d7455bc491920a2f2dfcf9542b81a91751163139c16eb0498",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c"},
    7: {"adapter_config.json": "33f536c16b513ab6673fa7106a9e38be80df456a246ccef0131f4d85aefdc91e",
        "adapter_model.safetensors": "49c56ea1ea44cb6d4dd50c6362f4304531af410c355b03917c5cafa94bd00c40",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a"},
}
assert sorted(ADAPTER_SHA256) == sorted(a.TASKS)


def step8_cell(marker):
    found = [c for c in step8["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(found) == 1, marker
    return "".join(found[0]["source"])


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


N_EPISODES = len(a.TASKS) * len(a.HARNESS_SEEDS) * len(a.POLICIES)
SETTINGS = step8_cell("T_SESSION = time.time()")
SETTINGS = swap(SETTINGS, SETTINGS.split('REPO_REF = "')[1].split('"')[0], REF)
SETTINGS = SETTINGS[:SETTINGS.index("# The protocol of README.md step 8")] + f'''# The check of README.md ("Colab T4 check"); nothing here is meant to be edited. The analysis refuses anything else.
CHECKPOINT_REPO, CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"
TASKS = {list(a.TASKS)}
N_ACTION_STEPS = {a.N_ACTION_STEPS}
HARNESS_STATE = {dict(a.HARNESS_STATE)}   # step 12's harness episodes: this state x seeds 16-19 per task
HARNESS_SEEDS = "{a.HARNESS_SEEDS[0]}-{a.HARNESS_SEEDS[-1]}"
POLICIES = {dict(a.POLICIES)}   # name: (--eval_policies, csv suffix); N0 is step 10's adapter N<t>_0
ARCHIVE = "/content/drive/MyDrive/Alienware_Research_archives/efficient_vla"   # tools/archive_kaggle_output.sh folders
ADAPTERS_IN = f"{{ARCHIVE}}/noise-reachability-step-10/output/step10/runs"         # step 10's N<t>_0/adapter/
ADAPTER_SHA256 = {ADAPTER_SHA256}
DRIVE_DIR = "/content/drive/MyDrive/efficient_vla_runs/colab_t4_check"
OUT_ROOT = f"{{DRIVE_DIR}}/run"
STOP_AT_H = 4.0                    # no new work after this many hours of the session
GB_PER_JOB = 6                     # base policy + one episode in RAM per simulator process

''' + SETTINGS[SETTINGS.index("# Commits of the editable installs"):]
start = SETTINGS.index('print(f"step 8 at')
end = SETTINGS.index("def sh(cmd")
SETTINGS = SETTINGS[:start] + f'''print(f"Colab T4 check at {{REPO_REF}}: tasks {{TASKS}}, states {{HARNESS_STATE}}, seeds {{HARNESS_SEEDS}}, "
      f"policies {{list(POLICIES)}}, n_action_steps {{N_ACTION_STEPS}}, {N_EPISODES} episodes, output {{OUT_ROOT}}")


''' + SETTINGS[end:]
CODE = step8_cell("log -1 --format='%H %s'")
ENVIRONMENT = step8_cell("uv venv")
SANITY = step8_cell("RUN_ENV = dict(os.environ")
JOBS = step8_cell("def run_jobs(jobs")
JOBS = JOBS[:JOBS.index("SEARCH = f")]
JOBS = swap(JOBS, "to the zip in step 15", "to the zip in step 10")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md(f"""
# Colab T4 reproduction check: does a Colab T4 reproduce Kaggle T4 episodes exactly?

`noise_reachability/README.md`, "Colab T4 check". The frozen base and step 10's adapter N<t>_0 re-run {N_EPISODES} episodes
that step 10 ran on Kaggle T4 (tasks 0, 6, 7; step 12's harness states; seeds 16-19). If at least 11 of 12 match exactly
for each policy, Colab T4 can be used interchangeably with Kaggle T4 for paired comparisons.

**Inputs:** step 10's adapters N0_0, N6_0, N7_0 are read from the archive on Google Drive,
`MyDrive/Alienware_Research_archives/efficient_vla/noise-reachability-step-10/output/step10/runs/N<t>_0/adapter/`
(`tools/archive_kaggle_output.sh`). Step 6 checks their sha256 against the archive's manifest.

**Runtime: T4 GPU** (Runtime > Change runtime type). **Run all.** Nothing needs editing. About 1 h with the install
(estimate). If the runtime is lost, run all again: finished episodes are kept on Drive and skipped.

Download from step 10: `colab_t4_check.zip`.
""")

md("## 1. Settings")
code(SETTINGS)

md("## 2. GPU, RAM and Drive")
code('''
smi = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
gpus = [line.split(", ") for line in smi.stdout.strip().splitlines()]
assert gpus and all("T4" in g[0] for g in gpus), f"GPU {gpus}: set the runtime type to T4 GPU (this check is about T4)"
CUDA_DRIVER = float(re.search(r"CUDA Version: ([0-9.]+)", sh("nvidia-smi").stdout).group(1))
avail_gb = int(re.search(r"MemAvailable:\\s+(\\d+)", open("/proc/meminfo").read()).group(1)) / 1e6
JOBS = max(1, min(2, int((avail_gb - 4) // GB_PER_JOB), os.cpu_count()))
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
code(SANITY + '''
Path(OUT_ROOT, "environment.json").write_text(json.dumps(
    {"gpus": gpus, "cuda_driver": CUDA_DRIVER, "torch_build": TORCH_BUILD, "ram_available_gb": round(avail_gb, 1)},
    indent=1) + "\\n")
''')

md("## 6. Commit of this run and step 10's adapters (sha256 against the archive)")
code('''
head = sh(f"git -C {REPO} log -1 --format='%H'").stdout.strip()
first = Path(OUT_ROOT, "repo_commit.txt")
started_with = first.read_text().split()[0] if first.exists() and first.read_text().strip() else None
if started_with and started_with != head:
    raise RuntimeError(f"the check started at commit {started_with}, this session cloned {head}")
with open(first, "a") as f:
    f.write(head + "\\n")
for t, files in ADAPTER_SHA256.items():
    for name, want in files.items():
        p = Path(ADAPTERS_IN, f"N{t}_0", "adapter", name)
        assert p.exists(), f"upload step 10's adapter N{t}_0 to {p.parent}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == want, f"{p}: sha256 {got}, expected step 10's {want}"
    # self_filtered_bc refuses an adapter trained with another checkpoint path; Kaggle and Colab both run as root
    trained_with = json.loads(Path(ADAPTERS_IN, f"N{t}_0", "adapter", "run_config.json").read_text())["policy_path"]
    assert trained_with == POLICY_PATH, f"N{t}_0 was trained with {trained_with}, this session has {POLICY_PATH}"
print("adapters N0_0, N6_0, N7_0: sha256 equal to step 10's archive, trained with this checkpoint path")
''')

md("## 7. Helpers: jobs with resume and a deadline")
code(JOBS + '''
MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"


def sfbc(task):
    return f"cd {REPO} && {PY} -u {MAIN} --task_id {task} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}"
''')

md(f"## 8. The {N_EPISODES} episodes: base and N<t>_0 on step 12's harness episodes (one process per task and policy)")
code('''
jobs = {}
for t in TASKS:
    for policy, (flag, suffix) in POLICIES.items():
        out = f"{OUT_ROOT}/task_{t}/{policy}"
        if len(read_csv(f"{out}/per_episode_{suffix}.csv")) < n_range(HARNESS_SEEDS):
            lora = f" --lora_dir {ADAPTERS_IN}/N{t}_0/adapter" if flag == "lora" else ""
            jobs[f"task{t}_{policy}"] = (f"{sfbc(t)} --stage eval --eval_policies {flag}{lora} --eval_seeds {HARNESS_SEEDS} "
                                         f"--eval_init_states {HARNESS_STATE[t]} --eval_output_dir {out} "
                                         f"--max_episodes_per_process 10", 0)
print(f"{len(jobs)} of {len(TASKS) * len(POLICIES)} jobs to run", flush=True)
codes = run_jobs(jobs, max_parallel=JOBS)
if failed(codes):
    fail(f"evaluation failed: {failed(codes)}; see {OUT_ROOT}/logs")
''')

md("## 9. The comparison with step 10's Kaggle T4 episodes")
code('''
if not FAILURE:
    r = sh(f"cd {REPO} && python3 -S noise_reachability/analyze_colab_t4_check.py --root {OUT_ROOT} "
           f"--out {OUT_ROOT}/colab_t4_check.txt", check=False)
    if r.returncode != 0:
        fail("the analysis refused the run: read its message above")
''')

md("## 10. Download")
code('''
sh(f"cd '{OUT_ROOT}' && zip -q -r /content/colab_t4_check.zip . -x '*.pyc' -x '*.tmp*'")
from google.colab import files
files.download("/content/colab_t4_check.zip")
if FAILURE:
    raise RuntimeError(FAILURE)
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT, "at", REF)
