"""Builds noise_reachability/kaggle_step17_u1_cpu.ipynb (run from the repo root).

    python noise_reachability/build_step17_u1_notebook.py <commit> [output path]

Step 17's test U1 (README.md, step 17) on a Kaggle CPU session, so the ~1,000 critic refits do not take the local
machine's cores. It runs ``step17_g1b.py`` unchanged at a pushed commit, single-threaded (BLAS threads on these small
matrices only add overhead), with the local ``lerobot_env``'s numpy and scipy, on the features file of the checked
step-17 run (sha256-checked). The rule's settings are the module's constants; the notebook passes none.
Edit this builder, not the notebook.
"""
import json
import sys
from pathlib import Path

REPO = Path.cwd()
REF = sys.argv[1] if len(sys.argv) > 1 else "SET_AFTER_COMMIT"
assert REF == "SET_AFTER_COMMIT" or len(REF) == 40, "the full commit is needed: the code cell checks it out by name"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "noise_reachability" / "kaggle_step17_u1_cpu.ipynb"

FEATURES_SHA256 = "07abc733f4e1ac481bdfe542dcba582bcfa8a4063de4394401580438ba0de015"   # checked 2026-11-24
NUMPY, SCIPY = "2.2.6", "1.17.1"                                                          # local lerobot_env
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# noise_reachability step 17: test U1 on CPU (Kaggle, no accelerator)

Step 17 of `noise_reachability/README.md`: `step17_g1b.py` (the critic over SmolVLA's first-inference features, 1,000
stratified permutations) on the checked feature file of the step-17 run. No GPU and no simulator.

**Input:** the step-17 features notebook's output (**Add Input > Your Work**, the version that produced the checked
`step17_results.zip`), or a dataset holding `step17_features.json`. The file is found by name and its sha256 must be
the checked one. **Run with Save & Run All (Accelerator: None, Internet on).** Nothing needs editing. About 1-2 h (an
estimate: ~55 min single-threaded on the local machine).

Download from the Output tab: `step17_u1_results.zip`.
""")

md("## 1. Settings")
code(f'''
import glob, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

T0 = time.time()
REPO_URL = "https://github.com/swagatk/efficient_vla.git"
REPO_REF = "{REF}"
assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no commit yet"
REPO, VENV = "/tmp/efficient_vla", "/tmp/venv_u1"
PY = f"{{VENV}}/bin/python"
FEATURES_SHA256 = "{FEATURES_SHA256}"
NUMPY, SCIPY = "{NUMPY}", "{SCIPY}"
OUT = Path("/kaggle/working/step17_u1")
OUT.mkdir(parents=True, exist_ok=True)
THREADS = {{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def sh(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"failed: {{cmd}}")
    return r


hits = sorted(set(glob.glob("/kaggle/input/**/step17_features.json", recursive=True)))
good = [p for p in hits if hashlib.sha256(Path(p).read_bytes()).hexdigest() == FEATURES_SHA256]
assert good, f"no step17_features.json with sha256 {{FEATURES_SHA256[:12]}}... under /kaggle/input (found {{hits}})"
FEATURES = good[0]
print(f"step 17 U1 at {{REPO_REF[:7]}}: features {{FEATURES}} (sha256 {{FEATURES_SHA256[:12]}}... ok), "
      f"numpy {{NUMPY}}, scipy {{SCIPY}}, 1 BLAS thread, output {{OUT}}")
''')

md("## 2. Code and environment")
code('''
if not Path(REPO, ".git").exists():
    sh(f"git clone {REPO_URL} {REPO}")
sh(f"git -C {REPO} fetch origin && git -C {REPO} checkout --detach {REPO_REF}")
head = sh(f"git -C {REPO} rev-parse HEAD").stdout.strip()
assert head == REPO_REF, f"checked out {head}, expected {REPO_REF}"
dirty = sh(f"git -C {REPO} status --porcelain").stdout.strip()
assert not dirty, f"the clone is not clean: {dirty}"
sh(f"{sys.executable} -m venv {VENV} && {PY} -m pip install -q numpy=={NUMPY} scipy=={SCIPY} pytest")
print(sh(f"{PY} -c 'import numpy, scipy, sys; print(numpy.__version__, scipy.__version__, sys.version.split()[0])'").stdout)
print(f"{os.cpu_count()} CPU cores; {platform.processor() or platform.machine()}")
''')

md("## 3. The rule's tests at this commit")
code('''
env = {**os.environ, **THREADS}
r = subprocess.run(f"cd {REPO} && {PY} -m pytest tests/test_step17_g1b.py -q -p no:cacheprovider", shell=True,
                   capture_output=True, text=True, env=env)
print(r.stdout[-1500:])
assert r.returncode == 0, "the rule's tests fail on this machine: do not read U1"
''')

md("## 4. Test U1 (about 1-2 h)")
code('''
t = time.time()
cmd = (f"cd {REPO} && {PY} -u noise_reachability/step17_g1b.py --features {FEATURES} "
       f"--out {OUT}/step17_u1.txt")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **THREADS})
(OUT / "step17_u1_stdout.txt").write_text(r.stdout + r.stderr)
print(r.stdout, r.stderr[-3000:])
assert r.returncode == 0, "step17_g1b.py failed"
minutes = (time.time() - t) / 60
print(f"U1 took {minutes:.1f} min")
''')

md("## 5. Provenance and download")
code('''
versions = sh(f"{PY} -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'").stdout.split()
prov = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": head, "git_dirty": bool(dirty),
        "command": cmd, "threads": THREADS, "numpy": versions[0], "scipy": versions[1],
        "features": FEATURES, "features_sha256": FEATURES_SHA256, "cpu_count": os.cpu_count(),
        "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\\n")[0].strip("\\t: "),
        "u1_minutes": round(minutes, 1), "session_minutes": round((time.time() - T0) / 60, 1)}
(OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\\n")
sh(f"cd /kaggle/working && rm -f step17_u1_results.zip && zip -qr step17_u1_results.zip step17_u1")
print(json.dumps(prov, indent=1))
print("download step17_u1_results.zip from the Output tab")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT} pinned to {REF}")
