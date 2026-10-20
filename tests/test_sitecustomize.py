"""sitecustomize must not load robosuite_mujoco_patch into pyGLFW's library-probe processes.

pyGLFW checks each candidate library by running ``python -c <source calling glfwGetVersion>``. Those children
inherit PYTHONPATH, so sitecustomize loaded the patch, which imported MuJoCo/robosuite, which imported GLFW
again, which started another probe: an endless chain of processes (one every ~30 s, ~190 MB each) whenever
MUJOCO_GL was not egl. It filled the 28 GB of RAM on 2026-10-19 (logs/crash_monitor/monitor.log).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK = "import sys; print('robosuite_mujoco_patch' in sys.modules)"


def run(source):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "MUJOCO_GL": "egl"}
    out = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=300,
                         cwd="/")
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_patch_loaded_in_ordinary_processes():
    assert run(CHECK) == "True"


def test_patch_skipped_in_glfw_probe_processes():
    probe = "# library_handle.glfwGetVersion(major, minor, rev)\n" + CHECK
    assert run(probe) == "False"
