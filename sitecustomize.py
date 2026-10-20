"""
Site customization hook: Automatically loads robosuite_mujoco_patch
when running any script (like lerobot-eval) with repo root in PYTHONPATH.

Not in pyGLFW's library-probe processes (``python -c`` with source calling glfwGetVersion): they inherit
PYTHONPATH, and loading the patch there imports GLFW again, which starts another probe, without end
(tests/test_sitecustomize.py).
"""
import sys

_argv = getattr(sys, "orig_argv", [])
if not any("glfwGetVersion" in a for a in _argv):
    try:
        import robosuite_mujoco_patch  # noqa: F401
    except ImportError:
        pass
