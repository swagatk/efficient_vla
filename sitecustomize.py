"""
Site customization hook: Automatically loads robosuite_mujoco_patch
when running any script (like lerobot-eval) with repo root in PYTHONPATH.
"""

try:
    import robosuite_mujoco_patch  # noqa: F401
except ImportError:
    pass
