"""Adding a learned correction to a base action, in the environment's action units.

The residual must be added after the base policy's action has been un-normalised. Adding it to the normalised
action and clamping that to [-1, 1] (what train_online_sac.py did before the fix) also clamps the base action
to +-1 standard deviation of the training actions, so even a zero residual changed the base policy.
"""
from __future__ import annotations

import numpy as np

from vla_core.envs import ACTION_HIGH, ACTION_LOW


def apply_residual(base_action: np.ndarray, delta: np.ndarray, max_residual: float | np.ndarray) -> np.ndarray:
    """``base + max_residual * delta``, with ``delta`` in [-1, 1], clipped to the env's action bounds.

    robosuite clips commands to [-1, 1] itself, so the clip changes nothing the environment would not; it
    only keeps the stored action equal to the executed one. With ``delta == 0`` the base action is returned
    as executed.
    """
    delta = np.asarray(delta, dtype=np.float32)
    if np.any(np.abs(delta) > 1.0 + 1e-6):
        raise ValueError("delta must lie in [-1, 1]; scale it with max_residual")
    out = np.asarray(base_action, dtype=np.float32) + np.asarray(max_residual, dtype=np.float32) * delta
    return np.clip(out, ACTION_LOW, ACTION_HIGH).astype(np.float32)
