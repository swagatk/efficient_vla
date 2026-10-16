"""One LIBERO episode from a given init state, and per-episode result files.

Result rows use the columns of baseline_performance's per_episode.csv (task_id, seed, init_state, success),
plus ``steps``, so ``baseline_performance/compare_per_state.py`` compares two policies episode by episode.
"""
from __future__ import annotations

import csv
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

EPISODE_FIELDS = ["task_id", "seed", "init_state", "success", "steps"]
# Larger than the 50 LIBERO init states, so (seed, init_state) pairs never share a seed.
SEED_STRIDE = 1000


def episode_seed(seed: int, init_state: int) -> int:
    """Seed for the environment and the policy's sampling noise in one episode."""
    if not 0 <= init_state < SEED_STRIDE or seed < 0:
        raise ValueError(f"need seed >= 0 and 0 <= init_state < {SEED_STRIDE}, got {seed}, {init_state}")
    return seed * SEED_STRIDE + init_state


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(env, init_state: int, seed: int, choose_action: Callable, max_steps: int,
                on_step: Callable | None = None, on_reset: Callable | None = None) -> dict:
    """Run one episode from benchmark init state ``init_state``.

    ``choose_action(obs) -> np.ndarray`` returns the action to execute. ``on_reset(obs)`` is called once after
    the reset; ``on_step(obs, action, reward, next_obs, terminated, info)`` after every step. The episode ends
    at success, at a terminal state, or after ``max_steps`` steps, as in lerobot-eval.
    """
    from vla_core.envs import reset_to_init_state

    s = episode_seed(seed, init_state)
    seed_everything(s)
    obs, _ = reset_to_init_state(env, init_state, seed=s)
    if on_reset is not None:
        on_reset(obs)
    success, steps = False, 0
    for steps in range(1, max_steps + 1):
        action = choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        success = bool(info.get("is_success", False))
        if on_step is not None:
            on_step(obs, action, reward, next_obs, terminated, info)
        obs = next_obs
        if terminated or truncated or success:
            break
    return {"success": success, "steps": steps}


def load_episode_rows(path: str | os.PathLike) -> dict[tuple[int, int], dict]:
    """{(seed, init_state): row} of a per-episode CSV, or {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return {(int(r["seed"]), int(r["init_state"])): r for r in csv.DictReader(f)}


def append_episode_row(path: str | os.PathLike, row: dict) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in EPISODE_FIELDS})
