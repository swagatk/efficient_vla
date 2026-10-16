"""LIBERO environments built exactly as lerobot-eval builds them.

`make_libero_env` constructs LeRobot's ``LiberoEnv`` with the arguments ``lerobot-eval`` passes for
``--env.type=libero``, so a policy run through ``vla_core`` sees the same environment as
``baseline_performance/results_300926``: benchmark init states set after the reset, 10 no-op settling steps,
relative (delta) control, 256x256 cameras, horizon 520. The 180-degree image rotation and the 8-dimensional
state are applied by LeRobot's LIBERO processor step in ``vla_core.policy`` (FAIL-006 lists what goes wrong
when these are re-implemented by hand).

Nothing here imports LIBERO, LeRobot or MuJoCo at module import time.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 7


def libero_env_config(task_id: int, suite: str = "libero_10", episode_length: int = 520, observation_size: int = 256,
                      hard_reset: bool = True):
    """LeRobot's LIBERO env config with the lerobot-eval settings used for results_300926."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    return LiberoEnvConfig(task=suite, task_ids=[task_id], episode_length=episode_length,
                           observation_height=observation_size, observation_width=observation_size,
                           hard_reset=hard_reset)


def make_libero_env(env_cfg):
    """One LeRobot ``LiberoEnv`` for the single task in ``env_cfg.task_ids``, as lerobot-eval creates it.

    ``hard_reset=True`` (lerobot-eval's default) rebuilds the simulator on every reset; ``False`` keeps one
    simulator and restores the init state, which is faster and avoids the resource build-up of FAIL-004 over
    long training runs, but is not bit-identical to a hard reset. Evaluate with hard resets.
    """
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.envs.utils import parse_camera_names

    if not env_cfg.task_ids or len(env_cfg.task_ids) != 1:
        raise ValueError(f"expected exactly one task id, got {env_cfg.task_ids}")
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    return LiberoEnv(
        task_suite=_get_suite(env_cfg.task), task_id=env_cfg.task_ids[0], task_suite_name=env_cfg.task,
        camera_name=parse_camera_names(env_cfg.camera_name), init_states=env_cfg.init_states,
        episode_length=env_cfg.episode_length, episode_index=0, n_envs=1, control_mode=env_cfg.control_mode,
        camera_name_mapping=env_cfg.camera_name_mapping, is_libero_plus=env_cfg.is_libero_plus, **gym_kwargs,
    )


def reset_to_init_state(env, init_state: int, seed: int | None = None):
    """Reset ``env`` to benchmark init state ``init_state`` (then settle), as lerobot-eval's episode ``init_state``.

    LiberoEnv applies the init state after its own reset and then runs the settling steps, so nothing may reset
    the environment again afterwards: a second reset re-randomises the scene and discards the init state.
    """
    env.init_state_id = int(init_state)
    return env.reset(seed=seed)


def add_batch_dim(obs: Any) -> Any:
    """Give every array a leading batch dimension of 1, as a one-env SyncVectorEnv returns observations."""
    if isinstance(obs, dict):
        return {k: add_batch_dim(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return obs[None]
    return obs


def parse_index_ranges(spec: str) -> list[int]:
    """'0-9' -> [0..9]; '10-49,3' -> [10..49, 3]. Used for init-state and seed lists on the command line."""
    out: list[int] = []
    for part in str(spec).replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"empty range {part!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate indices in {spec!r}")
    return out


def task_sim(env):
    """The robosuite environment inside a LiberoEnv (for reward shaping); valid after the first reset."""
    if env._env is None:
        raise RuntimeError("the LIBERO simulator is created on the first reset")
    return env._env.env
