"""Storage of base-policy rollouts and the training samples built from them.

An episode is saved as one compressed ``.npz``: the two camera images and the 8-dimensional state exactly as
the policy saw them (after LeRobot's LIBERO env processor, i.e. what ``vla_core.policy.BasePolicy.act`` returns
as ``env_obs``; images stored as uint8, which is lossless because they come from uint8 renders), every executed
action in env units, and a JSON metadata record (task, init state, seed, success, steps).

A training sample is one recorded frame with the next ``chunk_size`` executed actions as the target and a pad
mask past the end of the episode, the format of LeRobot's dataset samples for SmolVLA. Nothing here imports
LeRobot, LIBERO or MuJoCo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EPISODE_FORMAT = 1
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PAD_KEY = "action_is_pad"
SELECTIONS = ("success", "all", "all_matched")


def image_to_uint8(image) -> np.ndarray:
    """(1, C, H, W) or (C, H, W) float image in [0, 1] -> (C, H, W) uint8; refuses anything that is not a
    uint8 render divided by 255, so the stored frame is exactly what the policy saw."""
    x = image.detach().to("cpu", torch.float32).numpy() if torch.is_tensor(image) else np.asarray(image, np.float32)
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(f"expected a batch of one image, got {x.shape}")
        x = x[0]
    if x.ndim != 3 or x.shape[0] not in (1, 3):
        raise ValueError(f"expected a (C, H, W) image, got {x.shape}")
    scaled = x * 255.0
    out = np.rint(scaled)
    if out.min() < 0 or out.max() > 255 or np.abs(scaled - out).max() > 1e-3:
        raise ValueError("image is not a uint8 render scaled to [0, 1]")
    return out.astype(np.uint8)


class EpisodeRecorder:
    """Collects one episode: images and state every ``frame_stride`` steps, the executed action every step."""

    def __init__(self, frame_stride: int = 1):
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.frame_stride = frame_stride
        self.images = {k: [] for k in IMAGE_KEYS}
        self.states, self.frame_steps, self.actions = [], [], []

    def add(self, env_obs: dict, action: np.ndarray) -> None:
        """Record the observation the action was chosen from and the action executed at this step."""
        t = len(self.actions)
        if t % self.frame_stride == 0:
            for k in IMAGE_KEYS:
                self.images[k].append(image_to_uint8(env_obs[k]))
            state = env_obs[STATE_KEY]
            state = state.detach().to("cpu", torch.float32).numpy() if torch.is_tensor(state) else np.asarray(state)
            self.states.append(np.asarray(state, np.float32).reshape(-1))
            self.frame_steps.append(t)
        self.actions.append(np.asarray(action, np.float32).reshape(-1))

    def __len__(self) -> int:
        return len(self.actions)


def save_episode(path: str | os.PathLike, recorder: EpisodeRecorder, meta: dict) -> None:
    """Write the episode atomically (a crash mid-write leaves no partial file under ``path``)."""
    if len(recorder) == 0:
        raise ValueError("empty episode")
    meta = {**meta, "format": EPISODE_FORMAT, "frame_stride": recorder.frame_stride, "n_steps": len(recorder)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        image=np.stack(recorder.images[IMAGE_KEYS[0]]), image2=np.stack(recorder.images[IMAGE_KEYS[1]]),
        state=np.stack(recorder.states), frame_steps=np.asarray(recorder.frame_steps, np.int64),
        actions=np.stack(recorder.actions), meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)


def load_episode(path: str | os.PathLike) -> dict:
    with np.load(path) as z:
        ep = {k: z[k] for k in ("image", "image2", "state", "frame_steps", "actions")}
        ep["meta"] = json.loads(str(z["meta"]))
    if ep["meta"].get("format") != EPISODE_FORMAT:
        raise ValueError(f"{path}: episode format {ep['meta'].get('format')}, expected {EPISODE_FORMAT}")
    return ep


def load_meta(path: str | os.PathLike) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def select_episodes(metas: list[dict], selection: str, seed: int = 0) -> list[int]:
    """Indices of the episodes to train on.

    ``success``: successful episodes only (the method). ``all``: every episode. ``all_matched``: as many
    episodes as ``success`` would give, drawn uniformly from all episodes (the control: same amount of data,
    no filtering by outcome).
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    successes = [i for i, m in enumerate(metas) if m["success"]]
    if selection == "success":
        return successes
    if selection == "all":
        return list(range(len(metas)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(metas), size=len(successes), replace=False).tolist())


def action_chunk(actions: np.ndarray, t: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Actions ``t .. t+chunk_size-1`` and their pad mask; positions past the episode repeat the last action
    and are marked as padding (SmolVLA's loss ignores them)."""
    n = len(actions)
    if not 0 <= t < n:
        raise IndexError(f"step {t} outside episode of {n} steps")
    idx = np.minimum(np.arange(t, t + chunk_size), n - 1)
    return actions[idx].astype(np.float32), np.arange(t, t + chunk_size) >= n


class RolloutDataset(Dataset):
    """Training samples from saved episodes: (images, state, task) -> next ``chunk_size`` executed actions."""

    def __init__(self, paths: list[str | os.PathLike], chunk_size: int = 50):
        self.episodes = [load_episode(p) for p in paths]
        self.chunk_size = chunk_size
        self.index = [(e, f) for e, ep in enumerate(self.episodes) for f in range(len(ep["frame_steps"]))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, f = self.index[i]
        ep = self.episodes[e]
        actions, pad = action_chunk(ep["actions"], int(ep["frame_steps"][f]), self.chunk_size)
        return {
            IMAGE_KEYS[0]: torch.from_numpy(ep["image"][f]).float() / 255.0,
            IMAGE_KEYS[1]: torch.from_numpy(ep["image2"][f]).float() / 255.0,
            STATE_KEY: torch.from_numpy(ep["state"][f]),
            ACTION_KEY: torch.from_numpy(actions),
            PAD_KEY: torch.from_numpy(pad),
            "task": ep["meta"]["task"],
        }
