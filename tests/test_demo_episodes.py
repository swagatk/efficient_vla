import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))

import demo_episodes as de  # noqa: E402
from rollout_data import IMAGE_KEYS, STATE_KEY, RolloutDataset, load_episode, save_episode  # noqa: E402


def frames(n, episode=8, task=5, seed=0):
    """Dataset-like frames: float images that are exact uint8 / 255, state, action, indices."""
    g = np.random.default_rng(seed)
    out = []
    for k in range(n):
        f = {key: torch.from_numpy(g.integers(0, 256, (3, 4, 4)).astype(np.float32) / 255.0) for key in IMAGE_KEYS}
        f[STATE_KEY] = torch.from_numpy(g.normal(size=8).astype(np.float32))
        f["action"] = torch.from_numpy(g.normal(size=7).astype(np.float32))
        f.update(episode_index=torch.tensor(episode), task_index=torch.tensor(task), frame_index=torch.tensor(k))
        out.append(f)
    return out


def test_task_index_by_exact_string():
    tasks = {"put both the alphabet soup and the cream cheese box in the basket": 4,
             "put both the alphabet soup and the tomato sauce in the basket": 5}
    assert de.task_index_for(tasks, "put both the alphabet soup and the tomato sauce in the basket") == 5
    with pytest.raises(ValueError):
        de.task_index_for(tasks, "put both the alphabet soup and the tomato sauce")
    with pytest.raises(ValueError):
        de.task_index_for({"a": 1, "b": 2}, "c")


def test_episodes_of_task_and_task_per_episode():
    m = de.task_per_episode([3, 3, 8, 8, 400, 2], [5, 5, 5, 5, 5, 4])
    assert m == {3: 5, 8: 5, 400: 5, 2: 4}
    assert de.episodes_of_task(m, 5) == [3, 8]                          # 400 is past LIBERO-10's 379 episodes
    assert de.episodes_of_task(m, 5, n_episodes=401) == [3, 8, 400]
    assert de.episodes_of_task({378: 5, 379: 5}, 5) == [378]              # 0-378 only
    with pytest.raises(ValueError, match="tasks 5 and 4"):
        de.task_per_episode([3, 3], [5, 4])


def test_record_demo_round_trip_matches_frames(tmp_path):
    fr = frames(7)
    rec = de.record_demo(fr, frame_stride=2)
    meta = de.demo_meta(0, "task text", 8, 2, len(rec))
    save_episode(tmp_path / "episodes" / de.episode_file(2, 8), rec, meta)
    ep = load_episode(tmp_path / "episodes" / "demo002_ep0008.npz")
    assert ep["frame_steps"].tolist() == [0, 2, 4, 6] and len(ep["actions"]) == 7
    for j, t in enumerate([0, 2, 4, 6]):
        assert np.array_equal(ep["image"][j], np.rint(fr[t][IMAGE_KEYS[0]].numpy() * 255).astype(np.uint8))
        assert np.array_equal(ep["image2"][j], np.rint(fr[t][IMAGE_KEYS[1]].numpy() * 255).astype(np.uint8))
        assert np.array_equal(ep["state"][j], fr[t][STATE_KEY].numpy())
    assert np.array_equal(ep["actions"], np.stack([f["action"].numpy() for f in fr]))
    assert ep["meta"]["source"] == "demonstration" and ep["meta"]["dataset_revision"] == de.DATASET_REVISION
    assert ep["meta"]["task_id"] == 0 and ep["meta"]["init_state"] is None and ep["meta"]["success"] is True
    # the training sample of frame 1 (step 2) targets actions 2..51, padded past step 6
    s = RolloutDataset([tmp_path / "episodes" / "demo002_ep0008.npz"], chunk_size=50)[1]
    assert torch.equal(s["action"][:5], torch.from_numpy(ep["actions"][2:7]))
    assert s["action_is_pad"][:5].sum() == 0 and s["action_is_pad"][5:].all() and s["task"] == "task text"


def test_record_demo_refuses_mixed_or_out_of_order_frames():
    with pytest.raises(ValueError, match="episode 9"):
        de.record_demo(frames(3) + frames(1, episode=9), 1)
    with pytest.raises(ValueError, match="task 4"):
        de.record_demo(frames(3) + frames(1, task=4), 1)
    fr = frames(4)
    fr[2], fr[3] = fr[3], fr[2]
    with pytest.raises(ValueError, match="frame_index"):
        de.record_demo(fr, 1)
    lossy = frames(1)
    lossy[0][IMAGE_KEYS[0]] = lossy[0][IMAGE_KEYS[0]] + 0.3 / 255
    with pytest.raises(ValueError, match="uint8"):
        de.record_demo(lossy, 1)
    with pytest.raises(ValueError, match="no frames"):
        de.record_demo([], 1)


def save_demos(tmp_path, n, task_id=0):
    for k in range(n):
        rec = de.record_demo(frames(3, episode=10 + k, seed=k), 1)
        save_episode(tmp_path / "episodes" / de.episode_file(k, 10 + k), rec,
                     de.demo_meta(task_id, "t", 10 + k, k, len(rec)))


def test_split_holds_out_every_fifth_demo(tmp_path):
    import self_filtered_bc as sfbc

    save_demos(tmp_path, 12)
    train, val, metas = sfbc.split_episodes(tmp_path, "success", set(), seed=0, val_demo_every=5)
    assert [load_episode(p)["meta"]["demo_index"] for p in val] == [4, 9]
    assert len(train) == 10 and not {p.name for p in train} & {p.name for p in val}
    train, val, _ = sfbc.split_episodes(tmp_path, "success", set(), seed=0, val_demo_every=1)  # all held out
    assert train == [] and len(val) == 12
    train, val, _ = sfbc.split_episodes(tmp_path, "success", set(), seed=0)              # default: no hold-out
    assert len(train) == 12 and val == []


def test_is_validation_rollouts_unchanged():
    import self_filtered_bc as sfbc

    assert sfbc.is_validation({"init_state": 45}, {45, 46}, val_demo_every=5)
    assert not sfbc.is_validation({"init_state": 44}, {45, 46}, val_demo_every=5)
    demo = {"source": "demonstration", "demo_index": 4, "init_state": None}
    assert sfbc.is_validation(demo, set(), 5) and not sfbc.is_validation(demo, set(), 0)
    assert not sfbc.is_validation({**demo, "demo_index": 5}, set(), 5)


def test_check_episode_metas_demos_skip_policy_path_only():
    import self_filtered_bc as sfbc

    args = argparse.Namespace(policy_path="lerobot/smolvla_libero", task_id=0)
    sfbc.check_episode_metas([de.demo_meta(0, "t", 8, 0, 3)], args)
    with pytest.raises(SystemExit, match="task_id"):
        sfbc.check_episode_metas([de.demo_meta(1, "t", 8, 0, 3)], args)
    with pytest.raises(SystemExit, match="policy_path"):
        sfbc.check_episode_metas([{"task_id": 0, "policy_path": "other", "init_state": 10}], args)
    sfbc.check_episode_metas([{"task_id": 0, "policy_path": "lerobot/smolvla_libero", "init_state": 10}], args)


def test_parse_args_default_keeps_task6_behaviour():
    import self_filtered_bc as sfbc

    assert sfbc.parse_args(["--stage", "train"]).val_demo_every == 0


def test_episode_groups_are_consecutive_runs():
    fr = frames(2, episode=8) + frames(3, episode=13) + frames(1, episode=8)
    groups = [(e, len(f)) for e, f in de.episode_groups(iter(fr))]
    assert groups == [(8, 2), (13, 3), (8, 1)]          # a non-contiguous episode shows up twice
    assert list(de.episode_groups(iter([]))) == []
