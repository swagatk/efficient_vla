"""deep_noise_search.py --record_dir: names and metadata of recorded episodes, and that self_filtered_bc reads them."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))
sys.path.insert(0, str(ROOT))

import deep_noise_search as dns  # noqa: E402
import rollout_data as rd  # noqa: E402
import self_filtered_bc as sfbc  # noqa: E402


def args(**change):
    a = SimpleNamespace(task_id=7, init_state=33, family="repeat", policy_path="/p/a24f", n_action_steps=10,
                        rename_map={"a": "b"}, lora_dir=None)
    a.__dict__.update(change)
    return a


def test_record_names_are_unique_per_noise_jitter_state_and_seed():
    names = {dns.record_name(f, k, j, s, e) for f in ("repeat", "full") for k in (1, 1459) for j in (None, 5000)
             for s in (8, 33) for e in (2459, 5000)}
    assert len(names) == 2 * 2 * 2 * 2 * 2
    assert dns.record_name("repeat", 1459, None, 33, 2459) == "repeat_n1459_init33_seed2459.npz"
    assert dns.record_name("full", 29, 5001, 6, 5001) == "full_n29_j5001_init06_seed5001.npz"


def test_record_meta_carries_the_noise_and_what_training_checks():
    m = dns.record_meta(args(), "put both pots", 1459, None, 2459, {"success": True, "steps": 307})
    assert m["noise_family"] == "repeat" and m["noise_id"] == 1459 and m["jitter_seed"] is None
    assert m["seed"] == 2459 and m["init_state"] == 33 and m["success"] is True and m["steps"] == 307
    assert not sfbc.is_demo(m)
    assert sfbc.is_validation(m, {33}) and not sfbc.is_validation(m, {38})
    sfbc.check_episode_metas([m], SimpleNamespace(policy_path="/p/a24f", task_id=7))


def recorded_episode(tmp_path, success, k):
    rec = rd.EpisodeRecorder(frame_stride=2)
    for t in range(6):
        obs = {key: torch.full((1, 3, 4, 4), t / 255.0) for key in rd.IMAGE_KEYS}
        obs[rd.STATE_KEY] = torch.zeros(1, 8)
        rec.add(obs, np.full(7, t, np.float32))
    meta = dns.record_meta(args(), "task", k, None, 1000 + k, {"success": success, "steps": 6})
    path = tmp_path / "episodes" / dns.record_name("repeat", k, None, 33, 1000 + k)
    rd.save_episode(path, rec, meta)
    return path


def test_recorded_episodes_are_selected_by_success(tmp_path):
    recorded_episode(tmp_path, True, 1459)
    recorded_episode(tmp_path, False, 1460)
    recorded_episode(tmp_path, True, 1467)
    train, val, metas = sfbc.split_episodes(tmp_path, "success", set(), seed=0)
    assert sorted(p.name for p in train) == ["repeat_n1459_init33_seed2459.npz", "repeat_n1467_init33_seed2467.npz"]
    assert val == []
    ep = rd.load_episode(train[0])
    assert ep["actions"].shape == (6, 7) and ep["image"].shape[0] == 3       # every step's action, every 2nd frame
    assert ep["meta"]["noise_id"] == 1459


def test_a_temporary_file_left_by_a_stopped_process_is_not_trained_on(tmp_path):
    recorded_episode(tmp_path, True, 1459)
    (tmp_path / "episodes" / "repeat_n1467_init33_seed2467.npz.tmp.npz").write_bytes(b"cut off")
    train, val, metas = sfbc.split_episodes(tmp_path, "success", set(), seed=0)
    assert [p.name for p in train] == ["repeat_n1459_init33_seed2459.npz"] and len(metas) == 1


def test_record_meta_names_the_adapter_that_explored():
    assert dns.record_meta(args(), "t", 1, None, 1001, {"success": False, "steps": 520})["lora_dir"] is None
    m = dns.record_meta(args(lora_dir="/in/step10/runs/N7_0/adapter"), "t", 7000, None, 8000, {"success": True, "steps": 9})
    assert m["lora_dir"] == "/in/step10/runs/N7_0/adapter"


def test_lora_dir_defaults_to_the_base_and_reaches_attach_lora(monkeypatch):
    a = dns.parse_args(["--family", "repeat", "--init_state", "38", "--noise_ids", "7000-7049", "--output_dir", "/tmp/x"])
    assert a.lora_dir is None
    calls = []
    monkeypatch.setattr(sfbc, "attach_lora", lambda base, lora_dir, args: calls.append((base, lora_dir, args)))
    dns.attach_adapter("BASE", a)                                    # no adapter: the base is untouched
    assert calls == []
    b = dns.parse_args(["--family", "repeat", "--init_state", "38", "--noise_ids", "7000", "--output_dir", "/tmp/x",
                        "--lora_dir", "/in/runs/N7_0/adapter"])
    dns.attach_adapter("BASE", b)
    assert calls == [("BASE", "/in/runs/N7_0/adapter", b)]


def test_record_successes_only_keeps_successes_and_defaults_to_everything():
    a = dns.parse_args(["--family", "repeat", "--init_state", "38", "--noise_ids", "7000", "--output_dir", "/tmp/x"])
    assert a.record_successes_only == 0
    assert dns.keep_recording(a, {"success": False}) and dns.keep_recording(a, {"success": True})
    a.record_successes_only = 1
    assert dns.keep_recording(a, {"success": True}) and not dns.keep_recording(a, {"success": False})
