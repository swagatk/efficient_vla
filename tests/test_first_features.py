"""Step 17's feature recorder (noise_reachability/first_features.py): it must not change the episode."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noise_reachability"))
import first_features as ff  # noqa: E402
import noise_steering  # noqa: E402


def fake_policy():
    """A policy whose model already has an instance-level sampler wrapper (as deep_noise_search installs one)."""
    model = SimpleNamespace()
    calls = []

    def inner(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        calls.append((images, img_masks, lang_tokens, lang_masks, state, noise, kwargs))
        return state * 2.0 + 1.0

    model.sample_actions = inner
    return SimpleNamespace(model=model), calls


def test_recorder_passes_every_call_through_and_records_the_first_only(monkeypatch):
    policy, calls = fake_policy()
    seen = []

    def fake_prefix(model, images, img_masks, lang_tokens, lang_masks, state, state_dim):
        seen.append((model, state_dim))
        return torch.cat([state.float(), torch.full((1, 3), float(len(seen)))], dim=-1)

    monkeypatch.setattr(noise_steering, "prefix_features", fake_prefix)
    store = {}
    ff.install_first_features(policy, store, state_dim=4)
    s1, s2 = torch.tensor([[0.5, -1.0]]), torch.tensor([[2.0, 3.0]])
    out1 = policy.model.sample_actions("im", "mask", "tok", "lmask", s1, extra=7)
    out2 = policy.model.sample_actions("im2", "mask2", "tok2", "lmask2", s2)
    assert torch.equal(out1, s1 * 2.0 + 1.0) and torch.equal(out2, s2 * 2.0 + 1.0)   # the inner wrapper's output, unchanged
    assert calls[0] == ("im", "mask", "tok", "lmask", s1, None, {"extra": 7})            # arguments handed on as given
    assert calls[1][0] == "im2" and len(calls) == 2
    assert seen == [(policy.model, 4)]                                                  # one prefix pass, first call only
    assert store["calls"] == 2 and store["features"].dtype == np.float32
    assert np.array_equal(store["features"], np.array([0.5, -1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_recorder_needs_a_noise_wrapper_and_installs_once():
    with pytest.raises(RuntimeError, match="noise wrapper first"):
        ff.install_first_features(SimpleNamespace(model=SimpleNamespace()), {})
    policy, _ = fake_policy()
    ff.install_first_features(policy, {})
    with pytest.raises(RuntimeError, match="already installed"):
        ff.install_first_features(policy, {})


def test_recorder_refuses_a_batch(monkeypatch):
    policy, _ = fake_policy()
    monkeypatch.setattr(noise_steering, "prefix_features", lambda *a: torch.zeros(2, 5))
    ff.install_first_features(policy, {})
    with pytest.raises(ValueError, match="one observation"):
        policy.model.sample_actions(None, None, None, None, torch.zeros(2, 2))


def test_features_round_trip_exactly(tmp_path):
    rng = np.random.default_rng(0)
    f = rng.normal(size=968).astype(np.float32) * np.float32(1e3)
    path = ff.feature_file(tmp_path / "feat", 7, 10000)
    ff.write_features(path, {"features": f, "calls": 52}, {"task_id": 4, "init_state": 7, "noise_id": 10000})
    meta, back = ff.read_features(path)
    assert np.array_equal(np.array(back, dtype=np.float32), f)                          # bit-identical after JSON
    assert meta == {"task_id": 4, "init_state": 7, "noise_id": 10000, "dim": 968, "dtype": "float32", "inferences": 52}
    assert path.name == "features_state7_id10000.json" and not Path(str(path) + ".tmp").exists()
    with pytest.raises(ValueError, match="no features"):
        ff.write_features(tmp_path / "x.json", {}, {})


def test_read_refuses_a_truncated_record(tmp_path):
    path = tmp_path / "f.json"
    ff.write_features(path, {"features": np.ones(4, dtype=np.float32), "calls": 1}, {})
    text = path.read_text().replace('"dim": 4', '"dim": 5')
    path.write_text(text)
    with pytest.raises(ValueError, match="dim 5"):
        ff.read_features(path)


def test_two_records_go_into_one_folder(tmp_path):
    d = tmp_path / "task_0" / "features" / "json"                       # parents created too
    for k in (10000, 10001):
        ff.write_features(ff.feature_file(d, 0, k), {"features": np.ones(3, dtype=np.float32), "calls": 1}, {"noise_id": k})
    assert sorted(p.name for p in d.iterdir()) == ["features_state0_id10000.json", "features_state0_id10001.json"]


def test_recorder_on_the_real_fixed_noise_wrapper_and_its_uninstall(monkeypatch):
    import deep_noise_search as dns

    class Model:
        config = SimpleNamespace(chunk_size=4, max_action_dim=3)

        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
            return noise + state.sum()

    model = Model()
    policy = SimpleNamespace(model=model)
    monkeypatch.setattr(noise_steering, "prefix_features", lambda m, *a: torch.ones(1, 2))
    fixed = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    outs = []
    for episode in range(2):                                       # as deep_noise_search's loop: uninstall, install, record
        dns.uninstall_fixed_noise(policy)
        dns.install_fixed_noise(policy, fixed + episode)
        store = {}
        ff.install_first_features(policy, store)
        state = torch.tensor([[1.0, 2.0]])
        outs.append(model.sample_actions(None, None, None, None, state))
        model.sample_actions(None, None, None, None, state)
        assert store["calls"] == 2 and np.array_equal(store["features"], np.ones(2, dtype=np.float32))
    assert torch.equal(outs[0], fixed + 3.0) and torch.equal(outs[1], fixed + 1 + 3.0)   # each episode its own noise
    dns.uninstall_fixed_noise(policy)
    assert "sample_actions" not in vars(model)                    # recorder and fixed-noise wrapper both gone
