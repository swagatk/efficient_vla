import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import noise_awr as na  # noqa: E402
import noise_steering  # noqa: E402
import train_noise_awr as tna  # noqa: E402
from vla_core.evaluation import episode_seed  # noqa: E402

FEATS = 12


def test_defaults_are_the_protocol_of_the_readme():
    c = na.NoiseAWRConfig(feature_dim=968)
    assert (c.noise_dim, c.hidden, c.layers, c.value_bins, c.batch_size) == (32, 512, 4, 16, 2048)
    assert (c.actor_lr, c.value_lr, c.betas, c.weight_decay, c.grad_clip) == (1e-5, 2e-5, (0.9, 0.95), 0.01, 1.0)
    assert (c.beta, c.max_weight, c.gamma, c.success_fraction) == (0.5, 20.0, 0.99, 0.5)
    assert na.LOG_STD_RANGE == (-20.0, 2.0)
    a = tna.parse_args(["--mode", "collect", "--output_dir", "x"])
    assert (a.task_id, a.n_action_steps, a.max_steps, a.train_init_states) == (0, 10, 520, "10-49")
    assert (a.parts, a.updates_per_epoch, a.seeds, a.init_states) == (4, 100, "16-31", "0-9")
    assert a.policy_path == "lerobot/smolvla_libero"


def test_run_updates_runs_joint_and_value_only_updates():
    learner = SimpleNamespace(cfg=na.NoiseAWRConfig(feature_dim=FEATS))
    calls = []
    learner.update = lambda b, update_actor: calls.append(update_actor) or {}
    replay = SimpleNamespace(sample=lambda *a: {})
    assert len(tna.run_updates(learner, replay, 3, "cpu")) == 3
    assert calls == [True, False] * 3


def test_every_state_runs_once_per_epoch_with_distinct_seeds():
    states = list(range(10, 50))
    jobs = [j for e in range(37) for k in range(4) for j in tna.part_jobs(states, e, k, 4)]
    assert len(jobs) == 1480 and len({j[1] for j in jobs}) == 1480
    assert all(sum(1 for j in tna.part_jobs(states, 0, k, 4)) == 10 for k in range(4))
    assert sorted(j[0] for k in range(4) for j in tna.part_jobs(states, 2, k, 4)) == states
    assert tna.part_jobs(states, 1, 1, 4)[0] == (11, 41)
    with pytest.raises(ValueError):
        tna.part_jobs(states, 0, 4, 4)


def test_training_seeds_never_meet_evaluation_or_other_training_seeds():
    ev = {episode_seed(s, i) for s in range(32) for i in range(10)}
    tr = {episode_seed(tna.train_episode_seed(t, e), 10 + e % 40) for t in (0, 1, 2) for e in range(1480)}
    assert len(tr) == 3 * 1480 and not tr & ev
    assert (tna.train_episode_seed(0, 0), tna.train_episode_seed(2, 37)) == (200_000, 220_037)
    with pytest.raises(ValueError):
        tna.train_episode_seed(0, 10_000)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(chunk_size=6, max_action_dim=5, use_cache=True)
        self.received = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        self.received.append(noise)
        return noise


def steered(monkeypatch, choose):
    monkeypatch.setattr(noise_steering, "prefix_features", lambda m, i, im, t, lm, state, d: state[:, :d].float())
    cfg = SimpleNamespace(n_action_steps=2, chunk_size=6, action_feature=SimpleNamespace(shape=(3,)))
    policy = SimpleNamespace(model=FakeModel(), config=cfg)
    st = noise_steering.install_repeated(policy, choose)
    return policy, st


def call(policy, n=1):
    return policy.model.sample_actions([torch.rand(n, 3, 4, 4)], [None], None, None, torch.rand(n, 8, dtype=torch.float16))


def test_chosen_vector_is_repeated_along_the_chunk(monkeypatch):
    v = torch.arange(5.0)
    policy, st = steered(monkeypatch, lambda f: v)
    z = call(policy)
    assert z.shape == (1, 6, 5) and z.dtype == torch.float32
    assert all(torch.equal(z[0, k], v) for k in range(6)) and st.n_inferences == 1


def test_none_chosen_leaves_the_sampling_to_the_base(monkeypatch):
    policy, _ = steered(monkeypatch, lambda f: None)
    assert call(policy) is None


def test_wrapper_refuses_batches_wrong_shapes_and_double_install(monkeypatch):
    policy, _ = steered(monkeypatch, lambda f: torch.zeros(4))
    with pytest.raises(ValueError):
        call(policy)
    with pytest.raises(ValueError):
        call(policy, n=2)
    with pytest.raises(ValueError):
        policy.model.sample_actions(None, None, None, None, torch.rand(1, 8), noise=torch.zeros(1))
    with pytest.raises(RuntimeError):
        noise_steering.install_repeated(policy, lambda f: None)
    noise_steering.uninstall(policy)
    assert "sample_actions" not in vars(policy.model)


def args_for(tmp_path, epoch, part=0):
    return tna.parse_args(["--mode", "update", "--output_dir", str(tmp_path), "--epoch", str(epoch), "--part", str(part),
                           "--train_init_states", "10-13", "--parts", "2", "--updates_per_epoch", "1"])


def write_part(tmp_path, epoch, part, states, decisions=2):
    folder = tmp_path / "collect" / f"e{epoch:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    n = decisions * len(states)
    tna.write_npz(folder / f"part{part}.npz", rows=np.array([[s, 200_000 + s, 1, 50, decisions] for s in states]),
                  feats=np.zeros((n, FEATS), np.float16), z=np.zeros((n, 32), np.float32),
                  ret=np.ones(n, np.float32), success=np.ones(n, np.float32))


def test_merge_checks_the_episodes_of_the_epoch(tmp_path):
    write_part(tmp_path, 0, 0, [10, 12])
    write_part(tmp_path, 0, 1, [11, 13])
    path = tna.merge_epoch(tmp_path, 0, args_for(tmp_path, 0), [10, 11, 12, 13])
    with np.load(path) as z:
        assert len(z["z"]) == 8 and sorted(z["rows"][:, 0].tolist()) == [10, 11, 12, 13]
    write_part(tmp_path, 1, 0, [10, 12])
    write_part(tmp_path, 1, 1, [11, 11])
    with pytest.raises(SystemExit):
        tna.merge_epoch(tmp_path, 1, args_for(tmp_path, 1), [10, 11, 12, 13])


def test_update_trains_resumes_and_logs_across_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(tna, "provenance", lambda out, args: None)
    for e in (0, 1):
        write_part(tmp_path, e, 0, [10, 12])
        write_part(tmp_path, e, 1, [11, 13])
        assert tna.update(args_for(tmp_path, e)) == 0
    first = (tmp_path / "actor_e001.pt").read_bytes()
    assert tna.update(args_for(tmp_path, 1)) == 0
    assert (tmp_path / "actor_e001.pt").read_bytes() == first   # a finished epoch is not redone
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert len(rows) == 8 and {r["epoch"] for r in rows} == {"0", "1"}
    ups = list(csv.DictReader(open(tmp_path / "update_log.csv")))
    assert [int(u["replay_decisions"]) for u in ups] == [8, 16]
    changed = tna.parse_args(["--mode", "update", "--output_dir", str(tmp_path), "--epoch", "2", "--parts", "2",
                              "--train_init_states", "10-13", "--updates_per_epoch", "2"])
    write_part(tmp_path, 2, 0, [10, 12])
    write_part(tmp_path, 2, 1, [11, 13])
    with pytest.raises(SystemExit):                             # a changed setting refuses to resume
        tna.update(changed)


def test_a_crash_after_the_checkpoint_costs_no_episode_and_a_lost_replay_file_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(tna, "provenance", lambda out, args: None)
    for e in (0, 1, 2):
        write_part(tmp_path, e, 0, [10, 12])
        write_part(tmp_path, e, 1, [11, 13])
    tna.update(args_for(tmp_path, 0))
    tna.update(args_for(tmp_path, 1))
    (tmp_path / "train_log.csv").write_text("epoch,init_state,seed,success,steps,decisions\n")   # logs lost
    (tmp_path / "replay" / "e001.npz").unlink()                                                  # upload lost
    tna.update(args_for(tmp_path, 2))
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert [int(r["epoch"]) for r in rows] == [0] * 4 + [1] * 4 + [2] * 4
    assert (tmp_path / "replay" / "e001.npz").exists()


def test_features_that_overflow_float16_are_refused():
    with pytest.raises(ValueError):
        tna.episode_arrays([np.full(FEATS, 7e4)], [np.zeros(32)], True, 0.5)
    with pytest.raises(ValueError):                     # one entry is enough
        tna.episode_arrays([np.eye(1, FEATS, 0).ravel() * 7e4], [np.zeros(32)], True, 0.5)


def test_episode_arrays_shapes_and_returns():
    e = tna.episode_arrays([np.zeros(FEATS)] * 3, [np.zeros(32)] * 3, True, 0.5)
    assert e["feats"].dtype == np.float16 and e["z"].shape == (3, 32)
    assert e["ret"].tolist() == [0.25, 0.5, 1.0] and e["success"].tolist() == [1, 1, 1]


def test_log_fields_match_the_analysis():
    assert tna.LOG_FIELDS == ["epoch", "init_state", "seed", "success", "steps", "decisions"]
