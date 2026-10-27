import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import train_noise_sac as tns  # noqa: E402
from noise_sac import ChunkReplay, NoiseSAC, NoiseSACConfig, Transition  # noqa: E402
from vla_core.evaluation import episode_seed  # noqa: E402


def test_defaults_are_the_protocol_of_the_readme():
    a = tns.parse_args(["--mode", "train", "--task_id", "7", "--output_dir", "x"])
    assert (a.policy_path, a.n_action_steps, a.max_steps) == ("lerobot/smolvla_libero", 10, 520)
    assert (a.train_init_states, a.num_episodes, a.warmup_episodes, a.episodes_per_process) == ("10-49", 1500, 20, 25)
    assert (a.init_alpha, a.buffer_size) == (0.01, 200_000)
    assert (a.updates_per_decision, a.batch_size, a.noise_scale, a.init_std) == (8, 256, 2.5, 0.45)
    assert (a.lr, a.gamma, a.tau, a.hidden, a.layers, a.train_hard_reset) == (3e-4, 0.99, 0.005, 256, 3, 0)
    assert (a.policies, a.seeds, a.init_states) == ("base,actor", "0,1,2", "0-9")
    assert a.rename_map == {"observation.images.image": "observation.images.camera1",
                            "observation.images.image2": "observation.images.camera2"}


def test_training_seeds_are_distinct_and_never_an_evaluation_seed():
    eval_seeds = {episode_seed(s, i) for s in range(100) for i in range(50)}
    train = [episode_seed(tns.train_episode_seed(0, e), 10 + e % 40) for e in range(1500)]
    assert len(set(train)) == 1500 and not set(train) & eval_seeds
    assert tns.train_episode_seed(1, 0) != tns.train_episode_seed(0, 0)
    assert (tns.train_episode_seed(0, 0), tns.train_episode_seed(2, 37)) == (100_000, 120_037)
    with pytest.raises(ValueError):
        tns.train_episode_seed(0, 10_000)


def test_init_states_are_reproducible_per_episode_and_cover_the_training_set():
    states = list(range(10, 50))
    a = [tns.train_init_state(0, e, states) for e in range(400)]
    assert a == [tns.train_init_state(0, e, states) for e in range(400)]    # same after a resume
    assert set(a) == set(states) and a != [tns.train_init_state(1, e, states) for e in range(400)]


def test_a_changed_setting_refuses_to_resume():
    args = tns.parse_args(["--mode", "train", "--task_id", "7", "--output_dir", "x"])
    saved = tns.run_config(args)
    tns.check_same_run(saved, args)
    args.noise_scale = 2.0
    with pytest.raises(SystemExit):
        tns.check_same_run(saved, args)
    tns.check_same_run(saved, args, tns.BASE_KEYS)    # evaluation checks only the base policy's settings
    args.n_action_steps = 1
    with pytest.raises(SystemExit):
        tns.check_same_run(saved, args, tns.BASE_KEYS)


def test_log_keeps_only_episodes_before_the_resume_point(tmp_path):
    rows = [{k: "" for k in tns.LOG_FIELDS} | {"episode": e, "success": e % 2} for e in range(6)]
    tns.write_log(tmp_path / "log.csv", rows)
    assert [int(r["episode"]) for r in tns.read_log(tmp_path / "log.csv", 4)] == [0, 1, 2, 3]
    assert tns.read_log(tmp_path / "missing.csv", 4) == []


def test_checkpoint_lists_every_replay_part_and_no_part_is_removed(tmp_path):
    import torch

    args = tns.parse_args(["--mode", "train", "--task_id", "7", "--output_dir", str(tmp_path)])
    learner = NoiseSAC(NoiseSACConfig(feature_dim=4, action_dim=3, hidden=8, layers=1))
    rb = ChunkReplay(10, 4, 3)
    parts = []
    for start, stop, n_new in ((0, 25, 2), (25, 50, 3)):
        part_start = len(rb)
        for _ in range(n_new):
            rb.add(Transition(np.full(4, stop, np.float32), np.zeros(3, np.float32), 0.0, np.ones(4, np.float32), False))
        parts = tns.save_checkpoint(tmp_path, learner, rb, args, start, stop, parts, part_start)
    ck = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert ck["episodes_done"] == 50 and ck["replay_size"] == 5
    assert ck["replay_parts"] == parts == ["replay_part_00000-00024.npz", "replay_part_00025-00049.npz"]
    assert sorted(p.name for p in tmp_path.glob("replay_*.npz")) == parts
    assert ck["run_config"] == tns.run_config(args) and not list(tmp_path.glob("*.tmp*"))
    back = ChunkReplay(10, 4, 3)
    back.load_parts([tmp_path / p for p in parts])
    assert len(back) == 5 and back.feats[:, 0].tolist()[:5] == [25, 25, 50, 50, 50]


def test_buffer_too_small_for_the_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tns, "require_mujoco_patch", lambda: (_ for _ in ()).throw(SystemExit("past the check")))
    base = ["--mode", "train", "--task_id", "7", "--output_dir", str(tmp_path)]
    # 1,500 episodes x (520 // 10 + 1) decisions at most = 79,500 transitions
    with pytest.raises(SystemExit, match="buffer_size 79499 is below the 79500"):
        tns.main(base + ["--buffer_size", "79499"])
    with pytest.raises(SystemExit, match="past the check"):
        tns.main(base + ["--buffer_size", "79500"])


def test_log_fields_cover_every_logged_value():
    with open(Path(tns.__file__)) as f:
        src = f.read()
    for k in ("critic_loss", "q_mean", "actor_loss", "entropy", "alpha", "noise_abs_mean", "decisions"):
        assert k in tns.LOG_FIELDS and k in src
    assert csv  # the log is a csv.DictWriter with exactly LOG_FIELDS


# --- the training loop with a fake environment and policy ------------------------------------------------------
import torch  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import noise_steering  # noqa: E402

N_EXEC, ACT = 10, 7


class FakeEnv:
    """A step counter. Episodes from even init states succeed at step 25, odd ones run to the step limit."""

    task_description = "fake task"

    def __init__(self):
        self.actions = []

    def reset(self, seed=None):
        self.t = 0
        return {"t": 0}, {}

    def step(self, action):
        self.t += 1
        self.actions.append((self.init_state_id, self.t, np.array(action)))
        success = self.init_state_id % 2 == 0 and self.t == 25
        return {"t": self.t}, 0.0, success, False, {"is_success": success}

    def close(self):
        pass


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32, use_cache=True)
        self.noise, self.drawn = [], []

    def sample_noise(self, shape, device):
        self.drawn.append(torch.normal(0.0, 1.0, size=shape) * 1.5)   # wider than N(0, 1) so the clip is exercised
        return self.drawn[-1]

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        self.noise.append(noise.clone())
        return noise  # the executed actions are the noise itself


class FakeBase:
    """select_action's queue: one inference per N_EXEC steps; the state carries the step count."""

    def __init__(self):
        cfg = SimpleNamespace(n_action_steps=N_EXEC, chunk_size=50, action_feature=SimpleNamespace(shape=(ACT,)))
        self.policy = SimpleNamespace(model=FakeModel(), config=cfg)
        self.queue = []

    def reset(self):
        self.queue = []

    def act(self, obs, task):
        if not self.queue:
            state = torch.zeros(1, 32)
            state[0, 0] = obs["t"]
            chunk = self.policy.model.sample_actions([None], [None], None, None, state)
            self.queue = list(chunk[0, :N_EXEC, :ACT].numpy())
        return self.queue.pop(0), {}


def test_training_loop_stores_the_transitions_it_executed(tmp_path, monkeypatch):
    env, base = FakeEnv(), FakeBase()
    monkeypatch.setattr(tns, "require_mujoco_patch", lambda: None)
    monkeypatch.setattr(tns, "provenance", lambda out, args: None)
    monkeypatch.setattr(tns, "load_base", lambda args, hard_reset: (env, base, "cpu"))
    monkeypatch.setattr(tns, "feature_dim", lambda policy: 8)
    monkeypatch.setattr(noise_steering, "prefix_features", lambda m, i, im, t, lm, state, d: state[:, :d].clone())
    argv = ["--mode", "train", "--task_id", "7", "--output_dir", str(tmp_path), "--max_steps", "40",
            "--num_episodes", "3", "--warmup_episodes", "1", "--episodes_per_process", "2", "--batch_size", "3",
            "--updates_per_decision", "1", "--hidden", "16", "--layers", "1"]
    assert tns.main(argv) == tns.EXIT_MORE_EPISODES       # episodes 0-1, then a save
    assert tns.main(argv) == 0                            # resumes: episode 2
    assert tns.main(argv) == 0                            # nothing left

    ck = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert ck["episodes_done"] == 3 and ck["replay_parts"] == ["replay_part_00000-00001.npz", "replay_part_00002-00002.npz"]
    states = [tns.train_init_state(0, e, list(range(10, 50))) for e in range(3)]
    log = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert [int(r["init_state"]) for r in log] == states and [r["warmup"] for r in log] == ["1", "0", "0"]
    # A success at step 25 ends the third decision; a failure runs 4 decisions to step 40.
    want_decisions = [3 if s % 2 == 0 else 4 for s in states]
    assert [int(r["decisions"]) for r in log] == want_decisions
    assert [int(r["success"]) for r in log] == [int(s % 2 == 0) for s in states]
    # Warm-up trains only the critics; afterwards the actor too.
    assert log[0]["critic_loss"] != "" and log[0]["actor_loss"] == "" and log[0]["alpha"] != ""
    assert all(r["actor_loss"] != "" and r["entropy"] != "" for r in log[1:])

    rb = ChunkReplay(1000, 8, N_EXEC * ACT)
    rb.load_parts([tmp_path / p for p in ck["replay_parts"]])
    assert len(rb) == sum(want_decisions) == ck["replay_size"]
    # Executed noise per inference, with the bootstrap inference of each failed episode in between.
    noise = base.policy.model.noise
    k = 0
    for e, (s, n) in enumerate(zip(states, want_decisions)):
        ep = slice(sum(want_decisions[:e]), sum(want_decisions[:e + 1]))
        f, nf = rb.feats[ep, 0].astype(float), rb.next_feats[ep, 0].astype(float)
        assert f.tolist() == [10.0 * j for j in range(n)]                   # features of the decision's observation
        if s % 2 == 0:
            assert nf.tolist()[:-1] == [10.0 * j for j in range(1, n)] and rb.terminal[ep, 0].tolist() == [0] * (n - 1) + [1]
            assert rb.reward[ep, 0].tolist() == [0] * (n - 1) + [1]
        else:
            assert nf.tolist() == [10.0 * j for j in range(1, n)] + [40.0]  # bootstrap from the final observation
            assert not rb.terminal[ep].any() and not rb.reward[ep].any()
        for j in range(n):
            used = noise[k][0, :N_EXEC, :ACT]
            assert np.allclose(rb.u[ep][j] * 2.5, used.reshape(-1).numpy(), atol=1e-6)
            drawn = base.policy.model.drawn[k][0, :N_EXEC, :ACT]
            if e == 0:   # warm-up: the base's own draw, clipped to the actor's range
                assert torch.allclose(used, drawn.clamp(-2.5 * 0.999, 2.5 * 0.999), atol=1e-6)
                assert (drawn.abs() > 2.5).any()
            else:        # the stochastic actor: its own noise, not the draw
                assert not torch.allclose(used, drawn)
            k += 1
        k += 0 if s % 2 == 0 else 1                                        # skip the bootstrap inference
    assert k == len(noise)
    later = rb.u[want_decisions[0]:len(rb)]
    assert later.std() > 0.1 and (np.abs(later) < 1).all()                # exploration, inside the tanh range
    first = [a for (i, t, a) in env.actions if t == 1]
    assert np.allclose(first[0], noise[0][0, 0, :ACT].numpy())             # the env executed the chosen noise
