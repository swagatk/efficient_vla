"""CPU tests for vla_core's environment helpers, residual application and episode loop (no simulator)."""
import numpy as np
import pytest

from vla_core.envs import add_batch_dim, parse_index_ranges
from vla_core.evaluation import append_episode_row, episode_seed, load_episode_rows, run_episode
from vla_core.residual import apply_residual


def test_zero_residual_returns_base_action():
    base = np.array([0.9, -0.95, 0.3, 0.01, -0.02, 0.0, -1.0], dtype=np.float32)
    assert np.array_equal(apply_residual(base, np.zeros(7), 0.1), base)


def test_residual_is_bounded_and_clipped_only_at_env_limits():
    base = np.array([0.5, 0.98, -0.99, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    out = apply_residual(base, np.ones(7), 0.05)
    assert np.allclose(out, [0.55, 1.0, -0.94, 0.05, 0.05, 0.05, 1.0])
    assert np.all(np.abs(apply_residual(base, -np.ones(7), 0.05) - base) <= 0.05 + 1e-7)
    with pytest.raises(ValueError):
        apply_residual(base, 2 * np.ones(7), 0.05)


def test_parse_index_ranges():
    assert parse_index_ranges("0-9") == list(range(10))
    assert parse_index_ranges("10-12,3") == [10, 11, 12, 3]
    assert parse_index_ranges("0,1,2") == [0, 1, 2]
    for bad in ("3-1", "1,1"):
        with pytest.raises(ValueError):
            parse_index_ranges(bad)


def test_add_batch_dim_nested():
    obs = {"pixels": {"image": np.zeros((4, 4, 3))}, "robot_state": {"eef": {"pos": np.zeros(3)}}, "x": None}
    out = add_batch_dim(obs)
    assert out["pixels"]["image"].shape == (1, 4, 4, 3) and out["robot_state"]["eef"]["pos"].shape == (1, 3)
    assert out["x"] is None


def test_episode_seed_distinct_and_validated():
    seeds = {episode_seed(s, i) for s in range(3) for i in range(50)}
    assert len(seeds) == 150
    with pytest.raises(ValueError):
        episode_seed(0, 1000)


class FakeLiberoEnv:
    """Mimics LiberoEnv: applies init_state_id inside reset; succeeds after `succeed_at` steps."""

    def __init__(self, succeed_at=None):
        self.init_state_id, self.resets, self.succeed_at, self.t = 0, [], succeed_at, 0

    def reset(self, seed=None):
        self.resets.append((self.init_state_id, seed))
        self.t = 0
        return {"t": 0}, {}

    def step(self, action):
        self.t += 1
        success = self.succeed_at is not None and self.t >= self.succeed_at
        return {"t": self.t}, float(success), success, False, {"is_success": success}


def test_run_episode_resets_once_to_the_init_state_and_stops_on_success():
    env, calls = FakeLiberoEnv(succeed_at=4), []
    r = run_episode(env, 7, 2, lambda obs: np.zeros(7), max_steps=10,
                    on_step=lambda *a: calls.append(a[4]), on_reset=lambda obs: calls.append("reset"))
    assert env.resets == [(7, episode_seed(2, 7))]
    assert r == {"success": True, "steps": 4}
    assert calls == ["reset", False, False, False, True]


def test_run_episode_time_limit():
    r = run_episode(FakeLiberoEnv(), 0, 0, lambda obs: np.zeros(7), max_steps=5)
    assert r == {"success": False, "steps": 5}


def test_episode_rows_round_trip(tmp_path):
    path = tmp_path / "per_episode_base.csv"
    assert load_episode_rows(path) == {}
    append_episode_row(path, {"task_id": 3, "seed": 1, "init_state": 4, "success": 1, "steps": 200})
    append_episode_row(path, {"task_id": 3, "seed": 1, "init_state": 5, "success": 0, "steps": 520})
    rows = load_episode_rows(path)
    assert set(rows) == {(1, 4), (1, 5)} and rows[(1, 4)]["success"] == "1"
