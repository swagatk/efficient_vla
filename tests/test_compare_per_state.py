import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline_performance"))

from compare_per_state import compare, mcnemar_exact  # noqa: E402


def episodes(fn, tasks=2, seeds=3, states=10):
    return {(t, s, i): int(fn(t, s, i)) for t in range(tasks) for s in range(seeds) for i in range(states)}


def test_identical_runs_have_no_differences():
    a = episodes(lambda t, s, i: i % 2)
    r = compare(a, dict(a), n_perm=200)
    assert r["states_one_run_only"] == 0 and r["state_absdiff"] == 0 and r["mcnemar_p"] == 1.0
    assert r["per_task"][0] == {"a": 15, "b": 15, "n": 30, "only_a": [], "only_b": []}


def test_state_dependent_difference_is_detected():
    # A solves even states always, B solves odd states always: same overall rate, opposite states.
    a = episodes(lambda t, s, i: i % 2 == 0)
    b = episodes(lambda t, s, i: i % 2 == 1)
    r = compare(a, b, n_perm=500)
    assert r["a_successes"] == r["b_successes"]
    assert r["states_one_run_only"] == 20 and r["perm_p_one_run_only"] < 0.01
    assert r["per_task"][1]["only_a"] == [0, 2, 4, 6, 8]


def test_mismatched_episodes_rejected():
    with pytest.raises(ValueError):
        compare(episodes(lambda *_: 1), episodes(lambda *_: 1, states=9))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
