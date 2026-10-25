import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

from reachability import parse_range, pass_at_k, pool, repeat_stats, summarize  # noqa: E402


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success"])
        w.writeheader()
        w.writerows({"task_id": t, "seed": s, "init_state": i, "success": ok} for (t, s, i), ok in rows.items())


def test_parse_range():
    assert parse_range("0-3") == [0, 1, 2, 3] and parse_range("0,5-6") == [0, 5, 6]


def test_pass_at_k_estimator():
    assert pass_at_k(16, 0, 16) == 0.0 and pass_at_k(16, 1, 16) == 1.0
    assert pass_at_k(16, 4, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)  # 1 - C(3,2)/C(4,2)


def test_pool_merges_files_and_rejects_gaps_and_duplicates(tmp_path):
    a = {(0, s, i): int(i == 0) for s in range(2) for i in range(3)}
    b = {(0, s, i): int(i == 0 and s == 2) for s in range(2, 4) for i in range(3)}
    write(tmp_path / "a.csv", a)
    write(tmp_path / "b.csv", b)
    per_state = pool([tmp_path / "a.csv", tmp_path / "b.csv"], [0], [0, 1, 2, 3])
    assert per_state[(0, 0)] == [1, 1, 1, 0] and per_state[(0, 2)] == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        pool([tmp_path / "a.csv"], [0], [0, 1, 2, 3])
    with pytest.raises(ValueError):
        pool([tmp_path / "a.csv", tmp_path / "a.csv"], [0], [0, 1])


def test_summary_headroom_and_systematic_states():
    # 4 states x 4 seeds: always, half, once, never.
    per_state = {(0, 0): [1, 1, 1, 1], (0, 1): [1, 0, 1, 0], (0, 2): [0, 0, 1, 0], (0, 3): [0, 0, 0, 0]}
    s = summarize(per_state)
    assert s["pass_at_k"][0] == pytest.approx(7 / 16)
    assert s["pass_at_k"][-1] == pytest.approx(3 / 4)
    assert s["headroom"] == pytest.approx((3 / 4 - 7 / 16) / (1 - 7 / 16))
    assert s["systematic_states"] == [[0, 3]] and not s["hypothesis"]["systematic_ok"]


def test_repeat_stats_counts_changes_against_fresh_seed_expectation():
    per_state = {(0, 0): [1, 0], (0, 1): [1, 1]}
    original = {(0, 0, 0): 1, (0, 0, 1): 1}
    repeat = {(0, 0, 0): 0, (0, 0, 1): 1}
    r = repeat_stats(original, repeat, per_state, seed=0)
    assert r["changed"] == 1 and r["expected_changed_fresh_seed"] == pytest.approx(0.5) and r["ratio"] == 2.0
    with pytest.raises(ValueError):
        repeat_stats(original, {}, per_state, seed=0)
