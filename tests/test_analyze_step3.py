import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import analyze_step3 as a3  # noqa: E402

SEEDS, STATES = [0, 1, 2], list(range(10))
SYSTEMATIC = {(0, 2), (0, 5), (0, 6), (0, 7), (0, 8), (4, 2)}


def eps(task, wins):
    """Episodes of one task; ``wins`` is the set of (seed, init_state) that succeed."""
    return {(task, s, i): int((s, i) in wins) for s in SEEDS for i in STATES}


def test_known_answer_gain_and_mcnemar():
    base = eps(7, {(s, i) for s in SEEDS for i in range(5)})                 # 15/30
    actor = eps(7, {(s, i) for s in SEEDS for i in range(9)} - {(0, 0)})      # 26/30, loses 1, wins 12
    r = a3.task_result(base, actor, 7, SEEDS, STATES, SYSTEMATIC)
    c = r["comparison"]
    assert (c["a_successes"], c["b_successes"], c["episodes_a_only"], c["episodes_b_only"]) == (15, 26, 1, 12)
    assert r["gain"] == pytest.approx(11 / 30) and c["mcnemar_p"] == pytest.approx(2 * 14 / 2 ** 13)
    assert r["gain_ok"] and "systematic" not in r                            # task 7 has no systematic state


def test_gain_threshold_edges():
    base = eps(7, set())
    # +3/30 = exactly 10 points, all discordant one way: p = 2/8 = 0.25 -> not significant
    r = a3.task_result(base, eps(7, {(0, 0), (0, 1), (0, 2)}), 7, SEEDS, STATES, SYSTEMATIC)
    assert r["gain"] == pytest.approx(0.1) and not r["gain_ok"]
    # 10 wins: p = 2/1024 < 0.05, gain 33 points
    r = a3.task_result(base, eps(7, {(s, i) for s in (0,) for i in STATES}), 7, SEEDS, STATES, SYSTEMATIC)
    assert r["gain_ok"]
    # equal gain but losses too: base wins 6, actor wins 9 -> +10 points, p large
    b = eps(7, {(0, i) for i in range(6)})
    a = eps(7, {(1, i) for i in range(9)})
    r = a3.task_result(b, a, 7, SEEDS, STATES, SYSTEMATIC)
    assert r["gain"] == pytest.approx(0.1) and not r["gain_ok"]
    # exactly +10 points and significant needs more episodes: 100, all 10 discordant to the actor
    seeds10 = list(range(10))
    b10 = {(7, s, i): 0 for s in seeds10 for i in STATES}
    a10 = dict(b10) | {(7, 0, i): 1 for i in STATES}
    r = a3.task_result(b10, a10, 7, seeds10, STATES, SYSTEMATIC)
    assert r["gain"] == pytest.approx(0.1) and r["comparison"]["mcnemar_p"] < 0.05 and r["gain_ok"]


def test_systematic_split_and_strict_third():
    actor = eps(0, {(s, i) for s in SEEDS for i in (0, 1, 3)} | {(0, 2)})     # reachable 9/15, systematic 1/15
    r = a3.task_result(eps(0, set()), actor, 0, SEEDS, STATES, SYSTEMATIC)
    s = r["systematic"]
    assert s["actor"] == [1, 15] and s["actor_reachable"] == [9, 15] and s["ok"]       # 1/15 < 3/15
    assert a3.systematic_ok(3, 15, 9, 15) is False                                     # exactly a third: no
    assert a3.systematic_ok(2, 15, 9, 15) is True
    assert a3.systematic_ok(0, 15, 0, 15) is False                                     # nothing solved anywhere


def test_incomplete_or_extra_episodes_are_refused():
    full = eps(7, set())
    short = dict(full)
    short.pop((7, 2, 9))
    with pytest.raises(ValueError):
        a3.task_result(full, short, 7, SEEDS, STATES, SYSTEMATIC)
    with pytest.raises(ValueError):
        a3.task_result(full | {(7, 3, 0): 1}, full, 7, SEEDS, STATES, SYSTEMATIC)


def test_train_summary_windows_and_gaps():
    rows = [{"episode": e, "success": int(e >= 70)} for e in range(250)]
    t = a3.train_summary(rows, warmup=20, window=100)
    assert t["episodes"] == 250 and t["warmup"] == [0, 20] and t["first_after_warmup"] == [50, 100]
    assert t["last"] == [100, 100]
    with pytest.raises(ValueError):
        a3.train_summary(rows[:10] + rows[11:], warmup=20)


def write_task(root, task, base, actor, log_rows=None):
    d = root / f"task_{task}" / "eval"
    d.mkdir(parents=True)
    for name, e in (("base", base), ("actor", actor)):
        with open(d / f"per_episode_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success", "steps"])
            w.writeheader()
            w.writerows({"task_id": k[0], "seed": k[1], "init_state": k[2], "success": v, "steps": 1} for k, v in e.items())
    if log_rows:
        with open(root / f"task_{task}" / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["episode", "success"])
            w.writeheader()
            w.writerows(log_rows)


def test_main_end_to_end(tmp_path, capsys):
    write_task(tmp_path, 7, eps(7, set()), eps(7, {(0, i) for i in STATES}), [{"episode": e, "success": int(e >= 25)} for e in range(30)])
    write_task(tmp_path, 0, eps(0, set()), eps(0, {(0, 0)}))
    step1 = tmp_path / "s1.json"
    step1.write_text(json.dumps({"systematic_states": sorted(SYSTEMATIC)}))
    a3.main(["--root", str(tmp_path), "--step1", str(step1), "--out", str(tmp_path / "r.txt")])   # default tasks 7 0
    text = (tmp_path / "r.txt").read_text()
    assert "7 | 0/30 | 10/30 | 0 | 10 | 0.0020 | +33.3 (+12.8 to +33.3) | yes" in text
    assert "task 0 systematic states: actor 0/15, reachable 1/15 (base 0/15 and 0/15); systematic below a third of reachable: yes" in text
    assert ("task 7 training (30 episodes, stochastic actor, init states 10-49): warm-up 0/20, first 100 after warm-up "
            "5/10, last 100 5/10") in text
    assert text.index("\n7 |") < text.index("\n0 |")
    assert json.loads((tmp_path / "r.txt.json").read_text())["0"]["gain_ok"] is False
