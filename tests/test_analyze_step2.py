import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

from analyze_step2 import choose, load_policy, report_stage, select_stage, systematic_rate  # noqa: E402


def eps(successes, n=20):
    """Episodes (0, seed, init) with the first ``successes`` succeeding."""
    return {(0, s, i): int(s * 10 + i < successes) for s in range(2) for i in range(n // 2)}


def test_choose_prefers_most_successes_then_smaller_n_then_medoid():
    assert choose({"base": 5, "first_n8": 9, "medoid_n8": 6, "accel_n8": 7}) == ("accel_n8", True)
    assert choose({"base": 5, "medoid_n16": 7, "accel_n8": 7}) == ("accel_n8", True)
    assert choose({"base": 5, "medoid_n8": 7, "accel_n8": 7}) == ("medoid_n8", True)
    assert choose({"base": 5, "accel_n8": 7, "medoid_n8": 7}) == ("medoid_n8", True)   # not by dict order
    assert choose({"base": 7, "medoid_n8": 7}) == ("medoid_n8", False)   # a tie with base is not ahead


def test_select_stage_requires_the_same_episodes():
    s = select_stage({"base": eps(8), "first_n8": eps(8), "medoid_n8": eps(12), "accel_n8": eps(10)})
    assert s["chosen"] == "medoid_n8" and s["chosen_ahead_of_base"]
    assert s["comparisons"]["medoid_n8"]["episodes_b_only"] == 4
    short = eps(8)
    short.pop((0, 0, 0))
    with pytest.raises(ValueError):
        select_stage({"base": eps(8), "medoid_n8": short})


def test_report_stage_gain_and_systematic_states():
    base, chosen = eps(4), eps(16)                       # +12 of 20 episodes
    systematic = {(0, 9)}                                # init state 9: chosen solves it in seed 0 only
    r = report_stage(base, chosen, systematic, lerobot_eval={**eps(5), (0, 5, 0): 1})
    assert r["gain"] == pytest.approx(0.6) and r["hypothesis"]["gain_ok"]
    assert systematic_rate(chosen, systematic) == (1, 2)
    assert r["systematic"] == {"chosen": 1, "base": 0, "episodes": 2} and not r["hypothesis"]["systematic_ok"]
    assert r["harness_check"] == {"base_here": 4, "lerobot_eval": 5, "episodes": 20, "gap_ok": True}
    assert not report_stage(eps(0), eps(0), set(), lerobot_eval=eps(18))["harness_check"]["gap_ok"]  # 18 > 17
    assert report_stage(eps(0), eps(0), set(), lerobot_eval=eps(17))["harness_check"]["gap_ok"]
    assert report_stage(eps(10), eps(10), set(), lerobot_eval=eps(10))["harness_check"]["gap_ok"]  # a gap, not a sum


def test_report_stage_thresholds_at_their_edges():
    two_hundred = {(0, s, i): 0 for s in range(2) for i in range(100)}
    gain = {**two_hundred, **{(0, 0, i): 1 for i in range(20)}}          # exactly +10 points, p < 0.05
    assert report_stage(two_hundred, gain, set())["hypothesis"]["gain_ok"]
    assert not report_stage(two_hundred, {**two_hundred, **{(0, 0, i): 1 for i in range(19)}},
                            set())["hypothesis"]["gain_ok"]                  # 9.5 points
    ten = {(0, s, 0): int(s == 0) for s in range(10)}                       # 1 in 10 systematic episodes
    assert report_stage(dict.fromkeys(ten, 0), ten, {(0, 0)})["hypothesis"]["systematic_ok"]
    assert not report_stage(dict.fromkeys(ten, 0), {**ten, (0, 1, 0): 1}, {(0, 0)})["hypothesis"]["systematic_ok"]


def test_report_stage_refuses_unequal_episode_sets():
    short = eps(4)
    short.pop((0, 1, 9))
    with pytest.raises(ValueError):
        report_stage(eps(4), short, set())
    with pytest.raises(ValueError):
        report_stage(eps(4), eps(4), set(), lerobot_eval=short)


def test_load_policy_reads_per_task_files(tmp_path):
    for t in (0, 4):
        (tmp_path / f"task_{t}").mkdir()
        with open(tmp_path / f"task_{t}" / "per_episode_base.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success", "steps"])
            w.writeheader()
            w.writerow({"task_id": t, "seed": 0, "init_state": 1, "success": int(t == 4), "steps": 9})
    assert load_policy(tmp_path, [0, 4], "base") == {(0, 0, 1): 0, (4, 0, 1): 1}
