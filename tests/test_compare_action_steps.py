import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

from compare_action_steps import clopper_pearson, compare_setting, decide, paired_difference_ci  # noqa: E402


def episodes(fn, tasks=(0, 4), seeds=3, states=10):
    return {(t, s, i): int(fn(t, s, i)) for t in tasks for s in range(seeds) for i in range(states)}


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_paired_difference_ci_reproduces_continued_training_interval():
    # week6/continued_training_kaggle/README.md: 26 base-only, 29 continued-only of 300, -4.1 to +6.0 points.
    lo, hi = paired_difference_ci(26, 29, 300)
    assert round(100 * lo, 1) == -4.1 and round(100 * hi, 1) == 6.0


def test_baseline_is_restricted_to_the_run_episodes():
    base = episodes(lambda t, s, i: 0, tasks=range(10))
    run = episodes(lambda t, s, i: i < 5)
    r = compare_setting(base, run, alpha=0.025)
    assert r["n_episodes"] == 60 and r["a_successes"] == 0 and r["b_successes"] == 30
    assert r["qualifies"] and r["mcnemar_p"] < 1e-6


def test_run_episode_missing_from_baseline_is_an_error():
    with pytest.raises(ValueError):
        compare_setting(episodes(lambda t, s, i: 0, tasks=(0,)), episodes(lambda t, s, i: 0), alpha=0.025)


def test_behind_or_not_significant_does_not_qualify():
    base = episodes(lambda t, s, i: i < 3)
    assert not compare_setting(base, episodes(lambda t, s, i: 0), 0.025)["qualifies"]
    assert not compare_setting(base, episodes(lambda t, s, i: i < 4), 0.025)["qualifies"]  # +6 of 60, p > 0.025


def test_decide_picks_most_successes_and_ties_to_the_smaller_setting():
    q = lambda b, ok: {"b_successes": b, "qualifies": ok}  # noqa: E731
    assert decide({10: q(40, False), 50: q(50, False)}) == 1
    assert decide({10: q(40, True), 50: q(50, True)}) == 50
    assert decide({10: q(45, True), 50: q(45, True)}) == 10
    assert decide({10: q(40, False), 50: q(39, True)}) == 50
