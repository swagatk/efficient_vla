"""heldout_headroom.py: the seed for a state is chosen without that state, ties average, and known answers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))
import heldout_headroom as hh  # noqa: E402


def test_choice_ignores_the_held_out_state():
    # seed 0 wins on states 1 and 2; state 0 succeeds only at seed 1, so the held-out choice scores 0 there
    per = {(0, 0): [0, 1, 0], (0, 1): [1, 0, 0], (0, 2): [1, 0, 0]}
    c = hh.leave_one_state_out(per)
    assert c[0, 0] == 0.0
    assert c[0, 1] == 0.5 and c[0, 2] == 0.5          # others then tie seeds 0 and 1 at 1; the state has 1 and 0


def test_ties_average_over_the_tied_seeds():
    per = {(0, 0): [1, 0, 0, 0], (0, 1): [1, 1, 0, 0]}
    c = hh.leave_one_state_out(per)
    assert c[0, 0] == 0.5                             # seeds 0 and 1 tie on state 1; state 0 has 1 and 0
    assert c[0, 1] == 1.0                             # only seed 0 wins on state 0


def test_choice_stays_within_task():
    per = {(0, 0): [1, 0], (0, 1): [1, 0], (4, 0): [0, 1], (4, 1): [0, 1]}
    c = hh.leave_one_state_out(per)
    assert all(v == 1.0 for v in c.values())


def test_a_task_with_one_state_is_refused():
    with pytest.raises(ValueError):
        hh.leave_one_state_out({(0, 0): [1, 0], (4, 0): [0, 1], (4, 1): [1, 0]})


def test_summary_known_answer():
    per = {(0, 0): [1, 0, 0, 0], (0, 1): [1, 0, 0, 0], (0, 2): [0, 0, 0, 0]}
    s = hh.summarize(per, perms=200, seed=1)
    assert s["pass@1"] == pytest.approx(2 / 12)
    assert s["pass@4"] == pytest.approx(2 / 3)
    assert s["held_out_choice"] == pytest.approx((1 + 1 + 0) / 3)
    assert s["hindsight_headroom"] == pytest.approx((2 / 3 - 1 / 6) / (5 / 6))
    assert s["held_out_headroom"] == pytest.approx((2 / 3 - 1 / 6) / (5 / 6))
    assert 0 <= s["null_p"] <= 1


def test_no_shared_seed_effect_gives_no_gain_on_average():
    # each state succeeds at a different seed: a seed chosen elsewhere never helps
    per = {(0, s): [int(j == s) for j in range(4)] for s in range(4)}
    s = hh.summarize(per, perms=50)
    assert s["held_out_choice"] == 0.0 and s["held_out_headroom"] < 0


def test_per_task_rates_and_null_counts_ties_as_extreme():
    # task 0 always succeeds; task 4 scores 0, the minimum, so every shuffle ties or beats it and p is 1
    per = {(0, 0): [1, 1], (0, 1): [1, 1], (4, 0): [1, 0], (4, 1): [0, 1]}
    s = hh.summarize(per, perms=20, seed=0)
    assert s["per_task"][0] == {"pass@1": 1.0, "pass@2": 1.0, "held_out_choice": 1.0}
    assert s["per_task"][4]["pass@1"] == 0.5 and s["per_task"][4]["held_out_choice"] == 0.0   # the other state points to the wrong seed
    assert s["null_p"] == 1.0
