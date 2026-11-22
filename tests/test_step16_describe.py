"""step16_describe.py: the descriptive breakdown agrees with step16_g1's statistics and null."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "tests"))
import step16_describe as dsc  # noqa: E402
import step16_g1 as g  # noqa: E402
from test_step16_g1 import simulate  # noqa: E402


def test_per_task_auroc_known_answer():
    # one task, 3 states, 4 ids; flat and near (m = 2 = all other states) coincide
    Y = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
    C = np.array([[0.0], [1.0], [2.0]])
    r = dsc.per_task_auroc({0: (Y, C)}, m=2)["0"]
    # state 0: flat scores (1, .5, .5, 0); successes score {1, .5}, failures {.5, 0}: (1 + 1 + .5 + 1)/4 = 0.875
    # state 1: flat scores (1, 1, 0, 0); successes score {1, 0}, failures {1, 0}: (.5 + 1 + 0 + .5)/4 = 0.5
    # state 2: same as state 0: 0.875
    assert r["informative_states"] == 3
    assert np.isclose(r["flat"], (0.875 + 0.5 + 0.875) / 3)
    assert np.isclose(r["near"], r["flat"])


def test_per_task_means_reproduce_t2_statistic():
    tasks = simulate(np.random.default_rng(5), S=20, K=30, sd_id=1.0, sd_inter=1.0)
    pt = dsc.per_task_auroc(tasks)
    n = sum(v["informative_states"] for v in pt.values())
    pooled = sum((v["near"] - v["flat"]) * v["informative_states"] for v in pt.values()) / n
    t2, n2 = g.t2_stat(tasks)
    assert n == n2 and np.isclose(pooled, t2)


def test_null_is_the_reports_null():
    tasks = simulate(np.random.default_rng(6), S=20, K=30, sd_id=1.0, sd_inter=1.0)
    plain = g.t2_test(tasks, n_perm=100)
    full = g.t2_test(tasks, n_perm=100, return_null=True)
    null = full.pop("null")
    assert plain == full                     # the flag only adds the null
    assert len(null) == 100 and np.isclose(null.mean(), plain["null_mean"])
    assert np.isclose(g.perm_p(plain["statistic"], null), plain["p"])
    d = dsc.describe(tasks, n_perm=100)
    assert np.isclose(d["T2"]["p"], plain["p"]) and np.isclose(d["T2"]["null_sd"], null.std())
    assert np.isclose(d["T2"]["z"], (plain["statistic"] - null.mean()) / null.std())
    assert d["T2"]["null_max"] == null.max() and "null" not in d["T2"]


def test_uninformative_states_are_left_out_not_fatal():
    # state 3 never succeeds: the means are over states 0-2 only (same values as the known-answer test)
    Y = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    C = np.array([[0.0], [1.0], [2.0], [50.0]])
    r = dsc.per_task_auroc({0: (Y, C)}, m=3)["0"]
    assert r["informative_states"] == 3
    assert np.isclose(r["flat"], np.mean([g.auroc_rows(g.flat_scores(Y), Y)[s] for s in range(3)]))
    assert r["near"] is not None and np.isfinite(r["near"])
    none = dsc.per_task_auroc({0: (np.zeros((3, 4), int), C[:3])}, m=2)["0"]
    assert none == {"flat": None, "near": None, "informative_states": 0}
