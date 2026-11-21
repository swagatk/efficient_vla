"""Step 16's tests T1 and T2: known answers, the permutation nulls, planted effects, Holm and refusals."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import step16_g1 as g  # noqa: E402


def test_flat_scores_leave_the_state_out():
    Y = np.array([[1, 0], [1, 1], [0, 0]])
    np.testing.assert_allclose(g.flat_scores(Y), [[0.5, 0.5], [0.5, 0.0], [1.0, 0.5]])


def test_neighbours_exclude_self_and_sort_by_distance():
    C = np.array([[0.0], [1.0], [3.0], [10.0]])
    np.testing.assert_array_equal(g.neighbours(C, 2), [[1, 2], [0, 2], [1, 0], [2, 1]])
    with pytest.raises(ValueError):
        g.neighbours(C, 4)
    with pytest.raises(ValueError):
        g.neighbours(C, 0)


def test_near_scores_average_the_neighbours():
    Y = np.array([[1, 0], [0, 1], [1, 1]])
    nb = np.array([[1, 2], [0, 2], [0, 1]])
    np.testing.assert_allclose(g.near_scores(Y, nb), [[0.5, 1.0], [1.0, 0.5], [0.5, 0.5]])


def test_auroc_known_answers_and_ties():
    Y = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
    s = np.array([[4, 3, 2, 1], [1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)
    a = g.auroc_rows(s, Y)
    assert a[0] == 1.0                    # perfect ranking
    assert a[1] == 0.5                    # all tied: one half
    assert np.isnan(a[2]) and np.isnan(a[3])   # not informative
    assert g.auroc_rows(np.array([[1.0, 4, 2, 3]]), np.array([[1, 1, 0, 0]]))[0] == 0.5   # 1 vs 2,3 and 4 vs 2,3


def test_midranks_share_tied_ranks():
    np.testing.assert_allclose(g.midranks(np.array([[3.0, 1, 3, 2]])), [[3.5, 1, 3.5, 2]])


def test_check_table_refuses_bad_input():
    with pytest.raises(ValueError):
        g.check_table(np.array([[0, 2]]))
    with pytest.raises(ValueError):
        g.check_table(np.array([[0, 1], [1, 0]]), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        g.check_table(np.array([[0, 1]]), np.array([[np.nan]]))


def test_t1_stat_known_answer():
    # every state ranks id 0 first: flat puts id 0 above id 1 for each state
    Y = np.array([[1, 0], [1, 0], [1, 0]])
    stat, n = g.t1_stat({0: (Y, np.zeros((3, 1)))})
    assert (stat, n) == (0.5, 3)
    with pytest.raises(ValueError):
        g.t1_stat({0: (np.zeros((3, 2), int), np.zeros((3, 1)))})


def test_t2_permuted_neighbours_equal_recomputed_ones():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(12, 3))
    base = g.neighbours(C, 4)
    perm = rng.permutation(12)
    inv = np.argsort(perm)
    direct = g.neighbours(C[perm], 4)
    np.testing.assert_array_equal(inv[base[perm]], direct)


def simulate(rng, S=30, K=48, tasks=4, sd_state=1.5, sd_id=0.0, sd_inter=0.0, d_noise=0):
    out = {}
    for t in range(tasks):
        z = rng.normal(size=(S, 2))
        v = rng.normal(size=(K, 2))
        logit = (rng.normal(0, sd_state, size=(S, 1)) + rng.normal(0, sd_id, size=(1, K))
                 + sd_inter * (z @ v.T) / np.sqrt(2))
        Y = (rng.random((S, K)) < 1 / (1 + np.exp(-logit))).astype(int)
        C = np.hstack([z, rng.normal(size=(S, d_noise))])
        out[t] = (Y, C)
    return out


def test_t1_detects_a_shared_id_effect_and_not_a_null():
    rng = np.random.default_rng(1)
    assert g.t1_test(simulate(rng, sd_id=1.5), n_perm=200)["p"] < 0.01
    assert g.t1_test(simulate(rng, sd_id=0.0), n_perm=200)["p"] > 0.05


def test_t2_detects_a_scene_dependent_effect_and_not_a_shared_one():
    rng = np.random.default_rng(2)
    assert g.t2_test(simulate(rng, sd_inter=2.0), n_perm=200)["p"] < 0.01
    # an id effect shared by all states is not scene-dependent: T2 must not pass
    assert g.t2_test(simulate(rng, sd_id=1.5), n_perm=200)["p"] > 0.05


def test_perm_p_counts_ties_and_adds_one():
    assert g.perm_p(1.0, np.array([0.0, 1.0, 2.0])) == 0.75
    assert g.perm_p(5.0, np.zeros(9)) == 0.1


def test_holm_steps_down():
    assert g.holm({"T1": 0.01, "T2": 0.04}) == {"T1": True, "T2": True}
    assert g.holm({"T1": 0.03, "T2": 0.04}) == {"T1": False, "T2": False}   # 0.03 >= 0.025
    assert g.holm({"T1": 0.02, "T2": 0.06}) == {"T1": True, "T2": False}
    assert g.holm({"T1": 0.025, "T2": 0.001}) == {"T1": True, "T2": True}   # 0.025 < 0.05 at the second step
    assert g.holm({"T1": 0.05, "T2": 0.001}) == {"T1": False, "T2": True}   # 0.05 is not < 0.05


def test_reading_follows_t2_then_t1():
    assert g.reading({"T1": False, "T2": True}).startswith("which noise suits")
    assert g.reading({"T1": True, "T2": False}).startswith("a noise's quality is shared")
    assert g.reading({"T1": False, "T2": False}).startswith("good noise is not shown")


def test_analyse_is_deterministic():
    rng = np.random.default_rng(3)
    tasks = simulate(rng, S=10, K=12, sd_id=1.0)
    assert g.M_NEAR == 8
    assert g.analyse(tasks, n_perm=50) == g.analyse(tasks, n_perm=50)


def test_per_task_reports_the_best_ticket_by_id():
    Y = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]] + [[0, 0, 0]] * 3)   # 9 states: m = 8
    r = g.per_task({4: (Y, np.arange(9.0)[:, None])}, ids={4: [10007, 10008, 10009]})["4"]
    assert (r["best_ticket"], r["best_ticket_successes"], r["successes"], r["informative_states"]) == (10008, 4, 8, 3)
    r0 = g.per_task({4: (np.zeros((9, 2), int), np.zeros((9, 1)))})["4"]
    assert r0["T1"] is None and r0["informative_states"] == 0 and r0["best_ticket"] == 0


# ---- independent reference implementations, for known answers ---------------------------------------------------

def auroc_ref(score, y):
    pos = [s for s, v in zip(score, y) if v]
    neg = [s for s, v in zip(score, y) if not v]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def stats_ref(Y, C, m):
    S, K = Y.shape
    t1, t2 = [], []
    for s in range(S):
        others = [j for j in range(S) if j != s]
        flat = [sum(Y[j, k] for j in others) / len(others) for k in range(K)]
        order = sorted(others, key=lambda j: (sum((C[s] - C[j]) ** 2), j))[:m]
        near = [sum(Y[j, k] for j in order) / m for k in range(K)]
        af, an = auroc_ref(flat, Y[s]), auroc_ref(near, Y[s])
        if af is not None:
            t1.append(af - 0.5)
            t2.append(an - af)
    return np.mean(t1), np.mean(t2), len(t1)


def test_t1_and_t2_match_the_reference_on_random_tables():
    rng = np.random.default_rng(7)
    for _ in range(5):
        Y = (rng.random((9, 7)) < 0.4).astype(int)
        C = rng.normal(size=(9, 3)) - 0.5          # negative coordinates too
        r1, r2, n = stats_ref(Y, C, 3)
        assert g.t1_stat({0: (Y, C)}) == pytest.approx((r1, n))
        assert g.t2_stat({0: (Y, C)}, m=3) == pytest.approx((r2, n))


def test_t2_stat_known_answer():
    # states 0, 1 and 2, 3 are neighbours; near ranks perfectly (1.0), flat ranks each state at 1/3
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    C = np.array([[0.0], [0.1], [5.0], [5.1]])
    assert g.t2_stat({0: (Y, C)}, m=1) == (1.0, 4)
    assert g.t1_stat({0: (Y, C)}) == (-0.5, 4)


def test_boundaries_of_m_and_states():
    C = np.array([[0.0], [-1.0], [2.0], [-4.0]])
    np.testing.assert_array_equal(g.neighbours(C, 1), [[1], [0], [0], [1]])
    assert g.neighbours(C, 3).shape == (4, 3)
    np.testing.assert_allclose(g.flat_scores(np.array([[1, 0], [0, 0]])), [[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        g.flat_scores(np.array([[1, 0]]))


def test_check_table_refuses_one_nan_and_a_row_mismatch():
    Y = np.array([[0, 1, 1], [1, 0, 0]])
    with pytest.raises(ValueError):
        g.check_table(Y, np.array([[0.0, 1.0], [np.nan, 2.0]]))
    with pytest.raises(ValueError):
        g.check_table(Y, np.zeros((3, 2)))
    g.check_table(Y, np.zeros((2, 3)))


def test_per_task_and_analyse_fields():
    rng = np.random.default_rng(5)
    Y = (rng.random((10, 6)) < 0.5).astype(int)
    C = rng.normal(size=(10, 2))
    r = g.per_task({0: (Y, C)})["0"]
    assert r["T1"] == g.t1_stat({0: (Y, C)})[0] and r["T2"] == g.t2_stat({0: (Y, C)})[0]
    assert r["states"] == 10 and r["episodes"] == 60
    full = g.analyse({0: (Y, C)}, n_perm=20)
    assert full["states"] == {"0": 10} and full["ids"] == {"0": 6} and full["n_perm"] == 20


def test_committed_contexts_file():
    c = json.loads((ROOT / "noise_reachability" / "step16_contexts.json").read_text())
    assert sorted(c) == ["0", "4", "6", "7"]
    for v in c.values():
        z = np.array(v["z"])
        assert z.shape == (50, len(v["columns"])) and v["shape"][0] == 50 and len(v["sha256"]) == 64
        np.testing.assert_allclose(z.mean(0), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(0), 1)


@pytest.mark.slow
def test_contexts_file_matches_libero(tmp_path):
    pytest.importorskip("libero")
    g.write_contexts(tmp_path / "c.json")
    new = json.loads((tmp_path / "c.json").read_text())
    old = json.loads((ROOT / "noise_reachability" / "step16_contexts.json").read_text())
    assert sorted(new) == sorted(old)
    for t in old:
        assert {k: new[t][k] for k in ("file", "sha256", "shape", "columns")} == {
            k: old[t][k] for k in ("file", "sha256", "shape", "columns")}
        np.testing.assert_allclose(new[t]["z"], old[t]["z"], rtol=1e-12, atol=1e-12)


# ---- the protocol's table ---------------------------------------------------------------------------------------

def write_table(path, states=range(10), ids=g.IDS, tasks=g.TASKS, success=lambda t, s, k: (s + k) % 3 == 0, drop=None):
    rows = ["task_id,init_state,noise_id,success,steps"]
    for t in tasks:
        for s in states:
            for k in ids:
                if (t, s, k) != drop:
                    rows.append(f"{t},{s},{k},{int(success(t, s, k))},520")
    path.write_text("\n".join(rows) + "\n")
    return path


def contexts(seed=0):
    rng = np.random.default_rng(seed)
    return {str(t): {"z": rng.normal(size=(50, 3)).tolist()} for t in g.TASKS}


def test_load_table_builds_the_protocol_tables(tmp_path):
    ctx = contexts()
    tasks, ids = g.load_table(write_table(tmp_path / "t.csv", states=[3, 1, 2, 0, 4, 5, 6, 7, 8]), ctx)
    assert sorted(tasks) == list(g.TASKS) and all(v == list(g.IDS) for v in ids.values())
    Y, C = tasks[6]
    assert Y.shape == (9, 50) and Y[1, 0] == int((1 + 10000) % 3 == 0)
    np.testing.assert_array_equal(C, np.array(ctx["6"]["z"])[:9])          # rows in state order 0-8


@pytest.mark.parametrize("kw, match", [
    ({"drop": (4, 2, 10017)}, "missing"),
    ({"ids": range(10000, 10030)}, "not the protocol"),
    ({"states": list(range(9)) + [30]}, "not the protocol"),
    ({"states": range(8)}, "not the protocol"),                       # too few states for 8 neighbours
    ({"tasks": (0, 4, 6)}, "tasks"),
])
def test_load_table_refuses_what_the_protocol_does_not_have(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        g.load_table(write_table(tmp_path / "t.csv", **kw), contexts())


def test_load_table_refuses_a_duplicate(tmp_path):
    p = write_table(tmp_path / "t.csv")
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[5]]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        g.load_table(p, contexts())


def test_main_writes_the_report_and_json(tmp_path, capsys):
    rng = np.random.default_rng(9)
    write_table(tmp_path / "t.csv", success=lambda t, s, k: rng.random() < 0.5)
    (tmp_path / "c.json").write_text(json.dumps(contexts(1)))
    out = tmp_path / "r.txt"
    assert g.main(["--table", str(tmp_path / "t.csv"), "--contexts", str(tmp_path / "c.json"), "--out", str(out),
                   "--n_perm", "30"]) == 0
    text = out.read_text()
    assert text == capsys.readouterr().out and "30 permutations" in text
    assert "T1: statistic" in text and ", m = 8, strata of 5)" in text and "task 4 (reported, not tested)" in text
    r = json.loads((tmp_path / "r.txt.json").read_text())
    assert r["ids"] == {str(t): 50 for t in g.TASKS} and r["T2"]["stratum_size"] == 5 and r["n_perm"] == 30


def test_task_contexts_keep_varying_object_columns_only():
    x = np.zeros((4, 13))
    x[:, 2] = [0.0, 0.1, 0.2, 0.3]        # a robot joint: dropped
    x[:, 10] = [1.0, 3.0, 5.0, 7.0]
    x[:, 12] = [2.0, 2.0, 4.0, 4.0]
    x[:, 11] = 5.0                        # constant: dropped
    c = g.task_contexts(x)
    assert c["columns"] == [10, 12]
    z = np.array(c["z"])
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(0), 1)


def test_committed_contexts_hold_object_columns_only():
    c = json.loads((ROOT / "noise_reachability" / "step16_contexts.json").read_text())
    assert all(min(v["columns"]) >= g.FIRST_OBJECT_COLUMN for v in c.values())
    assert {t: len(v["columns"]) for t, v in c.items()} == {"0": 16, "4": 10, "6": 8, "7": 10}


# ---- T2's stratified null ---------------------------------------------------------------------------------------

def test_strata_group_states_by_success_count():
    Y = np.zeros((12, 4), int)
    counts = [3, 0, 1, 4, 2, 0, 1, 3, 2, 4, 0, 1]
    for s, c in enumerate(counts):
        Y[s, :c] = 1
    groups = g.strata(Y, 5)
    assert [len(x) for x in groups] == [5, 7]                         # the remainder of 2 joins the last group
    assert sorted(np.concatenate(groups).tolist()) == list(range(12))
    assert max(counts[i] for i in groups[0]) <= min(counts[i] for i in groups[1])
    assert [len(x) for x in g.strata(np.zeros((10, 2), int), 5)] == [5, 5]


def test_stratified_permutation_stays_in_its_group():
    rng = np.random.default_rng(0)
    groups = [np.array([0, 3, 5]), np.array([1, 2, 4, 6])]
    for _ in range(20):
        perm = g.stratified_permutation(rng, groups, 7)
        assert sorted(perm.tolist()) == list(range(7))
        for grp in groups:
            assert set(perm[grp].tolist()) == set(grp.tolist())


def confounded(rng, conf=2.5, S=30, K=50):
    out = {}
    for t in range(4):
        z = rng.normal(size=(S, 2))
        logit = conf * z[:, :1] + rng.normal(0, 1.0, size=(1, K))       # difficulty from the scene, shared ids
        out[t] = ((rng.random((S, K)) < 1 / (1 + np.exp(-logit))).astype(int), z)
    return out


@pytest.mark.slow
def test_stratified_null_holds_its_level_under_the_confound_and_the_free_one_does_not():
    rng = np.random.default_rng(11)
    free = strat = 0
    for _ in range(20):
        d = confounded(rng)
        free += g.t2_test(d, n_perm=100, stratum_size=None)["p"] < 0.05
        strat += g.t2_test(d, n_perm=100)["p"] < 0.05
    assert free >= 6 and strat <= 3


def test_strata_edges():
    assert [len(x) for x in g.strata(np.zeros((7, 2), int), 5)] == [7]          # two groups merge into one
    assert [len(x) for x in g.strata(np.zeros((3, 2), int), 5)] == [3]          # one short group stays
    assert [len(x) for x in g.strata(np.zeros((22, 2), int), 5)] == [5, 5, 5, 7]
    assert [len(x) for x in g.strata(np.zeros((30, 2), int), 5)] == [5] * 6


def test_load_table_accepts_all_30_states_and_tasks_with_different_states(tmp_path):
    tasks, _ = g.load_table(write_table(tmp_path / "a.csv", states=range(30)), contexts())
    assert all(Y.shape == (30, 50) for Y, _ in tasks.values())
    rows = ["task_id,init_state,noise_id,success,steps"]
    for t in g.TASKS:
        for s in (range(9) if t == 6 else range(1, 12)):
            rows += [f"{t},{s},{k},{int((s + k) % 2 == 0)},520" for k in g.IDS]
    (tmp_path / "b.csv").write_text("\n".join(rows) + "\n")
    tasks, _ = g.load_table(tmp_path / "b.csv", contexts())
    assert tasks[6][0].shape == (9, 50) and tasks[4][0].shape == (11, 50)
