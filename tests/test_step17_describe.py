"""step17_describe.py: near in feature space, the critic on contexts, and the pose probe (descriptive, not the rule)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "tests"))
import step16_g1 as g16  # noqa: E402
import step17_describe as dsc  # noqa: E402
import step17_g1b as g17  # noqa: E402
from test_step17_g1b import features_json, sim_task, table_csv  # noqa: E402


def sim_tasks(seed, S=20, K=50, extra=3):
    rng = np.random.default_rng(seed)
    return {t: sim_task(rng, S=S, K=K, extra=extra) for t in (0, 4)}


# ---- near ------------------------------------------------------------------------------------------------------

def test_near_on_features_is_invariant_to_column_scale_and_offset():
    tasks = sim_tasks(1)
    rng = np.random.default_rng(2)
    scaled = {t: (Y, X * rng.uniform(0.1, 10, X.shape[1]) + rng.normal(size=X.shape[1])) for t, (Y, X) in tasks.items()}
    a = dsc.near_row({t: (Y, dsc.zscore_columns(X)) for t, (Y, X) in tasks.items()}, n_perm=50)
    b = dsc.near_row({t: (Y, dsc.zscore_columns(X)) for t, (Y, X) in scaled.items()}, n_perm=50)
    assert np.isclose(a["U1"], b["U1"]) and a["D"] == b["D"] and a["p"] == b["p"]


def test_zscore_drops_constant_columns_and_refuses_all_constant():
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 2.0], [5.0, 5.0, 2.0]])
    Z = dsc.zscore_columns(X)
    assert Z.shape == (3, 1) and np.allclose(Z[:, 0], [-1.2247449, 0.0, 1.2247449])
    with pytest.raises(ValueError):
        dsc.zscore_columns(np.ones((4, 3)))


def test_near_row_equals_step16_t2_and_its_own_d():
    tasks = sim_tasks(3)
    r = dsc.near_row(tasks, n_perm=100, seed=17)
    t2 = g16.t2_test(tasks, n_perm=100, seed=17)
    assert np.isclose(r["U1"], t2["statistic"]) and r["p"] == t2["p"]
    d = np.concatenate([g17.top_rate(g16.near_scores(Y, g16.neighbours(C)), Y) - g17.top_rate(g16.flat_scores(Y), Y)
                        for Y, C in tasks.values()])
    assert np.isclose(r["D"], d.mean())


def test_near_contexts_reproduces_step16_t2_on_the_committed_table():
    """Known answer: step 16's T2 (README, step 16: 0.0028 over 115 informative states) on all 120 map states."""
    ctx = json.loads(dsc.CONTEXTS.read_text())
    cells = {}
    import csv
    for row in csv.DictReader(open(g17.STEP16_TABLE)):
        cells[(int(row["task_id"]), int(row["init_state"]), int(row["noise_id"]))] = int(row["success"])
    tasks = {t: np.array([[cells[(t, s, k)] for k in g16.IDS] for s in g16.MAP_STATES]) for t in g16.TASKS}
    ct = dsc.context_tasks({t: (Y, None) for t, Y in tasks.items()}, {t: list(g16.MAP_STATES) for t in tasks}, ctx)
    scores = {t: g16.near_scores(Y, g16.neighbours(C)) for t, (Y, C) in ct.items()}
    parts = g17.u1_parts(ct, 8, 10.0, scores=scores)
    assert parts["informative_states"] == 115 and round(parts["U1"], 4) == 0.0028


# ---- critic on contexts, and that the features never reach the critic ---------------------------------------------

def test_critic_row_is_step17s_rule_on_the_given_inputs():
    tasks = sim_tasks(4)
    r = dsc.critic_row(tasks, d=2, lam=10.0, n_perm=5)
    u = g17.u1_test(tasks, 2, 10.0, n_perm=5, seed=g17.PERM_SEED)
    assert r["U1"] == u["U1"] and r["p"] == u["p"] and r["D"] == u["D"] and r["seed"] == g17.PERM_SEED


def test_describe_never_fits_the_critic_on_the_features(monkeypatch):
    rng = np.random.default_rng(5)
    tasks = {t: (sim_task(rng, S=12)[0], rng.normal(size=(12, 40))) for t in (0, 4)}       # 40-column "features"
    ctx = {t: (Y, rng.normal(size=(12, 9))) for t, (Y, _) in tasks.items()}                # 9-column contexts
    seen = []
    real = g17.critic_scores
    monkeypatch.setattr(g17, "critic_scores", lambda Y, X, d, lam: (seen.append(X.shape[1]), real(Y, X, d, lam))[1])
    monkeypatch.setattr(g17, "D_PCA", 3)
    d = dsc.describe(tasks, ctx, near_n_perm=5, critic_n_perm=2, probe_n_perm=2)
    assert seen and set(seen) == {9}
    assert set(d["rows"]) == {"near, features", "near, contexts", "critic, contexts"}


# ---- probe -----------------------------------------------------------------------------------------------------

def test_q2_is_zero_for_the_loo_mean_and_one_for_exact_predictions():
    T = np.random.default_rng(6).normal(size=(10, 3))
    loo = (T.sum(0, keepdims=True) - T) / 9
    assert np.isclose(dsc.q2(T, loo), 0.0) and dsc.q2(T, T) == 1.0
    # known answer: one column 0, 1, 2; LOO means 1.5, 1, 0.5; prediction 0, 1, 1 -> 1 - 1 / (2.25 + 0 + 2.25)
    T1 = np.array([[0.0], [1.0], [2.0]])
    assert np.isclose(dsc.q2(T1, np.array([[0.0], [1.0], [1.0]])), 1 - 1 / 4.5)


def test_ridge_probe_matches_a_direct_primal_solve():
    rng = np.random.default_rng(7)
    X, T = rng.normal(size=(8, 5)), rng.normal(size=(8, 2))
    pred = dsc.loo_ridge_predictions(X, T, alphas=(0.3,))[0.3]
    s = 3
    tr = np.arange(8) != s
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z, z = (X[tr] - mu) / sd, (X[s] - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + 0.3 * 5 * np.eye(5), Z.T @ (T[tr] - T[tr].mean(0)))
    assert np.allclose(pred[s], T[tr].mean(0) + z @ beta)


def test_probe_predictions_do_not_see_the_left_out_target():
    rng = np.random.default_rng(8)
    X, T = rng.normal(size=(12, 30)), rng.normal(size=(12, 3))
    T2 = T.copy()
    T2[4] += 100.0
    a = dsc.loo_ridge_predictions(X, T)
    b = dsc.loo_ridge_predictions(X, T2)
    assert all(np.allclose(a[k][4], b[k][4]) for k in a)
    assert np.allclose(dsc.loo_pca_predictions(X, T, d=3)[4], dsc.loo_pca_predictions(X, T2, d=3)[4])


def test_probe_finds_a_linear_scene_and_chance_does_not():
    rng = np.random.default_rng(9)
    S = 30
    T = {t: rng.normal(size=(S, 4)) for t in (0, 4)}
    X = {t: np.hstack([T[t] @ rng.normal(size=(4, 20)), 0.05 * rng.normal(size=(S, 20))]) for t in T}
    p = dsc.probe(X, T)
    assert p["ridge alpha 0.01"]["pooled"] > 0.9 and p["critic input (pca 8)"]["pooled"] > 0.9
    noise = {t: rng.normal(size=(S, 40)) for t in T}
    assert all(v["pooled"] < 0.1 for v in dsc.probe(noise, T).values())
    ch = dsc.probe_chance(X, T, n_perm=10)
    assert all(c["q95"] < 0.2 for c in ch.values())
    assert ch == dsc.probe_chance(X, T, n_perm=10)                                         # seeded


def test_pooled_probe_weights_tasks_by_their_press():
    rng = np.random.default_rng(10)
    T = {0: rng.normal(size=(15, 2)), 4: rng.normal(size=(15, 3))}
    X = {t: rng.normal(size=(15, 12)) for t in T}
    p = dsc.probe(X, T)["ridge alpha 1"]
    num = den = 0.0
    for t in T:
        a, b = dsc.press_parts(T[t], dsc.loo_ridge_predictions(X[t], T[t])[1.0])
        num, den = num + a, den + b
        assert np.isclose(p["per_task"][str(t)], 1 - a / b)
    assert np.isclose(p["pooled"], 1 - num / den)


# ---- inputs and main ---------------------------------------------------------------------------------------------

def test_context_tasks_take_the_rows_of_the_kept_states():
    ctx = {"0": {"z": [[float(s), -float(s)] for s in range(50)]}}
    Y = np.zeros((3, 4), dtype=int)
    out = dsc.context_tasks({0: (Y, None)}, {0: [2, 5, 29]}, ctx)
    assert np.array_equal(out[0][1], [[2, -2], [5, -5], [29, -29]])
    with pytest.raises(ValueError, match="outside the map"):
        dsc.context_tasks({0: (np.zeros((1, 4), dtype=int), None)}, {0: [30]}, ctx)
    with pytest.raises(ValueError, match="outside the map"):                                # one held-out among many
        dsc.context_tasks({0: (np.zeros((2, 4), dtype=int), None)}, {0: [5, 30]}, ctx)


def test_main_runs_end_to_end_on_small_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    by_task = {0: range(12), 4: range(12), 6: range(3, 15), 7: range(12)}
    table_csv(tmp_path / "t.csv", by_task, rng)
    features_json(tmp_path / "f.json", by_task, rng, dim=10)
    ctx = {str(t): {"z": rng.normal(size=(50, 9)).tolist()} for t in by_task}
    (tmp_path / "c.json").write_text(json.dumps(ctx))
    monkeypatch.setattr(g17, "D_PCA", 3)
    out = tmp_path / "r.txt"
    assert dsc.main(["--table", str(tmp_path / "t.csv"), "--features", str(tmp_path / "f.json"), "--contexts",
                     str(tmp_path / "c.json"), "--out", str(out), "--near_n_perm", "5", "--critic_n_perm", "2",
                     "--probe_n_perm", "2"]) == 0
    text = out.read_text()
    assert "critic on the policy's features is U1, not computed here" in text
    assert "near, features" in text and "critic input (pca 3)" in text
    d = json.loads((tmp_path / "r.txt.json").read_text())
    assert d["states"] == {"0": 12, "4": 12, "6": 12, "7": 12}
    assert d["provenance"] == {"table sha256": dsc.sha256(tmp_path / "t.csv"),
                               "features sha256": dsc.sha256(tmp_path / "f.json"),
                               "contexts sha256": dsc.sha256(tmp_path / "c.json")}


def test_report_prints_intervals_in_order_and_tasks_without_informative_states():
    row = {"informative_states": 7, "U1": 0.01, "U1_ci95": (-0.02, 0.03), "p": 0.2, "D": 0.04, "D_ci95": (-0.05, 0.06),
           "per_task": {"0": {"U1": None}, "4": {"U1": 0.5, "D": -0.25, "auroc_critic": 0.7, "auroc_flat": 0.6}}}
    d = {"settings": {"m": 8, "near_n_perm": 5, "near_seed": 17, "d": 8, "lam": 10.0, "critic_n_perm": 2,
                      "critic_seed": 18, "n_boot": 2000, "boot_seed": 19, "top_q": 5, "probe_n_perm": 2,
                      "probe_seed": 21},
         "states": {"0": 3}, "rows": {"near, features": row},
         "probe": {"ridge alpha 1": {"pooled": 0.3, "per_task": {"0": 0.2}}},
         "probe_chance": {"ridge alpha 1": {"mean": -0.1, "q95": 0.05}}}
    text = dsc.format_report(d)
    assert "near, features | 7 | +0.0100 [-0.0200, +0.0300] | 0.2 | +0.0400 [-0.0500, +0.0600]" in text
    assert "0 no informative state; 4 +0.500 / -0.250 / 0.700 / 0.600" in text
    assert "ridge alpha 1 | +0.300 | -0.100 | +0.050 | 0 +0.200" in text
