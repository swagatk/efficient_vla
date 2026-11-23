"""Step 17's rule (noise_reachability/step17_g1b.py): the critic, its leave-one-state-out scores, U1, D and the null."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noise_reachability"))
import step17_g1b as g  # noqa: E402


def sim_task(rng, inter=1.0, S=30, K=50, extra=0, sd_id=0.5):
    z = rng.normal(size=(S, 2))
    v = rng.normal(size=(K, 2))
    logit = rng.normal(0, 2, (S, 1)) + rng.normal(0, sd_id, (1, K)) + inter * (z @ v.T) / np.sqrt(2)
    Y = (rng.random((S, K)) < 1 / (1 + np.exp(-logit))).astype(int)
    return Y, np.hstack([z, rng.normal(size=(S, extra))])


def test_critic_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    Y = (rng.random((6, 5)) < 0.5).astype(float)
    F = rng.normal(size=(6, 2))
    theta = rng.normal(size=6 + 5 + 5 * 2)
    _, grad = g.critic_loss(theta, Y, F, lam=0.7)
    eps = 1e-6
    num = np.array([(g.critic_loss(theta + eps * e, Y, F, 0.7)[0] - g.critic_loss(theta - eps * e, Y, F, 0.7)[0]) / (2 * eps)
                    for e in np.eye(len(theta))])
    assert np.allclose(grad, num, atol=1e-5)


def test_pca_is_fitted_on_training_rows_with_unit_variance_and_fixed_signs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(29, 12)) @ rng.normal(size=(12, 40))
    X[:, 3] = 7.0                                            # a constant column is dropped, not divided by zero
    p = g.pca_fit(X, 4)
    F = g.pca_apply(p, X)
    assert F.shape == (29, 4) and np.allclose(F.std(0), 1.0) and np.allclose(F.mean(0), 0.0, atol=1e-12)
    # The sign is fixed on the loadings, so negated data gives the same loadings and negated scores.
    assert np.allclose(g.pca_fit(-X, 4)["comp"], p["comp"]) and np.allclose(g.pca_apply(g.pca_fit(-X, 4), -X), -F)
    with pytest.raises(ValueError):
        g.pca_fit(X[:3], 4)


def test_left_out_state_scores_do_not_see_its_outcomes():
    rng = np.random.default_rng(2)
    Y, X = sim_task(rng)
    a = g.critic_scores(Y, X, 2, 1.0)
    Y2 = Y.copy()
    Y2[7] = 1 - Y2[7]
    b = g.critic_scores(Y2, X, 2, 1.0)
    assert np.array_equal(a[7], b[7])                          # row 7 is scored by a model that never saw row 7
    assert not np.array_equal(a[8], b[8])                      # other rows did train on it


def test_critic_without_interaction_ranks_as_the_ticket():
    rng = np.random.default_rng(3)
    Y, X = sim_task(rng, inter=0.0)
    sc = g.critic_scores(Y, X, 2, lam=1e8)                     # w forced to ~0: score is b_k
    fl = g.g16.flat_scores(Y)
    for s in range(Y.shape[0]):
        i, j = np.triu_indices(Y.shape[1], 1)
        distinct = fl[s, i] != fl[s, j]
        assert np.all(np.sign(sc[s, i] - sc[s, j])[distinct] == np.sign(fl[s, i] - fl[s, j])[distinct])


def test_planted_interaction_is_found_and_none_is_not():
    rng = np.random.default_rng(4)
    planted = {t: sim_task(rng, inter=1.5) for t in range(2)}
    none = {t: sim_task(rng, inter=0.0) for t in range(2)}
    r1, r0 = g.u1_parts(planted, 2, 1.0), g.u1_parts(none, 2, 1.0)
    assert r1["U1"] > 0.08 and r1["D"] > 0.05
    assert r0["U1"] < 0.02


def test_same_inputs_give_the_same_result_twice():
    rng = np.random.default_rng(5)
    tasks = {0: sim_task(rng, extra=3)}
    a = g.u1_test(tasks, 2, 1.0, n_perm=3, seed=9)
    b = g.u1_test(tasks, 2, 1.0, n_perm=3, seed=9)
    assert a == b


def test_top_rate_breaks_ties_by_lower_index():
    score = np.array([[1.0, 1.0, 0.0, 1.0]])
    Y = np.array([[1, 0, 1, 0]])
    assert g.top_rate(score, Y, q=1)[0] == 1.0                 # id 0 wins the tie
    assert g.top_rate(score, Y, q=2)[0] == 0.5                 # ids 0 and 1


def test_null_permutes_only_features_within_strata(monkeypatch):
    rng = np.random.default_rng(6)
    Y, X = sim_task(rng)
    seen = []
    real = g.u1_parts

    def spy(tasks, d, lam, scores=None):
        (Yp, Xp), = tasks.values()
        seen.append((Yp, Xp))
        return real(tasks, d, lam, scores)

    monkeypatch.setattr(g, "u1_parts", spy)
    g.u1_test({0: (Y, X)}, 2, 1.0, n_perm=4, seed=1)
    groups = g.g16.strata(Y, g.STRATUM_SIZE)
    where = {s: i for i, grp in enumerate(groups) for s in grp}
    for Yp, Xp in seen[1:]:
        assert Yp is Y                                         # outcomes untouched
        src = [int(np.flatnonzero((X == row).all(1))[0]) for row in Xp]
        assert sorted(src) == list(range(len(X)))              # a permutation of the rows
        assert all(where[s] == where[src[s]] for s in range(len(X)))


def test_passing_needs_p_and_both_signs(monkeypatch):
    rng = np.random.default_rng(7)
    tasks = {0: sim_task(rng)}
    for u1, dval, want in ((0.05, 0.02, True), (-0.01, 0.02, False), (0.05, -0.01, False), (0.05, 0.0, False)):
        # observed first, then the null: make every null copy lower than the observation
        calls = iter([{"U1": u1, "D": dval, "informative_states": 1, "per_task": {}}]
                     + [{"U1": -1.0, "D": 0.0, "informative_states": 1, "per_task": {}}] * 19)
        monkeypatch.setattr(g, "u1_parts", lambda tasks, d, lam, scores=None: next(calls))
        r = g.u1_test(tasks, 2, 1.0, n_perm=19, seed=0)
        assert r["p"] == pytest.approx(1 / 20) and r["passes"] is want


def test_load_inputs_keeps_states_with_features_only(tmp_path):
    import json

    rng = np.random.default_rng(8)
    table_csv(tmp_path / "t.csv", {t: range(12) for t in g.g16.TASKS}, rng)
    by_task = {t: [s for s in range(12) if not (t == 4 and s == 5)] for t in g.g16.TASKS}
    features_json(tmp_path / "f.json", by_task, rng)
    tasks, states = g.load_inputs(tmp_path / "t.csv", tmp_path / "f.json")
    assert states[4] == [s for s in range(12) if s != 5] and tasks[4][0].shape == (11, 50) and tasks[0][1].shape == (12, 6)
    rows = json.loads((tmp_path / "f.json").read_text())["rows"]
    (tmp_path / "d.json").write_text(json.dumps({"rows": rows + rows[:1]}))
    with pytest.raises(ValueError, match="duplicate"):
        g.load_inputs(tmp_path / "t.csv", tmp_path / "d.json")


def test_load_inputs_checks_the_committed_step16_table(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    table_csv(tmp_path / "t.csv", {t: range(10) for t in g.g16.TASKS}, rng)
    features_json(tmp_path / "f.json", {t: range(10) for t in g.g16.TASKS}, rng)
    monkeypatch.setattr(g, "STEP16_TABLE", tmp_path / "t.csv")        # pretend this is the committed table
    with pytest.raises(ValueError, match="not the committed table"):
        g.load_inputs(tmp_path / "t.csv", tmp_path / "f.json")
    import hashlib
    monkeypatch.setattr(g, "STEP16_TABLE_SHA256", hashlib.sha256((tmp_path / "t.csv").read_bytes()).hexdigest())
    assert set(g.load_inputs(tmp_path / "t.csv", tmp_path / "f.json")[0]) == set(g.g16.TASKS)


def test_committed_step16_table_has_the_recorded_hash():
    import hashlib
    assert hashlib.sha256(g.STEP16_TABLE.read_bytes()).hexdigest() == g.STEP16_TABLE_SHA256


def test_main_refuses_until_d_and_lambda_are_fixed(monkeypatch):
    monkeypatch.setattr(g, "D_PCA", None)
    with pytest.raises(SystemExit, match="not fixed"):
        g.main(["--table", "x", "--features", "y"])



def test_power_simulation_share_is_the_scene_fraction_of_feature_variance(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noise_reachability" / "calibration"))
    import step17_power as pw

    rng = np.random.default_rng(10)
    monkeypatch.setattr(pw, "STATES", 2000)
    for share in (0.05, 0.5):
        total = np.mean([X.var(0).sum() for Y, X in pw.simulate(rng, sd_id=0.5, sd_inter=0.0, share=share).values()])
        assert total == pytest.approx(1.0, rel=0.05)           # scene part + irrelevant part = unit total variance
    monkeypatch.setattr(pw, "STATES", 50)
    (_, X), *_ = pw.simulate(rng, sd_id=0.5, sd_inter=0.0, share=1.0).values()
    assert np.linalg.matrix_rank(X - X.mean(0), tol=1e-8) == 2  # share 1: features are a linear map of the 2-d latent


# ---- added after the first mutation check (each test pins a behaviour a mutant changed unnoticed) -------------------

def test_pca_components_are_the_correlation_matrix_eigenvectors_with_largest_loading_positive():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(29, 6)) @ rng.normal(size=(6, 6)) + 5.0      # an offset, so centring matters
    p = g.pca_fit(X, 3)
    Z = (X - X.mean(0)) / X.std(0)
    w, v = np.linalg.eigh(np.cov(Z.T))
    top = v[:, np.argsort(w)[::-1][:3]].T
    assert np.allclose(np.abs(p["comp"] @ top.T), np.eye(3), atol=1e-6)     # same directions as the eigenvectors
    lead = p["comp"][np.arange(3), np.abs(p["comp"]).argmax(1)]
    assert np.all(lead > 0)                                                   # sign convention: largest loading > 0


def test_d_and_u1_are_differences_from_the_ticket():
    rng = np.random.default_rng(12)
    tasks = {0: sim_task(rng, inter=0.0, sd_id=2.0)}                         # a strong shared id effect: the ticket works
    Y = tasks[0][0]
    fl = g.g16.flat_scores(Y)
    same = g.u1_parts(tasks, 2, 1.0, scores={0: fl})
    assert same["U1"] == 0.0 and same["D"] == 0.0                            # the ticket against itself
    rev = g.u1_parts(tasks, 2, 1.0, scores={0: -fl})
    assert rev["U1"] < 0 and rev["D"] < 0                                    # the ticket reversed is worse
    v = rev["per_task"]["0"]
    assert v["U1"] == pytest.approx(rev["U1"]) and v["D"] == pytest.approx(rev["D"])
    assert v["auroc_critic"] == pytest.approx(1 - v["auroc_flat"], abs=1e-9)


def test_per_task_values_are_reported_when_some_states_are_not_informative():
    rng = np.random.default_rng(13)
    Y, X = sim_task(rng)
    Y[0] = 0                                                                 # one state never succeeds
    r = g.u1_parts({0: (Y, X)}, 2, 1.0)
    v = r["per_task"]["0"]
    assert v["informative_states"] == r["informative_states"] < Y.shape[0]
    assert None not in (v["U1"], v["auroc_critic"], v["auroc_flat"])


def test_free_permutation_moves_features_across_all_states():
    rng = np.random.default_rng(14)
    Y, X = sim_task(rng, S=12, K=50)
    r = g.u1_test({0: (Y, X)}, 2, 1.0, n_perm=2, seed=3, stratum_size=None)
    assert r["stratum_size"] is None and 0 < r["p"] <= 1


def test_passing_needs_u1_strictly_above_zero(monkeypatch):
    calls = iter([{"U1": 0.0, "D": 0.02, "informative_states": 1, "per_task": {}}]
                 + [{"U1": -1.0, "D": 0.0, "informative_states": 1, "per_task": {}}] * 19)
    monkeypatch.setattr(g, "u1_parts", lambda tasks, d, lam, scores=None: next(calls))
    r = g.u1_test({0: sim_task(np.random.default_rng(0))}, 2, 1.0, n_perm=19, seed=0)
    assert r["p"] <= g.ALPHA and r["passes"] is False


def test_reading_follows_the_rule_at_the_boundary():
    assert g.reading({"passes": True, "p": 0.01}).startswith("the critic over the policy's view")
    assert g.reading({"passes": False, "p": g.ALPHA}).startswith("a scene signal")
    assert g.reading({"passes": False, "p": 0.051}).startswith("not shown")


def table_csv(path, states_by_task, rng):
    rows = ["task_id,init_state,noise_id,success,steps"]
    for t, states in states_by_task.items():
        for s in states:
            for k in g.g16.IDS:
                rows.append(f"{t},{s},{k},{int(rng.random() < 0.4)},100")
    path.write_text("\n".join(rows) + "\n")


def features_json(path, states_by_task, rng, dim=6):
    """The features file analyze_step17.py writes."""
    import json

    rows = [{"task_id": t, "init_state": s, "features": [float(x) for x in rng.normal(size=dim).astype(np.float32)]}
            for t, states in states_by_task.items() for s in states]
    Path(path).write_text(json.dumps({"rows": rows}))


def test_load_inputs_reads_each_task_on_its_own_and_checks_the_state_count(tmp_path):
    rng = np.random.default_rng(15)
    by_task = {0: range(30), 4: range(10), 6: range(5, 17), 7: range(12)}        # all 30, exactly 10, shifted, 12
    table_csv(tmp_path / "t.csv", by_task, rng)
    features_json(tmp_path / "f.json", by_task, rng)
    tasks, states = g.load_inputs(tmp_path / "t.csv", tmp_path / "f.json")
    assert {t: list(v) for t, v in states.items()} == {t: list(v) for t, v in by_task.items()}
    assert tasks[0][0].shape == (30, 50) and tasks[4][0].shape == (10, 50) and tasks[6][0].shape == (12, 50)
    features_json(tmp_path / "f9.json", {**by_task, 4: range(9)}, rng)          # nine states with features: refused
    with pytest.raises(ValueError, match="not the protocol's"):
        g.load_inputs(tmp_path / "t.csv", tmp_path / "f9.json")
    table_csv(tmp_path / "t31.csv", {**by_task, 0: range(31)}, rng)           # a state outside 0-29: refused
    features_json(tmp_path / "f31.json", {**by_task, 0: range(31)}, rng)
    with pytest.raises(ValueError, match="not the protocol's"):
        g.load_inputs(tmp_path / "t31.csv", tmp_path / "f31.json")


def test_critic_fit_accepts_a_line_search_stop_and_refuses_other_failures(monkeypatch):
    import scipy.optimize as so

    class Res:
        def __init__(self, ok, msg, n):
            self.success, self.message, self.x = ok, msg, np.zeros(n)

    Y, F = np.zeros((3, 4)), np.zeros((3, 1))
    n = 3 + 4 + 4
    monkeypatch.setattr(so, "minimize", lambda *a, **k: Res(False, "ABNORMAL_TERMINATION_IN_LNSRCH", n))
    b, W = g.fit_critic(Y, F, 1.0)
    assert b.shape == (4,) and W.shape == (4, 1)
    monkeypatch.setattr(so, "minimize", lambda *a, **k: Res(False, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", n))
    with pytest.raises(RuntimeError, match="did not converge"):
        g.fit_critic(Y, F, 1.0)


def test_main_writes_the_report_and_its_json(tmp_path, monkeypatch):
    import json

    rng = np.random.default_rng(16)
    by_task = {t: range(10) for t in g.g16.TASKS}
    table_csv(tmp_path / "t.csv", by_task, rng)
    features_json(tmp_path / "f.json", by_task, rng)
    monkeypatch.setattr(g, "D_PCA", 2)
    monkeypatch.setattr(g, "LAM", 1.0)
    out = tmp_path / "r.txt"
    assert g.main(["--table", str(tmp_path / "t.csv"), "--features", str(tmp_path / "f.json"), "--out", str(out),
                   "--n_perm", "3"]) == 0
    text = out.read_text()
    r = json.loads((tmp_path / "r.txt.json").read_text())
    assert r["n_perm"] == 3 and r["d"] == 2 and r["lam"] == 1.0 and set(r["per_task"]) == {"0", "4", "6", "7"}
    assert text.startswith("step 17 test U1 over") and text.endswith("\n") and text.count("\ntask ") == 4
    assert f"one-sided p {r['p']:.4g}" in text and f"U1: {r['U1']:.4f}" in text
    assert text.splitlines()[-1] == f"reading: {g.reading(r)}"
    assert "\n" + text.splitlines()[1] + "\n" in "\n" + text            # each part on its own line


def test_report_names_a_task_without_informative_states():
    r = {"informative_states": 5, "d": 2, "lam": 1.0, "n_perm": 3, "stratum_size": 5, "U1": 0.01, "null_mean": 0.0,
         "null_sd": 0.01, "p": 0.5, "D": 0.0, "passes": False,
         "per_task": {"0": {"U1": None, "D": 0.0, "informative_states": 0, "auroc_critic": None, "auroc_flat": None},
                      "4": {"U1": 0.02, "D": 0.01, "informative_states": 5, "auroc_critic": 0.7, "auroc_flat": 0.68}}}
    lines = g.format_report(r).splitlines()
    assert lines[2] == "task 0 (reported, not tested): no informative state"
    assert lines[3].startswith("task 4 (reported, not tested): U1 0.0200, D 0.0100, mean AUROC critic 0.700 / ticket 0.680")


def test_paired_ci_known_answers():
    assert g.paired_ci([0.25] * 30) == (0.25, 0.25)
    lo, hi = g.paired_ci(list(range(-10, 11)), n_boot=4000, seed=1)
    assert lo < 0 < hi and abs(lo + hi) < 0.6                                   # centred near the mean, 0
    assert g.paired_ci([1.0, 2.0, 3.0]) == g.paired_ci([1.0, 2.0, 3.0])        # seeded
    lo, hi = g.paired_ci(np.r_[np.zeros(50), np.ones(50)])
    assert 0.39 < lo < 0.45 and 0.55 < hi < 0.61                                 # mean 0.5, se 0.05
    with pytest.raises(ValueError):
        g.paired_ci([])


def test_per_state_values_are_what_u1_and_d_average():
    rng = np.random.default_rng(17)
    tasks = {0: sim_task(rng), 4: sim_task(rng)}
    r = g.u1_parts(tasks, 2, 1.0)
    assert len(r["u1_states"]) == r["informative_states"] and len(r["d_states"]) == 60
    assert np.mean(r["u1_states"]) == pytest.approx(r["U1"]) and np.mean(r["d_states"]) == pytest.approx(r["D"])


def test_reported_interaction_off_is_the_smoothed_ticket():
    rng = np.random.default_rng(18)
    null = {0: sim_task(rng, inter=0.0)}
    x0 = g.reported(null, g.u1_parts(null, 2, 1.0), 2, 1.0)
    assert x0["w_off"]["lam"] == g.W_OFF_LAM and abs(x0["w_off"]["U1"]) < 0.01  # w ~ 0 ranks about as the ticket
    tasks = {0: sim_task(rng, inter=1.5)}
    obs = g.u1_parts(tasks, 2, 1.0)
    x = g.reported(tasks, obs, 2, 1.0)
    assert obs["U1"] > x["w_off"]["U1"] + 0.05                                   # the planted interaction is the gain
    assert x["U1_ci95"][0] <= obs["U1"] <= x["U1_ci95"][1] and x["D_ci95"][0] <= obs["D"] <= x["D_ci95"][1]


def test_main_report_has_the_reported_line(tmp_path, monkeypatch):
    rng = np.random.default_rng(19)
    by_task = {t: range(10) for t in g.g16.TASKS}
    table_csv(tmp_path / "t.csv", by_task, rng)
    features_json(tmp_path / "f.json", by_task, rng)
    monkeypatch.setattr(g, "D_PCA", 2)
    g.main(["--table", str(tmp_path / "t.csv"), "--features", str(tmp_path / "f.json"), "--out", str(tmp_path / "r.txt"),
            "--n_perm", "2"])
    line = [l for l in (tmp_path / "r.txt").read_text().splitlines() if l.startswith("reported, not part of the rule")]
    assert len(line) == 1 and "interaction off (w ~ 0)" in line[0] and "U1 95% interval" in line[0]


def test_paired_ci_takes_the_2_5_and_97_5_percent_quantiles():
    v = np.random.default_rng(20).normal(size=40)
    rng = np.random.default_rng(g.BOOT_SEED)
    means = v[rng.integers(0, v.size, size=(g.N_BOOT, v.size))].mean(1)
    assert g.paired_ci(v) == (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    lo, hi = g.paired_ci(v, level=0.5)
    assert (lo, hi) == (float(np.quantile(means, 0.25)), float(np.quantile(means, 0.75)))


def test_report_prints_each_interval_bound_in_its_place():
    r = {"informative_states": 5, "d": 2, "lam": 1.0, "n_perm": 3, "stratum_size": 5, "U1": 0.01, "null_mean": 0.0,
         "null_sd": 0.01, "p": 0.5, "D": 0.0, "passes": False, "per_task": {},
         "reported": {"U1_ci95": (-0.0123, 0.0456), "D_ci95": (-0.0071, 0.0089), "n_boot": 2000,
                      "w_off": {"U1": 0.0011, "D": -0.0022, "lam": 1e8}}}
    line = g.format_report(r).splitlines()[2]
    assert line == ("reported, not part of the rule: U1 95% interval -0.0123 to +0.0456, D -0.0071 to +0.0089 (paired "
                    "bootstrap over states, 2000); interaction off (w ~ 0): U1 +0.0011, D -0.0022")
