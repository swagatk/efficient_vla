#!/usr/bin/env python3
"""Step 17's test U1 (README.md, step 17): does a critic over the policy's features of the first observation rank a
left-out state's noise ids better than the per-task ticket? Run locally on CPU on step 16's complete map, never on
Kaggle.

The data of one task is step 16's complete 0/1 table ``Y`` (states x noise ids) and a feature matrix ``X`` (states x D),
one row per state: SmolVLA's pooled prefix features at the episode's first inference (or, for comparison only, step 16's
privileged contexts).

The critic, per task and left-out state s, fitted on the task's other states r and all ids k:

    logit P(success | r, k) = a_r + b_k + f(r) . w_k

f(r): the first ``d`` principal components of the training states' features (columns z-scored on the training states,
components scaled to unit variance there), then applied to s. Penalties: ``lam`` * |w|^2 and LAM_AB * (|a|^2 + |b|^2) on
the summed Bernoulli log loss; L-BFGS from zero, so a fit is deterministic. The score of id k on s is b_k + f(s) . w_k
(a_s does not change a ranking within s). With w = 0 the score is a smoothed per-task ticket.

- **U1** = mean over informative states (tasks pooled) of AUROC_s(critic) - AUROC_s(flat), flat being step 16's
  per-task ticket score (id k's success rate on the task's other states).
- Null: step 16's stratified permutation of the feature rows within each task (strata of STRATUM_SIZE states by success
  count), with the PCA and the critic refitted on every permuted copy. It keeps every outcome and the link between scene
  and difficulty, and breaks only the link between scene and which ids succeed.
- **D** = mean over all states of the success rate of the critic's TOP_Q top-scored ids minus that of flat's TOP_Q
  (ties broken by the lower id index): what choosing by the critic gains over the ticket.
- U1 passes if p = (1 + #{null >= U1}) / (1 + n_perm) <= ALPHA, U1 > 0 and D > 0.

*Reported, not part of the rule* (``reported``): a paired bootstrap 95% interval over states for U1 and D (states
resampled with replacement, tasks pooled; approximate, since a task's leave-one-out fits share data), and U1 and D of
the critic with its interaction switched off (lambda W_OFF_LAM, w ~ 0: a smoothed per-task ticket), which separates "the
features help" from "the logistic smoothing helps". The sign conditions of the rule rest on point estimates; these say
by how much the critic is above the ticket.

    python noise_reachability/step17_g1b.py --features <analyze_step17's step17_features.json> --out r.txt

The table defaults to step 16's committed one (sha256-checked); the features file is analyze_step17.py's, with only the
states that passed the feature check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step16_g1 as g16  # noqa: E402  (flat scores, AUROC, strata and the stratified permutation are step 16's own)

D_PCA = 8           # chosen by calibration/step17_power.py before any feature existed (README.md, step 17): highest
LAM = 10.0          # mean power of the 3 x 3 grid (stage select), level <= 0.033 under the confound (stage level)
LAM_AB = 1e-3
TOP_Q = 5
N_PERM = 1000
ALPHA = 0.05
PERM_SEED = 18
STRATUM_SIZE = g16.STRATUM_SIZE
N_BOOT = 2000
BOOT_SEED = 19
W_OFF_LAM = 1e8     # a ridge this strong leaves w ~ 0: the critic without its interaction
MIN_STD = 1e-8      # a feature column that varies less than this over the training states is dropped
ROOT = Path(__file__).resolve().parent.parent
STEP16_TABLE = ROOT / "week6" / "noise_reachability_step16_kaggle" / "run" / "step16_table_parts5.csv"
STEP16_TABLE_SHA256 = "ed0b3e32ee140d639b57415b3c690230a8c84bb96939563a11eeb4d42c1e036b"


# ---- features --------------------------------------------------------------------------------------------------

def pca_fit(X: np.ndarray, d: int) -> dict:
    """PCA of the training rows X (R x D): column means and sds, the first d directions and their sds."""
    mu, sd = X.mean(0), X.std(0)
    keep = sd > MIN_STD
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    _, sv, vt = np.linalg.svd(Z, full_matrices=False)
    if d > len(sv):
        raise ValueError(f"d={d} components from {len(sv)} available")
    comp = vt[:d]
    # A component's sign is arbitrary; fix it (largest-magnitude loading positive) so fits are reproducible.
    comp = comp * np.sign(comp[np.arange(d), np.abs(comp).argmax(1)])[:, None]
    scale = (Z @ comp.T).std(0)
    scale[scale < MIN_STD] = 1.0
    return {"mu": mu[keep], "sd": sd[keep], "keep": keep, "comp": comp, "scale": scale}


def pca_apply(p: dict, X: np.ndarray) -> np.ndarray:
    return ((X[:, p["keep"]] - p["mu"]) / p["sd"]) @ p["comp"].T / p["scale"]


# ---- the critic ------------------------------------------------------------------------------------------------

def _unpack(theta, R, K, d):
    a, b = theta[:R], theta[R:R + K]
    return a, b, theta[R + K:].reshape(K, d)


def critic_loss(theta, Y, F, lam, lam_ab=LAM_AB):
    """(loss, gradient): summed Bernoulli log loss of logit a_r + b_k + F_r . W_k plus the ridge penalties."""
    R, K = Y.shape
    d = F.shape[1]
    a, b, W = _unpack(theta, R, K, d)
    eta = a[:, None] + b[None, :] + F @ W.T
    loss = np.logaddexp(0.0, eta).sum() - (Y * eta).sum()
    loss += lam * (W ** 2).sum() + lam_ab * ((a ** 2).sum() + (b ** 2).sum())
    G = 1.0 / (1.0 + np.exp(-eta)) - Y
    grad = np.concatenate([G.sum(1) + 2 * lam_ab * a, G.sum(0) + 2 * lam_ab * b, (G.T @ F + 2 * lam * W).ravel()])
    return loss, grad


def fit_critic(Y: np.ndarray, F: np.ndarray, lam: float, lam_ab: float = LAM_AB) -> tuple[np.ndarray, np.ndarray]:
    """(b (K,), W (K, d)) of the critic fitted on the training table Y (R x K) with features F (R x d)."""
    from scipy.optimize import minimize

    R, K = Y.shape
    d = F.shape[1]
    res = minimize(critic_loss, np.zeros(R + K + K * d), args=(Y, F, lam, lam_ab), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-6})
    if not res.success and "ABNORMAL" not in str(res.message):
        raise RuntimeError(f"critic fit did not converge: {res.message}")
    _, b, W = _unpack(res.x, R, K, d)
    return b, W


def critic_scores(Y: np.ndarray, X: np.ndarray, d: int, lam: float) -> np.ndarray:
    """(S x K) leave-one-state-out critic scores: row s from a PCA and critic fitted on the other states only."""
    S = Y.shape[0]
    out = np.empty(Y.shape, dtype=float)
    for s in range(S):
        tr = np.arange(S) != s
        p = pca_fit(X[tr], d)
        b, W = fit_critic(Y[tr], pca_apply(p, X[tr]), lam)
        out[s] = b + W @ pca_apply(p, X[s:s + 1])[0]
    return out


def top_rate(score: np.ndarray, Y: np.ndarray, q: int = TOP_Q) -> np.ndarray:
    """Per row: the success rate of its q top-scored ids (ties broken by the lower index)."""
    order = np.argsort(-score, axis=1, kind="stable")[:, :q]
    return np.take_along_axis(Y, order, axis=1).mean(1)


# ---- the statistic and its test ---------------------------------------------------------------------------------

def u1_parts(tasks: dict, d: int, lam: float, scores: dict | None = None) -> dict:
    """U1, D and their per-task values over tasks {t: (Y, X)}; ``scores`` gives each task's critic scores."""
    vals, dvals, per = [], [], {}
    for t, (Y, X) in tasks.items():
        sc = scores[t] if scores is not None else critic_scores(Y, X, d, lam)
        fl = g16.flat_scores(Y)
        a_c, a_f = g16.auroc_rows(sc, Y), g16.auroc_rows(fl, Y)
        ok = ~np.isnan(a_f)
        dd = top_rate(sc, Y) - top_rate(fl, Y)
        vals.extend(a_c[ok] - a_f[ok])
        dvals.extend(dd)
        per[str(t)] = {"U1": float(np.mean(a_c[ok] - a_f[ok])) if ok.any() else None, "D": float(dd.mean()),
                       "informative_states": int(ok.sum()), "auroc_critic": float(np.nanmean(a_c)) if ok.any() else None,
                       "auroc_flat": float(np.nanmean(a_f)) if ok.any() else None}
    if not vals:
        raise ValueError("no informative state")
    return {"U1": float(np.mean(vals)), "D": float(np.mean(dvals)), "informative_states": len(vals), "per_task": per,
            "u1_states": [float(v) for v in vals], "d_states": [float(v) for v in dvals]}


def u1_test(tasks: dict, d: int, lam: float, n_perm: int = N_PERM, seed: int = PERM_SEED,
            stratum_size: int | None = STRATUM_SIZE, return_null: bool = False) -> dict:
    """The rule: U1 and D on the data, U1's stratified permutation null, and whether U1 passes.
    ``stratum_size=None`` permutes freely (invalid; only to show the confound in calibration)."""
    obs = u1_parts(tasks, d, lam)
    rng = np.random.default_rng(seed)
    groups = {t: (g16.strata(Y, stratum_size) if stratum_size else [np.arange(Y.shape[0])]) for t, (Y, _) in tasks.items()}
    null = np.empty(n_perm)
    for i in range(n_perm):
        permuted = {}
        for t, (Y, X) in tasks.items():
            perm = g16.stratified_permutation(rng, groups[t], Y.shape[0])   # state s gets the features of perm[s]
            permuted[t] = (Y, X[perm])
        null[i] = u1_parts(permuted, d, lam)["U1"]
    p = g16.perm_p(obs["U1"], null)
    out = {**obs, "p": p, "null_mean": float(null.mean()), "null_sd": float(null.std()), "n_perm": n_perm, "d": d,
           "lam": lam, "stratum_size": stratum_size,
           "passes": bool(p <= ALPHA and obs["U1"] > 0 and obs["D"] > 0)}
    if return_null:
        out["null"] = null
    return out


def paired_ci(vals, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of per-state differences (states resampled with replacement)."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def reported(tasks: dict, obs: dict, d: int, lam: float) -> dict:
    """Not part of the rule: bootstrap intervals of U1 and D, and U1 and D with the critic's interaction switched off."""
    off = u1_parts(tasks, d, W_OFF_LAM)
    return {"U1_ci95": paired_ci(obs["u1_states"]), "D_ci95": paired_ci(obs["d_states"]),
            "w_off": {"U1": off["U1"], "D": off["D"], "lam": W_OFF_LAM}, "n_boot": N_BOOT, "boot_seed": BOOT_SEED}


def reading(r: dict) -> str:
    if r["passes"]:
        return ("the critic over the policy's view ranks a state's noises above the shuffled-feature null, and its U1 "
                "and D are above the per-task ticket's (by the intervals reported): D1 and H1' go ahead with this "
                "critic, fixed on states 0-29; H1' must still beat the ticket on held-out states")
    if r["p"] <= ALPHA:
        return ("a scene signal reaches the features but does not beat the per-task ticket (U1 or D not above 0): "
                "ENEI reduces to the ticket, and D1 is not run for it")
    return "not shown that the policy's view predicts which noise suits a scene: the ticket is the ceiling without search"


# ---- inputs ----------------------------------------------------------------------------------------------------

def load_inputs(table_path, features_path) -> tuple[dict, dict]:
    """({task: (Y, X)}, {task: states}) from step 16's table and a features file (JSON, {"rows": [{task_id, init_state,
    features}]}), restricted to the states that have features (a state dropped by the feature check has none)."""
    import hashlib

    if Path(table_path).resolve() == STEP16_TABLE.resolve():
        got = hashlib.sha256(Path(table_path).read_bytes()).hexdigest()
        if got != STEP16_TABLE_SHA256:
            raise ValueError(f"{table_path}: sha256 {got} is not the committed table's")
    rows = json.loads(Path(features_path).read_text())["rows"]
    feats = {(int(r["task_id"]), int(r["init_state"])): np.asarray(r["features"], dtype=np.float32) for r in rows}
    if len(feats) != len(rows):
        raise ValueError("duplicate (task, state) in the features file")
    cells: dict = {}
    for r in csv.DictReader(open(table_path)):
        key = (int(r["task_id"]), int(r["init_state"]), int(r["noise_id"]))
        if key in cells:
            raise ValueError(f"duplicate row {key}")
        cells[key] = int(r["success"])
    if {k[0] for k in cells} != set(g16.TASKS):
        raise ValueError(f"tasks {sorted({k[0] for k in cells})}, expected {list(g16.TASKS)}")
    tasks, states_out = {}, {}
    for t in g16.TASKS:
        ids = sorted({k[2] for k in cells if k[0] == t})
        states = sorted(s for s in {k[1] for k in cells if k[0] == t} if (t, s) in feats)
        if ids != list(g16.IDS) or not set(states) <= set(g16.MAP_STATES) or len(states) < 10:
            raise ValueError(f"task {t}: ids {ids[0]}-{ids[-1]} and states {states} are not the protocol's")
        missing = [(s, k) for s in states for k in ids if (t, s, k) not in cells]
        if missing:
            raise ValueError(f"task {t}: {len(missing)} missing cells, e.g. {missing[:3]}")
        Y = np.array([[cells[(t, s, k)] for k in ids] for s in states])
        X = np.array([feats[(t, s)] for s in states], dtype=float)
        g16.check_table(Y, X)
        tasks[t], states_out[t] = (Y, X), states
    return tasks, states_out


def format_report(r: dict) -> str:
    lines = [f"step 17 test U1 over {r['informative_states']} informative states, critic d {r['d']}, lambda {r['lam']}, "
             f"{r['n_perm']} permutations (strata of {r['stratum_size']})",
             f"U1: {r['U1']:.4f} (null mean {r['null_mean']:.4f}, sd {r['null_sd']:.4f}), one-sided p {r['p']:.4g}; "
             f"D (top {TOP_Q} ids, critic - ticket): {r['D']:.4f}; {'passes' if r['passes'] else 'does not pass'}"]
    if "reported" in r:
        x = r["reported"]
        lines.append(f"reported, not part of the rule: U1 95% interval {x['U1_ci95'][0]:+.4f} to {x['U1_ci95'][1]:+.4f}, "
                     f"D {x['D_ci95'][0]:+.4f} to {x['D_ci95'][1]:+.4f} (paired bootstrap over states, {x['n_boot']}); "
                     f"interaction off (w ~ 0): U1 {x['w_off']['U1']:+.4f}, D {x['w_off']['D']:+.4f}")
    for t, v in r["per_task"].items():
        if v["U1"] is None:
            lines.append(f"task {t} (reported, not tested): no informative state")
            continue
        lines.append(f"task {t} (reported, not tested): U1 {v['U1']:.4f}, D {v['D']:.4f}, mean AUROC critic "
                     f"{v['auroc_critic']:.3f} / ticket {v['auroc_flat']:.3f} over {v['informative_states']} states")
    lines.append(f"reading: {reading(r)}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(STEP16_TABLE))
    ap.add_argument("--features", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n_perm", type=int, default=N_PERM, help="the protocol's is 1000; fewer only for tests")
    a = ap.parse_args(argv)
    if D_PCA is None or LAM is None:
        raise SystemExit("d and lambda are not fixed yet (README.md, step 17): run calibration/step17_power.py first")
    tasks, _ = load_inputs(a.table, a.features)
    r = u1_test(tasks, D_PCA, LAM, n_perm=a.n_perm)
    r["reported"] = reported(tasks, r, D_PCA, LAM)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
