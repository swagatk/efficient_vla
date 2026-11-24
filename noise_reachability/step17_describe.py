#!/usr/bin/env python3
"""Step 17, descriptive only (README.md, step 17, "Reported, not part of the rule"). Written after the feature run and
before U1 is read; it never fits the critic on the policy's features (that is U1, ``step17_g1b.py``'s).

Three comparisons on the same states (those that passed the feature check) and the same rule code:

- **near, features**: step 16's nearest-neighbour score (M_NEAR = 8 states) with the distance taken in the policy's
  feature space (columns z-scored over the task's kept states, constant columns dropped; no outcome enters);
- **near, contexts**: step 16's own near over the privileged object coordinates (step 16's T2 on these states);
- **critic, contexts**: step 17's critic (d D_PCA, lambda LAM) with step 16's privileged contexts in place of the
  features: what the critic reaches when its input certainly carries the scene.

For each: U1-style value (mean over informative states of AUROC_s(score) - AUROC_s(flat)), D (top TOP_Q ids, score -
ticket), the stratified-null p (near: step 16's T2 null, seed NEAR_SEED, NEAR_N_PERM permutations, so "near, contexts"
reproduces step 16's T2 exactly; critic: step 17's null, seed and permutation count), and paired bootstrap intervals.

**Probe**: do the features carry the scene? Per task, leave-one-state-out ridge from the features to each privileged
object coordinate; Q^2 = 1 - PRESS(ridge) / PRESS(training mean), pooled over the task's coordinates (0: no better
than predicting the training mean). Two inputs: all feature columns (z-scored on the training states; penalty
alpha x number of columns, every alpha of PROBE_ALPHAS reported, none chosen) and the critic's own input (its first
D_PCA principal components, penalty PROBE_PCA_LAM). Chance level: the same probe with the feature rows permuted within
the task (PROBE_N_PERM copies, seed PROBE_SEED), mean and 95th percentile of the pooled Q^2.

The contexts are step 16's file (z-scored over a task's 50 init states). Every use here is invariant to a per-column
affine map (the critic and the probe re-standardise on training states, Q^2 is scale-free, near only uses step 16's
own C), so states 30-49 do not enter any number.

    python noise_reachability/step17_describe.py --features <step17_features.json> \
        --contexts noise_reachability/step16_contexts.json --out step17_describe.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step16_g1 as g16  # noqa: E402
import step17_g1b as g17  # noqa: E402

M_NEAR = g16.M_NEAR
NEAR_N_PERM = g16.N_PERM          # 5000, step 16's T2
NEAR_SEED = g16.PERM_SEED + 1     # 17, step 16's T2 seed
PROBE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
PROBE_PCA_LAM = 1e-3
PROBE_N_PERM = 200
PROBE_SEED = 21
CONTEXTS = Path(__file__).resolve().parent / "step16_contexts.json"


# ---- near in feature space -------------------------------------------------------------------------------------

def zscore_columns(X: np.ndarray) -> np.ndarray:
    """Columns z-scored over the rows; columns that vary less than step17_g1b.MIN_STD are dropped."""
    sd = X.std(0)
    keep = sd > g17.MIN_STD
    if not keep.any():
        raise ValueError("no feature column varies over the task's states")
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]


def near_row(tasks: dict, m: int = M_NEAR, n_perm: int = NEAR_N_PERM, seed: int = NEAR_SEED) -> dict:
    """U1-style value, D and per-state values of near over each task's C (as given), with step 16's T2 null."""
    scores = {t: g16.near_scores(Y, g16.neighbours(C, m)) for t, (Y, C) in tasks.items()}
    parts = g17.u1_parts(tasks, g17.D_PCA, g17.LAM, scores=scores)
    t2 = g16.t2_test(tasks, m=m, n_perm=n_perm, seed=seed)
    if not np.isclose(t2["statistic"], parts["U1"], rtol=0, atol=1e-12):
        raise AssertionError(f"near: T2 {t2['statistic']} != U1-style value {parts['U1']}")
    return {**parts, "p": t2["p"], "null_mean": t2["null_mean"], "n_perm": n_perm, "seed": seed, "m": m}


def critic_row(tasks: dict, d: int | None = None, lam: float | None = None, n_perm: int = g17.N_PERM,
               seed: int = g17.PERM_SEED) -> dict:
    """step17_g1b's rule on the given inputs (d and lambda default to step 17's, read when called)."""
    d = g17.D_PCA if d is None else d
    lam = g17.LAM if lam is None else lam
    r = g17.u1_test(tasks, d, lam, n_perm=n_perm, seed=seed)
    return {**r, "seed": seed}


def with_intervals(r: dict) -> dict:
    return {**r, "U1_ci95": g17.paired_ci(r["u1_states"]), "D_ci95": g17.paired_ci(r["d_states"])}


# ---- probe: do the features carry the object poses? ------------------------------------------------------------

def loo_ridge_predictions(X: np.ndarray, T: np.ndarray, alphas=PROBE_ALPHAS) -> dict:
    """{alpha: (S x J) leave-one-state-out predictions of the targets T from the features X}: ridge on the columns
    z-scored over the training states, penalty alpha x (kept columns), intercept = training mean."""
    S = X.shape[0]
    out = {a: np.empty(T.shape, dtype=float) for a in alphas}
    for s in range(S):
        tr = np.arange(S) != s
        mu, sd = X[tr].mean(0), X[tr].std(0)
        keep = sd > g17.MIN_STD
        Z = (X[tr][:, keep] - mu[keep]) / sd[keep]
        z = (X[s, keep] - mu[keep]) / sd[keep]
        tm = T[tr].mean(0)
        Tc = T[tr] - tm
        e, U = np.linalg.eigh(Z @ Z.T)            # dual ridge: beta = Z^T (Z Z^T + lam I)^-1 Tc
        k = Z @ z
        for a in alphas:
            lam = a * keep.sum()
            out[a][s] = tm + (k @ U) / (e + lam) @ (U.T @ Tc)
    return out


def loo_pca_predictions(X: np.ndarray, T: np.ndarray, d: int | None = None, lam: float = PROBE_PCA_LAM) -> np.ndarray:
    """(S x J) leave-one-state-out predictions of T from the critic's input: step17_g1b's PCA (d components, fitted on
    the training states), then ridge with penalty lam and the training mean as intercept."""
    d = g17.D_PCA if d is None else d
    S = X.shape[0]
    out = np.empty(T.shape, dtype=float)
    for s in range(S):
        tr = np.arange(S) != s
        p = g17.pca_fit(X[tr], d)
        F, f = g17.pca_apply(p, X[tr]), g17.pca_apply(p, X[s:s + 1])[0]
        fm = F.mean(0)
        tm = T[tr].mean(0)
        beta = np.linalg.solve((F - fm).T @ (F - fm) + lam * np.eye(d), (F - fm).T @ (T[tr] - tm))
        out[s] = tm + (f - fm) @ beta
    return out


def press_parts(T: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(PRESS of the predictions, PRESS of the leave-one-out training mean), summed over states and columns."""
    S = T.shape[0]
    loo_mean = (T.sum(0, keepdims=True) - T) / (S - 1)
    return float(((T - pred) ** 2).sum()), float(((T - loo_mean) ** 2).sum())


def q2(T: np.ndarray, pred: np.ndarray) -> float:
    a, b = press_parts(T, pred)
    return 1.0 - a / b


def probe(feat: dict, ctx: dict) -> dict:
    """{setting: {"pooled": Q^2 over all tasks' coordinates, "per_task": {t: Q^2}}} for the ridge alphas and the PCA."""
    settings = {f"ridge alpha {a:g}": None for a in PROBE_ALPHAS} | {f"critic input (pca {g17.D_PCA})": None}
    num = {k: 0.0 for k in settings}
    den = {k: 0.0 for k in settings}
    per = {k: {} for k in settings}
    for t in feat:
        X, T = feat[t], ctx[t]
        preds = {f"ridge alpha {a:g}": v for a, v in loo_ridge_predictions(X, T).items()}
        preds[f"critic input (pca {g17.D_PCA})"] = loo_pca_predictions(X, T)
        for k, pr in preds.items():
            a, b = press_parts(T, pr)
            num[k] += a
            den[k] += b
            per[k][str(t)] = 1.0 - a / b
    return {k: {"pooled": 1.0 - num[k] / den[k], "per_task": per[k]} for k in settings}


def probe_chance(feat: dict, ctx: dict, n_perm: int = PROBE_N_PERM, seed: int = PROBE_SEED) -> dict:
    """{setting: {"mean", "q95"}} of the pooled Q^2 with the feature rows freely permuted within each task."""
    rng = np.random.default_rng(seed)
    vals: dict = {}
    for _ in range(n_perm):
        pf = {t: X[rng.permutation(X.shape[0])] for t, X in feat.items()}
        for k, v in probe(pf, ctx).items():
            vals.setdefault(k, []).append(v["pooled"])
    return {k: {"mean": float(np.mean(v)), "q95": float(np.quantile(v, 0.95))} for k, v in vals.items()}


# ---- inputs and report -----------------------------------------------------------------------------------------

def context_tasks(tasks: dict, states: dict, contexts: dict) -> dict:
    """{task: (Y, C)}: the same states as ``tasks`` with step 16's privileged contexts in place of the features."""
    out = {}
    for t, (Y, _) in tasks.items():
        z = contexts[str(t)]["z"]
        if any(s >= 30 for s in states[t]):
            raise ValueError(f"task {t}: a state outside the map (0-29)")
        C = np.array([z[s] for s in states[t]], dtype=float)
        g16.check_table(Y, C)
        out[t] = (Y, C)
    return out


def describe(tasks: dict, ctx_tasks: dict, near_n_perm: int = NEAR_N_PERM, critic_n_perm: int = g17.N_PERM,
             probe_n_perm: int = PROBE_N_PERM) -> dict:
    """The three comparison rows and the probe. ``tasks`` carries the policy's features, which only near and the
    probe see; the critic is fitted on ``ctx_tasks`` only."""
    zfeat = {t: (Y, zscore_columns(X)) for t, (Y, X) in tasks.items()}
    rows = {"near, features": with_intervals(near_row(zfeat, n_perm=near_n_perm)),
            "near, contexts": with_intervals(near_row(ctx_tasks, n_perm=near_n_perm)),
            "critic, contexts": with_intervals(critic_row(ctx_tasks, n_perm=critic_n_perm))}
    feat = {t: X for t, (_, X) in tasks.items()}
    ctx = {t: C for t, (_, C) in ctx_tasks.items()}
    return {"rows": rows, "probe": probe(feat, ctx), "probe_chance": probe_chance(feat, ctx, n_perm=probe_n_perm),
            "states": {str(t): int(Y.shape[0]) for t, (Y, _) in tasks.items()},
            "settings": {"m": M_NEAR, "near_seed": NEAR_SEED, "near_n_perm": near_n_perm, "d": g17.D_PCA,
                         "lam": g17.LAM, "critic_seed": g17.PERM_SEED, "critic_n_perm": critic_n_perm,
                         "top_q": g17.TOP_Q, "probe_alphas": list(PROBE_ALPHAS), "probe_pca_lam": PROBE_PCA_LAM,
                         "probe_n_perm": probe_n_perm, "probe_seed": PROBE_SEED, "n_boot": g17.N_BOOT,
                         "boot_seed": g17.BOOT_SEED}}


def format_report(d: dict, provenance: dict | None = None) -> str:
    s = d["settings"]
    lines = ["step 17, descriptive (not part of the rule; the critic on the policy's features is U1, not computed here)"]
    if provenance:
        lines.append("inputs: " + ", ".join(f"{k} {v}" for k, v in provenance.items()))
    lines.append(f"states per task {d['states']}; near m {s['m']} ({s['near_n_perm']} permutations, seed "
                 f"{s['near_seed']}); critic d {s['d']}, lambda {s['lam']} ({s['critic_n_perm']} permutations, seed "
                 f"{s['critic_seed']}); intervals: paired bootstrap over states ({s['n_boot']}, seed {s['boot_seed']})")
    lines.append("score | informative states | U1-style (score - ticket) [95%] | p (stratified null) | "
                 f"D (top {s['top_q']}, score - ticket) [95%]")
    for name, r in d["rows"].items():
        lines.append(f"{name} | {r['informative_states']} | {r['U1']:+.4f} [{r['U1_ci95'][0]:+.4f}, "
                     f"{r['U1_ci95'][1]:+.4f}] | {r['p']:.4g} | {r['D']:+.4f} [{r['D_ci95'][0]:+.4f}, "
                     f"{r['D_ci95'][1]:+.4f}]")
    lines.append("per task, U1-style / D / mean AUROC score / mean AUROC ticket:")
    for name, r in d["rows"].items():
        lines.append(f"  {name}: " + "; ".join(
            f"{t} {v['U1']:+.3f} / {v['D']:+.3f} / {v['auroc_critic']:.3f} / {v['auroc_flat']:.3f}"
            if v["U1"] is not None else f"{t} no informative state" for t, v in r["per_task"].items()))
    lines.append(f"probe (leave-one-state-out Q^2, features -> privileged object coordinates; chance: rows permuted, "
                 f"{s['probe_n_perm']} copies, seed {s['probe_seed']}):")
    lines.append("setting | pooled Q^2 | chance mean | chance 95th pct | per task")
    for k, v in d["probe"].items():
        c = d["probe_chance"][k]
        lines.append(f"{k} | {v['pooled']:+.3f} | {c['mean']:+.3f} | {c['q95']:+.3f} | "
                     + ", ".join(f"{t} {q:+.3f}" for t, q in v["per_task"].items()))
    return "\n".join(lines) + "\n"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(g17.STEP16_TABLE))
    ap.add_argument("--features", required=True)
    ap.add_argument("--contexts", default=str(CONTEXTS))
    ap.add_argument("--out")
    ap.add_argument("--near_n_perm", type=int, default=NEAR_N_PERM)
    ap.add_argument("--critic_n_perm", type=int, default=g17.N_PERM)
    ap.add_argument("--probe_n_perm", type=int, default=PROBE_N_PERM)
    a = ap.parse_args(argv)
    tasks, states = g17.load_inputs(a.table, a.features)
    ctx_tasks = context_tasks(tasks, states, json.loads(Path(a.contexts).read_text()))
    d = describe(tasks, ctx_tasks, near_n_perm=a.near_n_perm, critic_n_perm=a.critic_n_perm,
                 probe_n_perm=a.probe_n_perm)
    prov = {"table sha256": sha256(a.table), "features sha256": sha256(a.features),
            "contexts sha256": sha256(a.contexts)}
    d["provenance"] = prov
    text = format_report(d, prov)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(d, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
