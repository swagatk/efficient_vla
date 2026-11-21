#!/usr/bin/env python3
"""Step 16's two tests (README.md, step 16): is a constant noise's success shared across a task's states (T1), and is
which noise suits a state predictable from the scene (T2)? Run locally on the finished crossed map, never on Kaggle.

The data of one task is a complete 0/1 table ``Y`` (states x noise ids, every state run with the same ids) and a context
``C`` (states x d): the varying object coordinates of the state's LIBERO init-state vector (the objects' x-y
positions; the robot's joints are left out; privileged information, not what the policy sees), z-scored within the
task over its 50 init states.

Both tests rank the ids *within* one state, so a state's difficulty cannot enter: for state s and id k,

- ``flat(s, k)``: the mean success of id k over the task's other states (a per-task "golden ticket" score);
- ``near(s, k)``: the mean success of id k over the ``M_NEAR`` other states nearest to s in context (Euclidean).

A state is *informative* if it has at least one success and one failure. AUROC_s(score) is the Mann-Whitney AUROC of
the score for the state's successes against its failures (midranks, so tied scores count one half).

- **T1** = mean over informative states (tasks pooled) of AUROC_s(flat) - 0.5. Null: each state's outcomes are
  permuted over its ids, independently per state, which breaks any id effect shared between states.
- **T2** = mean over informative states of AUROC_s(near) - AUROC_s(flat). Null: within each task the context rows are
  permuted over states of similar difficulty only (strata of STRATUM_SIZE states by success count), which keeps every
  outcome and the link between scene and difficulty, and breaks the link between scene and which ids succeed. A free
  permutation would also break scene -> difficulty, and since a state's near neighbours then share its difficulty, T2
  could pass with no scene-dependent choice of noise at all (found in review). In a model logit = a_s + b_k the success
  count is sufficient for a_s, so given the counts the outcomes do not depend on the scene: within equal counts the
  permutation is exact, within strata of similar counts approximate (calibration/step16_power.py checks its level).

Each p-value is (1 + #{permuted >= observed}) / (1 + N_PERM), one-sided; the two are Holm-corrected at ALPHA.

    python noise_reachability/step16_g1.py --table <analyze_step16 table.csv> --contexts step16_contexts.json --out r.txt
    python noise_reachability/step16_g1.py --write_contexts noise_reachability/step16_contexts.json   # needs LIBERO
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

STRATUM_SIZE = 5           # T2's null permutes contexts within strata of this many states of similar success count
FIRST_OBJECT_COLUMN = 10   # init-state vector: 0 time, 1-7 the arm's joints, 8-9 the gripper; objects from 10 on
TASKS = (0, 4, 6, 7)       # the protocol's table: these tasks, ids 10000-10049, states within 0-29
IDS = tuple(range(10000, 10050))
MAP_STATES = tuple(range(30))
M_NEAR = 8                 # neighbours of a state in context (chosen on calibration/step16_power.py's simulated data,
#                            before any step-16 episode)
N_PERM = 5000
ALPHA = 0.05
PERM_SEED = 16
CONTEXT_MIN_STD = 1e-9     # a coordinate that varies less than this over the task's 50 init states is dropped


# ---- scores and AUROC ------------------------------------------------------------------------------------------

def flat_scores(Y: np.ndarray) -> np.ndarray:
    """flat[s, k]: mean of Y[s', k] over s' != s."""
    S = Y.shape[0]
    if S < 2:
        raise ValueError("flat scores need at least two states")
    return (Y.sum(0, keepdims=True) - Y) / (S - 1)


def neighbours(C: np.ndarray, m: int = M_NEAR) -> np.ndarray:
    """(S, m) indices of each state's m nearest other states in C (Euclidean; ties broken by the lower index)."""
    S = C.shape[0]
    if not 1 <= m <= S - 1:
        raise ValueError(f"m={m} needs between 1 and {S - 1} neighbours")
    d = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    d[np.arange(S), np.arange(S)] = np.inf
    return np.argsort(d, axis=1, kind="stable")[:, :m]


def near_scores(Y: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """near[s, k]: mean of Y[s', k] over the neighbours s' of s."""
    return Y[nb].mean(1)


def midranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n along the last axis, tied values sharing their mean rank."""
    from scipy.stats import rankdata

    return rankdata(x, method="average", axis=-1)


def auroc_rows(score: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """AUROC of each row's score for its 1s against its 0s; NaN for a row without both."""
    pos = Y.sum(1)
    neg = Y.shape[1] - pos
    r = midranks(score)
    u = (r * Y).sum(1) - pos * (pos + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        a = u / (pos * neg)
    a[(pos == 0) | (neg == 0)] = np.nan
    return a


# ---- the statistics --------------------------------------------------------------------------------------------

def check_table(Y: np.ndarray, C: np.ndarray | None = None) -> None:
    if Y.ndim != 2 or not np.isin(Y, (0, 1)).all():
        raise ValueError("Y must be a complete 0/1 table (states x ids)")
    if C is not None and (C.ndim != 2 or C.shape[0] != Y.shape[0] or not np.isfinite(C).all()):
        raise ValueError(f"C must be finite with one row per state: {C.shape} for {Y.shape[0]} states")


def t1_stat(tasks: dict) -> tuple[float, int]:
    """(T1, informative states) over tasks {t: (Y, C)}."""
    vals = []
    for Y, _ in tasks.values():
        a = auroc_rows(flat_scores(Y), Y)
        vals.extend(a[~np.isnan(a)] - 0.5)
    if not vals:
        raise ValueError("no informative state")
    return float(np.mean(vals)), len(vals)


def t2_stat(tasks: dict, m: int = M_NEAR, nbs: dict | None = None) -> tuple[float, int]:
    """(T2, informative states); ``nbs`` gives each task's neighbour table (else from its C)."""
    vals = []
    for t, (Y, C) in tasks.items():
        nb = nbs[t] if nbs is not None else neighbours(C, m)
        a_near = auroc_rows(near_scores(Y, nb), Y)
        a_flat = auroc_rows(flat_scores(Y), Y)
        ok = ~np.isnan(a_flat)
        vals.extend(a_near[ok] - a_flat[ok])
    if not vals:
        raise ValueError("no informative state")
    return float(np.mean(vals)), len(vals)


def perm_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def t1_test(tasks: dict, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    obs, n = t1_stat(tasks)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = {t: (rng.permuted(Y, axis=1), C) for t, (Y, C) in tasks.items()}
        null[i] = t1_stat(shuffled)[0]
    return {"statistic": obs, "informative_states": n, "p": perm_p(obs, null), "null_mean": float(null.mean())}


def strata(Y: np.ndarray, size: int = STRATUM_SIZE) -> list[np.ndarray]:
    """The task's states in consecutive groups of ``size`` by success count (ties by index); the last group takes
    the remainder when fewer than ``size`` states are left over."""
    order = np.lexsort((np.arange(Y.shape[0]), Y.sum(1)))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    if len(groups) > 1 and len(groups[-1]) < size:
        last = groups.pop()
        groups[-1] = np.concatenate([groups[-1], last])
    return groups


def stratified_permutation(rng, groups: list[np.ndarray], n: int) -> np.ndarray:
    """perm with perm[s] in the same group as s, uniform within each group."""
    perm = np.arange(n)
    for grp in groups:
        perm[grp] = rng.permutation(grp)
    return perm


def t2_test(tasks: dict, m: int = M_NEAR, n_perm: int = N_PERM, seed: int = PERM_SEED + 1,
            stratum_size: int | None = STRATUM_SIZE) -> dict:
    """``stratum_size=None`` permutes freely (the invalid null, kept only to show the confound in calibration)."""
    obs, n = t2_stat(tasks, m)
    rng = np.random.default_rng(seed)
    base_nb = {t: neighbours(C, m) for t, (_, C) in tasks.items()}
    groups = {t: (strata(Y, stratum_size) if stratum_size else [np.arange(Y.shape[0])]) for t, (Y, _) in tasks.items()}
    null = np.empty(n_perm)
    for i in range(n_perm):
        nbs = {}
        for t in tasks:
            perm = stratified_permutation(rng, groups[t], len(base_nb[t]))   # state s gets the context of perm[s]
            inv = np.argsort(perm)
            nbs[t] = inv[base_nb[t][perm]]
        null[i] = t2_stat(tasks, m, nbs)[0]
    return {"statistic": obs, "informative_states": n, "p": perm_p(obs, null), "null_mean": float(null.mean()), "m": m,
            "stratum_size": stratum_size}


def holm(ps: dict, alpha: float = ALPHA) -> dict:
    """{name: passes} under Holm's step-down at level alpha."""
    out, stop = {}, False
    for i, (name, p) in enumerate(sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))):
        stop = stop or p >= alpha / (len(ps) - i)
        out[name] = not stop
    return out


def reading(passes: dict) -> str:
    if passes["T2"]:
        return ("which noise suits a state is predictable from the scene: ENEI goes ahead, next with the VLA's own "
                "features of the initial observation in place of the privileged state")
    if passes["T1"]:
        return "a noise's quality is shared across a task's states, but not shown to depend on the scene: a per-task ticket is the method's ceiling"
    return "good noise is not shown to be shared across states or predictable from the scene"


def per_task(tasks: dict, ids: dict | None = None) -> dict:
    """Reported, not part of the rule: per task the successes, informative states, T1 and T2 statistics, and the best
    global ticket (the id column with the most successes; the lowest index on a tie) with its count."""
    out = {}
    for t, (Y, C) in tasks.items():
        one = {t: (Y, C)}
        inf = int((~np.isnan(auroc_rows(flat_scores(Y), Y))).sum())
        col = Y.sum(0)
        best = int(np.argmax(col))
        out[str(t)] = {"successes": int(Y.sum()), "episodes": int(Y.size), "informative_states": inf,
                       "T1": t1_stat(one)[0] if inf else None, "T2": t2_stat(one)[0] if inf else None,
                       "best_ticket": (ids[t][best] if ids else best), "best_ticket_successes": int(col[best]),
                       "states": int(Y.shape[0])}
    return out


def analyse(tasks: dict, n_perm: int = N_PERM, ids: dict | None = None) -> dict:
    for Y, C in tasks.values():
        check_table(Y, C)
    r1, r2 = t1_test(tasks, n_perm), t2_test(tasks, n_perm=n_perm)
    passes = holm({"T1": r1["p"], "T2": r2["p"]})
    return {"T1": {**r1, "passes": passes["T1"]}, "T2": {**r2, "passes": passes["T2"]}, "reading": reading(passes),
            "per_task": per_task(tasks, ids),
            "states": {str(t): int(Y.shape[0]) for t, (Y, _) in tasks.items()},
            "ids": {str(t): int(Y.shape[1]) for t, (Y, _) in tasks.items()}, "n_perm": n_perm}


# ---- inputs ----------------------------------------------------------------------------------------------------

def load_table(path, contexts: dict) -> tuple[dict, dict]:
    """({task: (Y, C)}, {task: noise ids in column order}) from analyze_step16's table (task_id, init_state, noise_id, success) and the contexts file."""
    cells: dict = {}
    for r in csv.DictReader(open(path)):
        key = (int(r["task_id"]), int(r["init_state"]), int(r["noise_id"]))
        if key in cells:
            raise ValueError(f"duplicate row {key}")
        cells[key] = int(r["success"])
    if {k[0] for k in cells} != set(TASKS):
        raise ValueError(f"tasks {sorted({k[0] for k in cells})}, expected {list(TASKS)}")
    tasks, id_cols = {}, {}
    for t in sorted({k[0] for k in cells}):
        states = sorted({k[1] for k in cells if k[0] == t})
        ids = sorted({k[2] for k in cells if k[0] == t})
        missing = [(s, k) for s in states for k in ids if (t, s, k) not in cells]
        if missing:
            raise ValueError(f"task {t}: {len(missing)} missing cells, e.g. {missing[:3]}")
        if ids != list(IDS) or not set(states) <= set(MAP_STATES) or len(states) <= M_NEAR:
            raise ValueError(f"task {t}: ids {ids[0]}-{ids[-1]} ({len(ids)}) and states {states} are not the protocol's "
                             f"(ids {IDS[0]}-{IDS[-1]}, all five parts; more than {M_NEAR} of states 0-29)")
        Y = np.array([[cells[(t, s, k)] for k in ids] for s in states])
        ctx = contexts[str(t)]
        C = np.array([ctx["z"][s] for s in states])
        tasks[t] = (Y, C)
        id_cols[t] = ids
    return tasks, id_cols


def task_contexts(x: np.ndarray) -> dict:
    """The z-scored varying object coordinates (columns from FIRST_OBJECT_COLUMN on) of one task's (50, D) init-state
    array, with the kept columns."""
    std = x.std(0)
    keep = np.flatnonzero(std > CONTEXT_MIN_STD)
    keep = keep[keep >= FIRST_OBJECT_COLUMN]
    z = (x[:, keep] - x[:, keep].mean(0)) / std[keep]
    return {"columns": keep.tolist(), "z": z.tolist()}


def write_contexts(out, tasks=(0, 4, 6, 7)) -> None:
    """Contexts of every init state of ``tasks`` from LIBERO's installed init-state files (trusted package data)."""
    import hashlib
    import os

    import torch
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    res = {}
    for t in tasks:
        task = suite.tasks[t]
        p = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        x = np.asarray(torch.load(p, weights_only=False), dtype=float)   # LIBERO's own pickled numpy array
        res[str(t)] = {"file": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                       "shape": list(x.shape), **task_contexts(x)}
    Path(out).write_text(json.dumps(res) + "\n")


def format_report(r: dict) -> str:
    lines = [f"step 16 tests over tasks {r['states']} states and {r['ids']} ids, {r['n_perm']} permutations"]
    for k in ("T1", "T2"):
        v = r[k]
        extra = f", m = {v['m']}, strata of {v['stratum_size']}" if k == "T2" else ""
        lines.append(f"{k}: statistic {v['statistic']:.4f} over {v['informative_states']} informative states (null mean "
                     f"{v['null_mean']:.4f}{extra}), one-sided p {v['p']:.4g}; {'passes' if v['passes'] else 'does not pass'} (Holm, {ALPHA})")
    for t, v in r.get("per_task", {}).items():
        stats = ", ".join(f"{k} {v[k]:.4f}" for k in ("T1", "T2") if v[k] is not None) or "no informative state"
        lines.append(f"task {t} (reported, not tested): {v['successes']} of {v['episodes']} episodes, "
                     f"{v['informative_states']} of {v['states']} states informative, {stats}; best ticket id "
                     f"{v['best_ticket']} ({v['best_ticket_successes']} of {v['states']} states)")
    lines.append(f"reading: {r['reading']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table")
    ap.add_argument("--contexts")
    ap.add_argument("--write_contexts")
    ap.add_argument("--out")
    ap.add_argument("--n_perm", type=int, default=N_PERM, help="the protocol's is 5000; fewer only for tests")
    a = ap.parse_args(argv)
    if a.write_contexts:
        write_contexts(a.write_contexts)
        return 0
    tasks, ids = load_table(a.table, json.load(open(a.contexts)))
    r = analyse(tasks, n_perm=a.n_perm, ids=ids)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
