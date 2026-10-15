"""Compare two evaluate_all.py runs of the same protocol per (task, init state).

Both inputs are per_episode.csv files (task_id, seed, init_state, success) covering the same episodes. Reports
successes per task, the init states solved in one run but never in the other, and two tests of whether the
runs differ:

- exact McNemar test over episodes paired by (task, seed, init state), for the overall success rate;
- permutation test for per-state differences: within each (task, init state) the episodes of both runs are
  shuffled between the runs, and the number of states solved in only one run, and the summed absolute
  difference in successes per state, are compared with the observed values. A small p means the runs differ
  on specific init states more than sampling noise explains (as task 0 did between Colab and local). With
  one episode per state in each run the shuffle cannot change these statistics, so the test is skipped.

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import random
from math import comb


def load(path):
    with open(path) as f:
        return {(int(r["task_id"]), int(r["seed"]), int(r["init_state"])): int(r["success"]) for r in csv.DictReader(f)}


def per_state(episodes):
    """{(task, init_state): [successes in seed order]}."""
    out = {}
    for (t, s, i) in sorted(episodes):
        out.setdefault((t, i), []).append(episodes[(t, s, i)])
    return out


def state_stats(a, b):
    """(states solved in only one run, summed |successes a - successes b| over states)."""
    only = sum(1 for st in a if (sum(a[st]) > 0) != (sum(b[st]) > 0))
    absdiff = sum(abs(sum(a[st]) - sum(b[st])) for st in a)
    return only, absdiff


def mcnemar_exact(n_a_only, n_b_only):
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(n_a_only, n_b_only) + 1)) / 2 ** n)


def compare(a_eps, b_eps, n_perm=20000, seed=0):
    if set(a_eps) != set(b_eps):
        raise ValueError("the two runs do not cover the same (task, seed, init_state) episodes")
    a, b = per_state(a_eps), per_state(b_eps)
    observed = state_stats(a, b)
    if all(len(a[st]) == 1 and len(b[st]) == 1 for st in a):
        n_perm = 0
    rng = random.Random(seed)
    exceed = [0, 0]
    for _ in range(n_perm):
        pa, pb = {}, {}
        for st in a:
            pooled = a[st] + b[st]
            rng.shuffle(pooled)
            pa[st], pb[st] = pooled[:len(a[st])], pooled[len(a[st]):]
        s = state_stats(pa, pb)
        exceed[0] += s[0] >= observed[0]
        exceed[1] += s[1] >= observed[1]
    a_only = sum(1 for k in a_eps if a_eps[k] and not b_eps[k])
    b_only = sum(1 for k in a_eps if b_eps[k] and not a_eps[k])
    per_task = {}
    for t in sorted({t for t, _ in a}):
        states = sorted(i for tt, i in a if tt == t)
        per_task[t] = {
            "a": sum(sum(a[(t, i)]) for i in states), "b": sum(sum(b[(t, i)]) for i in states),
            "n": sum(len(a[(t, i)]) for i in states),
            "only_a": [i for i in states if sum(a[(t, i)]) and not sum(b[(t, i)])],
            "only_b": [i for i in states if sum(b[(t, i)]) and not sum(a[(t, i)])],
        }
    return {
        "n_episodes": len(a_eps), "a_successes": sum(a_eps.values()), "b_successes": sum(b_eps.values()),
        "episodes_a_only": a_only, "episodes_b_only": b_only, "mcnemar_p": mcnemar_exact(a_only, b_only),
        "states_one_run_only": observed[0], "state_absdiff": observed[1], "n_states": len(a),
        "perm_p_one_run_only": (exceed[0] + 1) / (n_perm + 1) if n_perm else None,
        "perm_p_absdiff": (exceed[1] + 1) / (n_perm + 1) if n_perm else None,
        "n_perm": n_perm, "per_task": per_task,
    }


def format_report(r, name_a, name_b):
    lines = [f"A = {name_a}", f"B = {name_b}", "",
             "task | A successes | B successes | init states solved in A but never in B | in B but never in A"]
    for t, p in r["per_task"].items():
        lines.append(f"{t:4d} | {p['a']:3d}/{p['n']} | {p['b']:3d}/{p['n']} | {p['only_a']} | {p['only_b']}")
    n = r["n_episodes"]
    lines += [
        "",
        f"overall: A {r['a_successes']}/{n} ({100 * r['a_successes'] / n:.2f}%), "
        f"B {r['b_successes']}/{n} ({100 * r['b_successes'] / n:.2f}%)",
        f"episodes (paired by task, seed, init state) succeeding only in A: {r['episodes_a_only']}, only in B: "
        f"{r['episodes_b_only']}; exact McNemar p = {r['mcnemar_p']:.4f}",
    ]
    if r["n_perm"]:
        lines += [
            f"init states solved in only one run: {r['states_one_run_only']} of {r['n_states']}; "
            f"permutation p = {r['perm_p_one_run_only']:.4f}",
            f"summed |successes A - successes B| per init state: {r['state_absdiff']}; "
            f"permutation p = {r['perm_p_absdiff']:.4f} ({r['n_perm']} permutations)",
        ]
    else:
        lines.append("per-state permutation test: not applicable with one episode per init state in each run")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="per_episode.csv of run A")
    parser.add_argument("b", help="per_episode.csv of run B")
    parser.add_argument("--n_perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Also write the report to this file")
    args = parser.parse_args()
    report = format_report(compare(load(args.a), load(args.b), args.n_perm, args.seed), args.a, args.b)
    print(report, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
