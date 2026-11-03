"""Exploratory reads of step 9 (not part of the rule fixed in advance): one-sided Fisher p for the two cells that stand
out, and the pooled comparison without task 7. Episodes are treated as independent, so these p values are descriptive.
Counts are read from the run's report JSON.

    python -I -S explore_step9.py <run folder>
"""
import json
import sys
from math import comb
from pathlib import Path


def tail(a, n, b, m, upper=True):
    """One-sided Fisher p of a/n against b/m: P(first count >= a) if upper else P(first count <= a), margins fixed."""
    t = a + b
    ks = range(a, min(t, n) + 1) if upper else range(max(0, t - m), a + 1)
    return sum(comb(n, i) * comb(m, t - i) for i in ks if 0 <= t - i <= m) / comb(n + m, t)


def main():
    r = json.loads((Path(sys.argv[1]) / "step9_report.txt.json").read_text())
    states = {(v["task"], v["state"]): v for v in r["states"]}
    tasks = {int(t): v for t, v in r["tasks"].items()}
    v = states[0, 14]
    print(f"0/14, library above random: {v['library']}/{v['library_episodes']} against "
          f"{v['random']}/{v['random_episodes']}, p {tail(v['library'], v['library_episodes'], v['random'], v['random_episodes']):.2g}")
    v = tasks[7]
    print(f"task 7, library below random: {v['library']}/{v['library_episodes']} against "
          f"{v['random']}/{v['random_episodes']}, p {tail(v['library'], v['library_episodes'], v['random'], v['random_episodes'], upper=False):.2g}")
    rest = [v for t, v in tasks.items() if t != 7]
    a, n = sum(v["library"] for v in rest), sum(v["library_episodes"] for v in rest)
    b, m = sum(v["random"] for v in rest), sum(v["random_episodes"] for v in rest)
    print(f"tasks 0, 4, 6 pooled, library above random: {a}/{n} against {b}/{m}, p {tail(a, n, b, m):.2g}")
    v = tasks[7]
    print(f"task 7 random on states never solved by sampling: {v['random']}/{v['random_episodes']}")


if __name__ == "__main__":
    main()
