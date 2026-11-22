#!/usr/bin/env python3
"""Step 16, descriptive only (not part of the rule; README.md, step 16, "Result"): what T2's pass means in absolute
terms. Per task, the mean over informative states of AUROC_s(flat) and AUROC_s(near), and the spread of T2's
stratified null (the same permutations as ``step16_g1.py``'s report, same seed).

T2 compares near (M_NEAR states) with flat (all the task's other states). Under its null, near averages fewer,
unrelated states and so ranks ids worse than flat: the null mean is negative. A pass therefore says that the scene's
neighbours do better than random neighbours of similar difficulty, not that near beats flat.

    python noise_reachability/step16_describe.py --table <step16_table_parts5.csv> \
        --contexts noise_reachability/step16_contexts.json --out step16_describe.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step16_g1 as g  # noqa: E402


def per_task_auroc(tasks: dict, m: int = g.M_NEAR) -> dict:
    """{task: {"flat", "near", "informative_states"}}: mean AUROC_s over the task's informative states."""
    out = {}
    for t, (Y, C) in tasks.items():
        a_flat = g.auroc_rows(g.flat_scores(Y), Y)
        a_near = g.auroc_rows(g.near_scores(Y, g.neighbours(C, m)), Y)
        ok = ~np.isnan(a_flat)
        out[str(t)] = {"flat": float(a_flat[ok].mean()) if ok.any() else None,
                       "near": float(a_near[ok].mean()) if ok.any() else None,
                       "informative_states": int(ok.sum())}
    return out


def describe(tasks: dict, n_perm: int = g.N_PERM) -> dict:
    r = g.t2_test(tasks, n_perm=n_perm, return_null=True)
    null = r.pop("null")
    r.update(null_sd=float(null.std()), null_q95=float(np.quantile(null, 0.95)), null_max=float(null.max()),
             z=float((r["statistic"] - null.mean()) / null.std()))
    return {"T2": r, "per_task": per_task_auroc(tasks), "n_perm": n_perm}


def format_report(d: dict) -> str:
    t2 = d["T2"]
    lines = [f"step 16, descriptive (not part of the rule), {d['n_perm']} permutations",
             f"T2 {t2['statistic']:.4f}; stratified null mean {t2['null_mean']:.4f}, sd {t2['null_sd']:.4f}, "
             f"95th percentile {t2['null_q95']:.4f}, max {t2['null_max']:.4f}; z {t2['z']:.1f}; p {t2['p']:.4g}",
             "task | informative states | mean AUROC flat | mean AUROC near (m = %d)" % t2["m"]]
    for t, v in d["per_task"].items():
        lines.append(f"{t} | {v['informative_states']} | {v['flat']:.3f} | {v['near']:.3f}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n_perm", type=int, default=g.N_PERM)
    a = ap.parse_args(argv)
    tasks, _ = g.load_table(a.table, json.load(open(a.contexts)))
    d = describe(tasks, n_perm=a.n_perm)
    text = format_report(d)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(d, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
