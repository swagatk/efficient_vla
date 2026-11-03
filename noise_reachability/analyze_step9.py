#!/usr/bin/env python3
"""Step 9 analysis (README.md, step 9): do noises that solved one resistant state help on other states of their task?
Standard library only.

    python3 -S noise_reachability/analyze_step9.py --root <run folder> --out step9_report.txt
    python3 -S noise_reachability/analyze_step9.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step9.py --root <run folder> --jobs           # task state arm ids, per job

<run folder>/task_<t>/ holds harness/seed<s>/, scene/ and test/{library,random}/.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import load_family  # noqa: E402
from analyze_step7 import ALPHA, clopper_pearson  # noqa: E402
from analyze_step8 import TASKS, check_provenance, harness_check, scene_of, task_root  # noqa: E402

HELD_OUT = {0: (14, 27), 4: (18, 26), 6: (14, 20), 7: (38, 47)}   # step 8's eligible states it did not test
# Every repeat hit of an earlier step on a state labelled systematic or eligible, per task, by the state it was found
# on (step 5a ran the same ids on five states, so an id can appear under several).
LIBRARY_SOURCES = {
    0: {"step 5a, 0/2": (6,), "step 5a, 0/5": (30,), "step 5a, 0/6": (1, 10, 12, 30, 41, 57, 60),
        "step 5a, 0/7": (15, 30, 55), "step 5a, 0/8": (15, 23, 35, 38, 44, 46, 53, 62),
        "step 8, 0/11": (1055, 1130, 1148)},
    4: {"step 7, 4/2": (23, 27, 49), "step 8, 4/11": (1239, 1276, 1286)},
    6: {"step 7, 6/8": (30, 42, 69), "step 8, 6/8": (125, 150, 164, 167, 190, 256, 293, 313, 350, 381),
        "step 8, 6/12": (1324, 1334, 1356, 1368, 1389, 1396, 1439)},
    7: {"step 8, 7/33": (1459, 1467, 1473, 1483, 1484, 1492, 1509, 1510, 1518, 1519, 1522, 1536, 1538, 1540, 1545,
                         1548, 1557, 1561, 1562, 1567, 1581, 1586, 1595, 1596, 1599)},
}
LIBRARY = {t: tuple(sorted({k for ids in v.values() for k in ids})) for t, v in LIBRARY_SOURCES.items()}
RANDOM_IDS = tuple(range(6000, 6050))
ARMS = ("library", "random")


def arm_ids(task: int, arm: str) -> tuple[int, ...]:
    return LIBRARY[task] if arm == "library" else RANDOM_IDS


def gate(root) -> str | None:
    """Why the run must stop after the harness, or None. Provenance is checked too, so a wrong setting stops early."""
    check_provenance(root)
    for t in TASKS:
        h = harness_check(root, t)
        if not h["ok"]:
            return f"harness failed: task {t}, base {h['base']}/30 against {h['reference']}"
    return None


def read_states(root) -> list[tuple[int, int]]:
    """Held-out states that passed the scene check; refuses a state not yet checked."""
    out = []
    for t in TASKS:
        for s in HELD_OUT[t]:
            same = scene_of(root, t, s)
            if same is None:
                raise ValueError(f"scene check of {t}/{s} is missing")
            if same:
                out.append((t, s))
    return out


def jobs(root) -> list[tuple[int, int, str, tuple[int, ...]]]:
    """(task, state, arm, noise ids) for every held-out state read."""
    return [(t, s, arm, arm_ids(t, arm)) for t, s in read_states(root) for arm in ARMS]


def subset_sum_counts(values: list[int], k: int) -> dict[int, int]:
    """{s: number of k-element subsets of the positions of ``values`` whose values sum to s}."""
    table = {(0, 0): 1}                    # (subset size, sum) -> count
    for v in values:
        nxt = dict(table)                  # subsets without this position
        for (size, s), c in table.items():
            if size < k:
                key = (size + 1, s + v)
                nxt[key] = nxt.get(key, 0) + c
        table = nxt
    return {s: c for (size, s), c in table.items() if size == k}


def permutation_p(per_task: dict) -> float:
    """One-sided exact p of the library's total successes under relabelling noise ids within each task.

    per_task[t] = (library per-noise totals, random per-noise totals), each noise's total summed over the task's states
    read, so a noise's outcomes on its task's states move together. Under the null every split of a task's noises into
    a library of the same size is equally likely, independently across tasks; the observed split is one of them."""
    observed = sum(sum(lib) for lib, _ in per_task.values())
    dist, total = {0: 1}, 1
    for lib, ran in per_task.values():
        counts = subset_sum_counts(list(lib) + list(ran), len(lib))
        n = comb(len(lib) + len(ran), len(lib))
        assert sum(counts.values()) == n
        new = {}
        for a, ca in dist.items():
            for b, cb in counts.items():
                new[a + b] = new.get(a + b, 0) + ca * cb
        dist, total = new, total * n
    return sum(c for s, c in dist.items() if s >= observed) / total


def reading(harness_ok: bool, n_read: int, p: float | None) -> str:
    if not harness_ok:
        return "not read: a harness failed"
    if n_read == 0:
        return "not read: no held-out state passed the scene check"
    if p < ALPHA:
        return "hit noises carry over to other resistant states of their task; a retrieval method is worth building"
    return "not shown: no carry-over found; the test had about 0.25-0.55 power for the hypothesised 2x gain"


def source_of(task: int, k: int) -> str:
    """Every earlier state this noise was found on."""
    return "; ".join(src for src, ids in LIBRARY_SOURCES[task].items() if k in ids)


def analyse(root) -> dict:
    harness = {t: harness_check(root, t) for t in TASKS}
    out = {"provenance": check_provenance(root), "harness": harness, "states": [], "tasks": {}}
    if not all(h["ok"] for h in harness.values()):
        out["decision"] = reading(False, 0, None)
        return out
    states = read_states(root)
    out["scene_failed"] = [[t, s] for t in TASKS for s in HELD_OUT[t] if (t, s) not in states]
    wins = {}
    for t, s in states:
        for arm in ARMS:
            wins[t, s, arm] = load_family(task_root(root, t) / "test" / arm, "repeat", s, arm_ids(t, arm))
        row = {"task": t, "state": s}
        for arm in ARMS:
            k, n = sum(wins[t, s, arm].values()), len(arm_ids(t, arm))
            row[arm] = k
            row[f"{arm}_episodes"] = n
            row[f"{arm}_interval"] = clopper_pearson(k, n)
        out["states"].append(row)
    per_task = {}
    for t in TASKS:
        mine = [s for tt, s in states if tt == t]
        if not mine:
            continue
        lib = [sum(wins[t, s, "library"][k] for s in mine) for k in LIBRARY[t]]
        ran = [sum(wins[t, s, "random"][k] for s in mine) for k in RANDOM_IDS]
        per_task[t] = (lib, ran)
        hit = {k: [s for s in mine if wins[t, s, "library"][k]] for k in LIBRARY[t]}
        out["tasks"][t] = {"states": mine, "library": sum(lib), "library_episodes": len(lib) * len(mine),
                           "random": sum(ran), "random_episodes": len(ran) * len(mine),
                           "library_hits": [[k, source_of(t, k), v] for k, v in hit.items() if v],
                           "library_ids_on_every_state": sum(len(v) == len(mine) > 1 for v in hit.values())}
    out["p"] = permutation_p(per_task) if per_task else None
    lib_n = sum(v["library_episodes"] for v in out["tasks"].values())
    ran_n = sum(v["random_episodes"] for v in out["tasks"].values())
    lib_k = sum(v["library"] for v in out["tasks"].values())
    ran_k = sum(v["random"] for v in out["tasks"].values())
    out["pooled"] = {"library": lib_k, "library_episodes": lib_n, "random": ran_k, "random_episodes": ran_n}
    if lib_n and ran_n:
        out["pooled"]["library_interval"] = clopper_pearson(lib_k, lib_n)
        out["pooled"]["random_interval"] = clopper_pearson(ran_k, ran_n)
        out["pooled"]["fisher_cross_check"] = fisher_unequal(lib_k, lib_n, ran_k, ran_n)
    out["decision"] = reading(True, len(states), out["p"])
    return out


def fisher_unequal(a: int, n: int, b: int, m: int) -> float:
    """One-sided exact p that a/n is at least this far above b/m (hypergeometric tail); a cross-check only."""
    t = a + b
    return sum(comb(n, i) * comb(m, t - i) for i in range(a, min(t, n) + 1) if t - i <= m) / comb(n + m, t)


def format_report(r: dict) -> str:
    lines = [f"step 9: do hit noises carry over to other resistant states of their task? commit {r['provenance']['commit']}"]
    for t, h in r["harness"].items():
        lines.append(f"harness task {t}: base {h['base']}/30 against {h['reference']} (gap {h['gap']:+d}, "
                     f"{'ok' if h['ok'] else 'FAILED'})")
    if "scene_failed" in r:
        lines.append(f"held-out states that failed the scene check: {r['scene_failed']}")
    for v in r["states"]:
        a, b, c, d = (100 * x for x in v["library_interval"] + v["random_interval"])
        lines.append(f"task {v['task']} state {v['state']}: library {v['library']}/{v['library_episodes']} ({a:.1f}-{b:.1f}%), "
                     f"random {v['random']}/{v['random_episodes']} ({c:.1f}-{d:.1f}%)")
    for t, v in r["tasks"].items():
        lines.append(f"task {t}: library {v['library']}/{v['library_episodes']}, random {v['random']}/{v['random_episodes']}; "
                     f"library hits (id, found on, hit here) {v['library_hits']}; "
                     f"library ids that hit every state read {v['library_ids_on_every_state']}")
    if r.get("p") is not None:
        q = r["pooled"]
        a, b, c, d = (100 * x for x in q["library_interval"] + q["random_interval"])
        lines.append(f"pooled: library {q['library']}/{q['library_episodes']} ({a:.1f}-{b:.1f}%), random {q['random']}/"
                     f"{q['random_episodes']} ({c:.1f}-{d:.1f}%); Fisher cross-check, episodes as independent and tasks pooled (anti-conservative) {q['fisher_cross_check']:.3g}")
        lines.append(f"test: one-sided exact permutation p {r['p']:.3g} (within task, over noise ids), "
                     f"{'passes' if r['decision'].startswith('hit') else 'does not pass'} at {ALPHA}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--gate", choices=("harness",), help="exit 1 if the run fails this gate")
    p.add_argument("--jobs", action="store_true", help="print task, state, arm, comma-separated noise ids per job")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.gate:
        reason = gate(a.root)
        print(f"STOP: {reason}" if reason else f"gate {a.gate} passed")
        return 1 if reason else 0
    if a.jobs:
        for t, s, arm, ids in jobs(a.root):
            print(t, s, arm, ",".join(map(str, ids)))
        return 0
    r = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
