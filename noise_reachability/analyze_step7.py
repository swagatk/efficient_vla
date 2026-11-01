#!/usr/bin/env python3
"""Step 7 analysis (README.md, step 7): step 5a's search on the systematic states of tasks 4, 6, 8. Standard library only.

    python3 -S noise_reachability/analyze_step7.py --root <run folder> --out step7_report.txt
    python3 -S noise_reachability/analyze_step7.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step7.py --root <run folder> --hits           # the confirmation jobs

<run folder>/task_<t>/ holds what step 5a's <run folder> holds, for one task.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import (CONTROL_IDS, FAMILIES, FIXED_FAMILIES, MIN_CONTROL, classify,  # noqa: E402
                            confirm_counts, hits_to_confirm, load_family, zero_upper)

SYSTEMATIC = {4: 2, 6: 8, 8: 4}
SCENE_VARIES = 8                                # the stove's pose is not fixed by the init state (README.md, step 7)
CONTROL = {4: 5, 6: 0, 8: 8}
HARNESS_REFERENCE = {4: 14, 6: 20, 8: 15}       # step 0: base, seeds 0-2, of 30
HARNESS_TOLERANCE = 6
ALPHA = 0.05
MIN_SPREAD = 2                                  # states with a repeat hit, of the 3
POLICY_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"


def task_root(root, task) -> Path:
    return Path(root) / f"task_{task}"


def harness_check(root, task) -> dict:
    rows = [r for f in sorted(task_root(root, task).glob("harness/seed*/per_episode_base.csv"))
            for r in csv.DictReader(open(f))]
    keys = {(int(r["seed"]), int(r["init_state"])) for r in rows}
    if keys != {(s, i) for s in (0, 1, 2) for i in range(10)} or len(rows) != 30:
        raise ValueError(f"harness task {task}: {len(rows)} episodes, expected init states 0-9 x seeds 0-2")
    if {int(r["task_id"]) for r in rows} != {task}:
        raise ValueError(f"harness task {task}: rows of another task")
    n = sum(int(r["success"]) for r in rows)
    gap = n - HARNESS_REFERENCE[task]
    return {"base": n, "reference": HARNESS_REFERENCE[task], "gap": gap, "ok": abs(gap) <= HARNESS_TOLERANCE}


def control_counts(root, task) -> tuple[dict, dict]:
    control = {f: sum(load_family(task_root(root, task), f, CONTROL[task], CONTROL_IDS).values()) for f in FAMILIES}
    return control, {f: control[f] >= MIN_CONTROL for f in FAMILIES}


def gate(root, which: str) -> str | None:
    """Why the run must stop, or None. Provenance is checked here too, so a wrong setting stops the run early."""
    check_provenance(root)
    for task in SYSTEMATIC:
        if which == "harness":
            h = harness_check(root, task)
            if not h["ok"]:
                return f"harness failed: task {task}, base {h['base']}/30 against {h['reference']}"
        else:
            control, read = control_counts(root, task)
            if not read["sampled"]:
                return f"sampled control failed: task {task}, {control['sampled']}/{len(CONTROL_IDS)}"
    return None


def check_provenance(root) -> dict:
    """One commit, clean trees and the protocol's settings in every record; each record belongs to its task's folder."""
    root = Path(root)
    records = []
    for task in SYSTEMATIC:
        folder = task_root(root, task)
        files = [folder / "provenance.jsonl"] if (folder / "provenance.jsonl").exists() else []
        for f in files + sorted(folder.glob("harness/seed*/provenance.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["settings"]["task_id"] != task:
                    raise ValueError(f"provenance: task {r['settings']['task_id']} in {f}")
                records.append(r)
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"provenance: commits {commits}")
    for r in records:
        st = r["settings"]
        if r.get("git_dirty") is not False or r.get("git_dirty_noise_reachability") is not False:
            raise ValueError("provenance: a run on a dirty tree")
        if st["n_action_steps"] != 10 or st["max_steps"] != 520:
            raise ValueError(f"provenance: settings {st}")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1:
        raise ValueError(f"provenance: policy paths {paths}")
    if POLICY_REVISION not in next(iter(paths)):
        raise ValueError(f"provenance: not revision {POLICY_REVISION}: {paths}")
    return {"records": len(records), "commit": commits.pop(), "policy_path": paths.pop()}


def fisher_greater(a: int, b: int, n: int) -> float:
    """One-sided exact p that the first of two groups of n is the higher, with a and b successes."""
    total = a + b
    return sum(comb(n, i) * comb(n, total - i) for i in range(a, min(total, n) + 1)) / comb(2 * n, total)


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact two-sided interval on a rate."""
    def solve(target, tail):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if binom_tail(tail, n, mid) < target else (lo, mid)
        return lo
    return (0.0 if k == 0 else solve(alpha / 2, k),
            1.0 if k == n else solve(1 - alpha / 2, k + 1))


def fixed_hits(state: dict) -> int:
    return sum(n for f, n in state["per_family"].items() if f in FIXED_FAMILIES)


def fixed_only(state: dict) -> bool:
    return fixed_hits(state) > 0 and state["per_family"]["sampled"] == 0


def reading(rate_passes: bool, spread: int, n_reached: int, any_fixed_read: bool = True) -> str:
    if not any_fixed_read:
        return "not read: no fixed family was read on any task"
    if rate_passes:
        if spread >= MIN_SPREAD:
            return "extends beyond task 0: pooled over these states, repeat noise succeeds more often than sampled noise"
        return "a higher rate, concentrated on too few states to call general"
    if n_reached:
        return "existence only: hits happened, at a rate not shown above sampling"
    return "unreached at this depth: step 5a stays a result about task 0"


def analyse(root) -> dict:
    harness = {t: harness_check(root, t) for t in SYSTEMATIC}
    out = {"provenance": check_provenance(root), "harness": harness, "control": {}, "family_read": {}, "states": {},
           "unread_hits": {}}
    if not all(h["ok"] for h in harness.values()):
        out["decision"] = "not read: a harness failed"
        return out
    counts = {t: control_counts(root, t) for t in SYSTEMATIC}
    out["control"] = {t: c[0] for t, c in counts.items()}
    out["family_read"] = {t: c[1] for t, c in counts.items()}
    if not all(c[1]["sampled"] for c in counts.values()):
        out["decision"] = "not read: a sampled control failed"
        return out
    pooled, hit_ids = {"repeat": 0, "sampled": 0, "episodes": 0}, set()
    steady = dict(pooled)
    for t, s in SYSTEMATIC.items():
        folder, read = task_root(root, t), counts[t][1]
        wins = {f: load_family(folder, f, s) for f in FAMILIES if read[f]}
        confirmed = {f: confirm_counts(folder, f, s, hits_to_confirm(wins[f])) for f in FIXED_FAMILIES if read[f]}
        for f in FIXED_FAMILIES:
            if not read[f] and sum(load_family(folder, f, s).values()):
                out["unread_hits"].setdefault(f"task {t}", {})[f] = sum(load_family(folder, f, s).values())
        n, hits = sum(len(w) for w in wins.values()), sum(sum(w.values()) for w in wins.values())
        state = {"state": s, "episodes": n, "hits": hits, "per_family": {f: sum(w.values()) for f, w in wins.items()},
                 "confirmed": {f: {str(k): c for k, c in v.items()} for f, v in confirmed.items()},
                 "class": classify(wins, confirmed), "upper_if_zero": zero_upper(n) if hits == 0 else None}
        state["fixed_only"] = fixed_only(state)
        state["repeat_hit_ids"] = sorted(k for k, v in wins.get("repeat", {}).items() if v)
        out["states"][t] = state
        if read["repeat"]:
            pooled["repeat"] += state["per_family"]["repeat"]
            pooled["sampled"] += state["per_family"]["sampled"]
            pooled["episodes"] += len(wins["repeat"])
            hit_ids.update(state["repeat_hit_ids"])
            if t != SCENE_VARIES:
                steady["repeat"] += state["per_family"]["repeat"]
                steady["sampled"] += state["per_family"]["sampled"]
                steady["episodes"] += len(wins["repeat"])
    p = fisher_greater(pooled["repeat"], pooled["sampled"], pooled["episodes"]) if pooled["episodes"] else 1.0
    spread = sum(v["per_family"].get("repeat", 0) > 0 for v in out["states"].values())
    any_fixed = any(c[1][f] for c in counts.values() for f in FIXED_FAMILIES)
    n = pooled["episodes"]
    n_reached = sum(fixed_hits(v) > 0 for v in out["states"].values())
    out["rate"] = {**pooled, "p": p, "passes": p < ALPHA, "repeat_noises_that_hit": len(hit_ids),
                   "repeat_interval": clopper_pearson(pooled["repeat"], n) if n else None,
                   "sampled_interval": clopper_pearson(pooled["sampled"], n) if n else None}
    out["rate_without_task_8"] = {**steady, "p": fisher_greater(steady["repeat"], steady["sampled"], steady["episodes"])
                                  if steady["episodes"] else 1.0}
    out["spread"] = spread
    out["fixed_only_states"] = sum(v["fixed_only"] for v in out["states"].values())
    out["decision"] = reading(p < ALPHA, spread, n_reached, any_fixed)
    return out


def format_report(r: dict) -> str:
    lines = [f"step 7: deep noise on tasks 4, 6, 8; commit {r['provenance']['commit']}"]
    for t, h in r["harness"].items():
        lines.append(f"harness task {t}: base {h['base']}/30 against {h['reference']} (gap {h['gap']:+d}, "
                     f"{'ok' if h['ok'] else 'FAILED'})")
    for t, c in r["control"].items():
        lines.append(f"control task {t}, state {CONTROL[t]}, of 35: " + ", ".join(
            f"{f} {n}{'' if r['family_read'][t][f] else ' (not read)'}" for f, n in c.items()))
    for t, v in r["states"].items():
        lines.append(f"task {t} state {v['state']}: {v['hits']}/{v['episodes']} {v['per_family']} confirmed "
                     f"{v['confirmed']} -> {v['class']}{', fixed-only' if v['fixed_only'] else ''}"
                     + (f" (one-sided 95% bound on the mixed rate {100 * v['upper_if_zero']:.1f}%)"
                        if v["upper_if_zero"] else ""))
    if r["unread_hits"]:
        lines.append(f"hits of families not read (they did happen): {r['unread_hits']}")
    if "rate" in r:
        q = r["rate"]
        lines.append(f"rate: repeat {q['repeat']}/{q['episodes']} against sampled {q['sampled']}/{q['episodes']}, "
                     f"one-sided Fisher p {q['p']:.3g}: {'passes' if q['passes'] else 'does not pass'}")
        if q["episodes"]:
            lines.append("95% intervals: repeat {:.1f}-{:.1f}%, sampled {:.1f}-{:.1f}%; {} distinct repeat noises hit".format(
                *(100 * x for x in q["repeat_interval"] + q["sampled_interval"]), q["repeat_noises_that_hit"]))
        s = r["rate_without_task_8"]
        lines.append(f"secondary, without task 8 (stove not fixed): repeat {s['repeat']}/{s['episodes']} against sampled "
                     f"{s['sampled']}/{s['episodes']}, one-sided Fisher p {s['p']:.3g}")
        lines.append(f"states with a repeat hit: {r['spread']} of {len(r['states'])} (required {MIN_SPREAD}); "
                     f"fixed-only (descriptive): {r['fixed_only_states']}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--hits", action="store_true", help="print the confirmation jobs: task family state noise id")
    p.add_argument("--gate", choices=("harness", "control"), help="exit 1 if the run fails this gate")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.gate:
        reason = gate(a.root, a.gate)
        print(f"STOP: {reason}" if reason else f"gate {a.gate} passed")
        return 1 if reason else 0
    if a.hits:
        for t, s in SYSTEMATIC.items():
            read = control_counts(a.root, t)[1]
            for f in (f for f in FIXED_FAMILIES if read[f]):
                for k in hits_to_confirm(load_family(task_root(a.root, t), f, s)):
                    print(t, f, s, k)
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
