#!/usr/bin/env python3
"""Step 5a analysis and the throughput pilot's choice (README.md, step 5a). Standard library only.

    python3 -S noise_reachability/analyze_step5a.py --root <run folder> --out step5a_report.txt
    python3 -S noise_reachability/analyze_step5a.py --pilot <pilot folder> --cpus 12 --out pilot.json
    python3 -S noise_reachability/analyze_step5a.py --root <run folder> --gate harness   # exit 1 stops the run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SYSTEMATIC_STATES = (2, 5, 6, 7, 8)
CONTROL_STATE = 9
FAMILIES = ("sampled", "full", "repeat")
FIXED_FAMILIES = ("full", "repeat")
NOISE_IDS = tuple(range(70))
CONTROL_IDS = tuple(range(35))
EPISODE_SEED_BASE = 1000
CONFIRM_JITTER_SEEDS = (5000, 5001, 5002, 5003)
MAX_CONFIRM_PER_CELL = 3          # hits confirmed per (fixed family, systematic state)
MIN_CONFIRMED = 2                 # of the 4 jittered noises, for a hit to count as a robust reach
MIN_CONTROL = 3                   # control successes of 35 for a family to be read
HARNESS_REFERENCE = 7             # step 0: base, task 0, seeds 0-2, 7/30
HARNESS_TOLERANCE = 6
MIN_STREAM_GAIN = 1.15            # a higher stream count must add this factor of throughput
ALPHA = 0.05


def read_rows(path) -> dict[int, int]:
    """{episode seed: success} of one result file; refuses a repeated seed."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s, v = int(r["seed"]), int(r["success"])
            if s in out:
                raise ValueError(f"{path}: seed {s} twice")
            if v not in (0, 1):
                raise ValueError(f"{path}: success {v} at seed {s}")
            out[s] = v
    return out


def zero_upper(n: int, alpha: float = ALPHA) -> float:
    """One-sided upper confidence bound on the success rate after 0 successes in n episodes."""
    return 1 - alpha ** (1 / n)


def load_family(root, family, state, ids=NOISE_IDS) -> dict[int, int]:
    """{noise id: success} for one family and state; refuses a missing or unexpected episode."""
    rows = read_rows(Path(root) / f"per_episode_{family}_state{state}.csv")
    expected = {EPISODE_SEED_BASE + k for k in ids}
    if set(rows) != expected:
        raise ValueError(f"{family} state {state}: {len(rows)} episodes, expected {len(expected)}; finish the run")
    return {s - EPISODE_SEED_BASE: v for s, v in rows.items()}


def hits_to_confirm(wins: dict[int, int], cap: int = MAX_CONFIRM_PER_CELL) -> list[int]:
    """The first ``cap`` noise ids that succeeded."""
    return sorted(k for k, v in wins.items() if v)[:cap]


def confirm_counts(root, family, state, ids) -> dict[int, int]:
    """{noise id: successes of the 4 jittered noises}; refuses an unfinished confirmation."""
    out = {}
    for k in ids:
        rows = read_rows(Path(root) / f"confirm_{family}_n{k}_state{state}.csv")
        if set(rows) != set(CONFIRM_JITTER_SEEDS):
            raise ValueError(f"confirmation of {family} {k} state {state} is unfinished")
        out[k] = sum(rows.values())
    return out


def classify(wins_by_family: dict, confirmed: dict) -> str:
    """One state: robust (a fixed noise held up under jitter), sampled, fragile (fixed hits only) or unreached."""
    if any(c >= MIN_CONFIRMED for per in confirmed.values() for c in per.values()):
        return "robust"
    if sum(wins_by_family.get("sampled", {}).values()):
        return "sampled"
    if any(sum(w.values()) for w in wins_by_family.values()):
        return "fragile"
    return "unreached"


def reading(classes: dict[int, str]) -> str:
    n = sum(c != "unreached" for c in classes.values())
    if n == 0:
        return "unreached at this depth: no systematic state reached by any read family"
    if n >= 3:
        return f"mostly reachable: {n} of {len(classes)} systematic states reached; 'systematic' means rare"
    return f"mixed: {n} of {len(classes)} systematic states reached"


def harness_check(path) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    keys = {(int(r["seed"]), int(r["init_state"])) for r in rows}
    if keys != {(s, i) for s in (0, 1, 2) for i in range(10)} or len(rows) != 30:
        raise ValueError(f"harness: {len(rows)} episodes, expected init states 0-9 x seeds 0-2")
    n = sum(int(r["success"]) for r in rows)
    return {"base": n, "gap": n - HARNESS_REFERENCE, "ok": abs(n - HARNESS_REFERENCE) <= HARNESS_TOLERANCE}


def check_provenance(root) -> dict:
    """One commit, clean trees and the protocol's settings in every record of the run (not the pilot)."""
    root = Path(root)
    records = [json.loads(line) for f in [root / "provenance.jsonl", *sorted(root.glob("harness/seed*/provenance.jsonl"))]
               for line in open(f)]
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"provenance: commits {commits}")
    for r in records:
        st = r["settings"]
        if r.get("git_dirty") is not False or r.get("git_dirty_noise_reachability") is not False:
            raise ValueError("provenance: a run on a dirty tree")
        if st["task_id"] != 0 or st["n_action_steps"] != 10 or st["max_steps"] != 520:
            raise ValueError(f"provenance: settings {st}")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1:
        raise ValueError(f"provenance: policy paths {paths}")
    return {"records": len(records), "commit": commits.pop(), "policy_path": paths.pop()}


def control_counts(root) -> tuple[dict, dict]:
    control = {f: sum(load_family(root, f, CONTROL_STATE, CONTROL_IDS).values()) for f in FAMILIES}
    return control, {f: control[f] >= MIN_CONTROL for f in FAMILIES}


def gate(root, which: str) -> str | None:
    """Why the run must stop, or None. Run after the harness and after the control stage, before spending more."""
    root = Path(root)
    if which == "harness":
        h = harness_check(root / "harness" / "per_episode_base.csv")
        return None if h["ok"] else f"harness failed: base {h['base']}/30 against {HARNESS_REFERENCE}"
    control, read = control_counts(root)
    return None if read["sampled"] else f"sampled control failed: {control['sampled']}/{len(CONTROL_IDS)}"


def analyse(root) -> dict:
    root = Path(root)
    harness = harness_check(root / "harness" / "per_episode_base.csv")
    control, read = control_counts(root)
    out = {"provenance": check_provenance(root), "harness": harness, "control": control, "family_read": read,
           "states": {}, "unread_hits": {}}
    if not harness["ok"] or not read["sampled"]:
        out["decision"] = "not read: the harness or the sampled control failed"
        return out
    classes = {}
    for s in SYSTEMATIC_STATES:
        wins = {f: load_family(root, f, s) for f in FAMILIES if read[f]}
        confirmed = {f: confirm_counts(root, f, s, hits_to_confirm(wins[f])) for f in FIXED_FAMILIES if read[f]}
        classes[s] = classify(wins, confirmed)
        for f in FIXED_FAMILIES:
            if not read[f] and sum(load_family(root, f, s).values()):
                out["unread_hits"].setdefault(f, {})[s] = sum(load_family(root, f, s).values())
        n = sum(len(w) for w in wins.values())
        hits = sum(sum(w.values()) for w in wins.values())
        out["states"][s] = {"episodes": n, "hits": hits, "per_family": {f: sum(w.values()) for f, w in wins.items()},
                            "confirmed": {f: {str(k): c for k, c in v.items()} for f, v in confirmed.items()},
                            "class": classes[s], "upper_if_zero": zero_upper(n) if hits == 0 else None}
    out["decision"] = reading(classes)
    return out


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"harness: base {h['base']}/30 against {HARNESS_REFERENCE} (gap {h['gap']:+d}, "
             f"{'ok' if h['ok'] else 'FAILED'})",
             f"control, state {CONTROL_STATE}, of 35: " + ", ".join(
                 f"{f} {n}{'' if r['family_read'][f] else ' (not read)'}" for f, n in r["control"].items())]
    for s, v in r["states"].items():
        lines.append(f"state {s}: {v['hits']}/{v['episodes']} {v['per_family']} confirmed {v['confirmed']} -> "
                     f"{v['class']}" + (f" (one-sided 95% bound on the mixed rate {100 * v['upper_if_zero']:.1f}%)"
                                            if v["upper_if_zero"] else ""))
    if r["unread_hits"]:
        lines.append(f"hits of families not read (they did happen): {r['unread_hits']}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def stage_rate(folder) -> dict:
    """Episodes and wall time of one pilot stage, model loads included."""
    folder = Path(folder)
    episodes = sum(len(re.findall(r" time=[0-9.]+s", f.read_text())) for f in sorted(folder.glob("*.log")))
    wall = float((folder / "wall_seconds.txt").read_text())
    if not episodes or wall <= 0:
        raise ValueError(f"{folder}: {episodes} episodes in {wall} s")
    return {"episodes": episodes, "wall_seconds": wall, "episodes_per_hour": episodes / wall * 3600}


def choose_streams(rates: dict[int, float], cpus: int) -> int:
    """Streams to use, from {streams: episodes per hour}: climb while throughput gains MIN_STREAM_GAIN."""
    best = None
    for n in sorted(k for k in rates if k < cpus):
        if best is None or rates[n] >= MIN_STREAM_GAIN * best[1]:
            best = (n, rates[n])
        else:
            break
    if best is None:
        raise ValueError(f"no stage below {cpus} CPUs")
    return best[0]


def analyse_pilot(folder, cpus: int) -> dict:
    stages = {int(d.name.split("_")[1]): stage_rate(d) for d in sorted(Path(folder).glob("streams_*"))}
    chosen = choose_streams({k: v["episodes_per_hour"] for k, v in stages.items()}, cpus)
    return {"stages": stages, "cpus": cpus, "streams": chosen, "episodes_per_hour": stages[chosen]["episodes_per_hour"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root")
    p.add_argument("--pilot")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--hits", action="store_true", help="print the confirmation jobs of --root")
    p.add_argument("--gate", choices=("harness", "control"), help="exit 1 if --root fails this gate")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.pilot:
        r = analyse_pilot(a.pilot, a.cpus)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r, indent=1))
        return
    if a.gate:
        reason = gate(a.root, a.gate)
        if reason:
            raise SystemExit(f"STOP: {reason}")
        print(f"gate {a.gate} passed")
        return
    if a.hits:
        read = control_counts(a.root)[1]
        for f in (f for f in FIXED_FAMILIES if read[f]):
            for s in SYSTEMATIC_STATES:
                for k in hits_to_confirm(load_family(a.root, f, s)):
                    print(f, s, k)
        return
    r = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")


if __name__ == "__main__":
    main()
