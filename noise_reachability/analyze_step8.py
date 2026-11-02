#!/usr/bin/env python3
"""Step 8 analysis (README.md, step 8): does fixed noise beat sampling on states that resist sampling? Standard library only.

    python3 -S noise_reachability/analyze_step8.py --root <run folder> --out step8_report.txt
    python3 -S noise_reachability/analyze_step8.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step8.py --root <run folder> --second         # task state: the screen's second part
    python3 -S noise_reachability/analyze_step8.py --root <run folder> --order          # task state: eligible, in order
    python3 -S noise_reachability/analyze_step8.py --root <run folder> --tests          # task state first-last noise ids
    python3 -S noise_reachability/analyze_step8.py --root <run folder> --hits           # task state noise id to confirm

<run folder>/task_<t>/ holds harness/seed<s>/, control/, screen/, scene/ and test/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import (EPISODE_SEED_BASE, MIN_CONTROL, confirm_counts, hits_to_confirm,  # noqa: E402
                            load_family, read_rows, zero_upper)
from analyze_step7 import ALPHA, HARNESS_TOLERANCE, POLICY_REVISION, clopper_pearson, fisher_greater  # noqa: E402

TASKS = (0, 4, 6, 7)
HARNESS_REFERENCE = {0: 7, 4: 14, 6: 20, 7: 18}  # step 0: base, seeds 0-2, of 30
CONTROL = {0: 9, 4: 5, 6: 0, 7: 4}
CONTROL_IDS = tuple(range(35, 70))
FAMILIES = ("sampled", "repeat")
TEST1 = (6, 8)
TEST1_IDS = tuple(range(100, 400))
SCREEN_TASKS = (0, 4, 6)
SCREEN_STATES = tuple(range(10, 30))
DIRECT = {7: (33, 38, 47)}                       # step 3b's only task-7 states with 0 hits in 10-49
FIRST_IDS = tuple(range(400, 408))
SECOND_IDS = tuple(range(408, 432))
ALL_SCREEN_IDS = FIRST_IDS + SECOND_IDS
MAX_SECOND = 20
MAX_SELECTED = 4
MIN_SELECTED = 2
TEST2_FIRST_ID, TEST2_IDS_PER_STATE = 1000, 150


def task_root(root, task) -> Path:
    return Path(root) / f"task_{task}"


def test2_ids(i: int) -> tuple[int, ...]:
    first = TEST2_FIRST_ID + TEST2_IDS_PER_STATE * i
    return tuple(range(first, first + TEST2_IDS_PER_STATE))


def round_robin(by_task: dict, limit: int | None = None) -> list[tuple[int, int]]:
    """(task, state) in rounds over the tasks in ascending order, each task's lowest state first."""
    queues = {t: sorted(by_task[t]) for t in sorted(by_task)}
    out = []
    while any(queues.values()):
        for t, q in queues.items():
            if q and (limit is None or len(out) < limit):
                out.append((t, q.pop(0)))
        if limit is not None and len(out) >= limit:
            break
    return out


def holm(ps: dict) -> dict:
    """{test: passes} at family-wise ALPHA. Ties go in the name's order."""
    passed = {k: False for k in ps}
    for i, k in enumerate(sorted(ps, key=lambda x: (ps[x], str(x)))):
        if ps[k] >= ALPHA / (len(ps) - i):
            break
        passed[k] = True
    return passed


def reading(test1: bool | None, test2: bool | None) -> str:
    """None: the test did not run."""
    if test1 is None and test2 is None:
        return "not read: neither test ran"
    if test1 and test2:
        return "fixed noise beats sampling on resistant states beyond task 0, on a state and on unlabelled states chosen in advance"
    if test1 and test2 is False:
        return "6/8 is confirmed, but the pattern did not appear on unlabelled states"
    if test2 and test1 is False:
        return "the pattern holds on unlabelled states, and step 7's result on 6/8 did not replicate"
    if test1:
        return "6/8 is confirmed; too few resistant states to test anything else"
    if test2:
        return "the pattern holds on unlabelled states; 6/8 was not tested"
    return "not shown: claim 2 stays a result about task 0"


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
    folder = task_root(root, task) / "control"
    control = {f: sum(load_family(folder, f, CONTROL[task], CONTROL_IDS).values()) for f in FAMILIES}
    return control, {f: control[f] >= MIN_CONTROL for f in FAMILIES}


def check_provenance(root) -> dict:
    """One commit, clean trees and the protocol's settings in every record; each record belongs to its task's folder."""
    records = []
    for task in TASKS:
        for f in sorted(task_root(root, task).rglob("provenance.jsonl")):
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
    packages = {json.dumps(r.get("packages"), sort_keys=True) for r in records}
    if len(packages) != 1:
        raise ValueError(f"provenance: packages changed between records: {packages}")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1:
        raise ValueError(f"provenance: policy paths {paths}")
    if POLICY_REVISION not in next(iter(paths)):
        raise ValueError(f"provenance: not revision {POLICY_REVISION}: {paths}")
    return {"records": len(records), "commit": commits.pop(), "policy_path": paths.pop()}


def gate(root, which: str) -> str | None:
    """Why the run must stop, or None. Provenance is checked here too, so a wrong setting stops the run early."""
    check_provenance(root)
    for task in TASKS:
        if which == "harness":
            h = harness_check(root, task)
            if not h["ok"]:
                return f"harness failed: task {task}, base {h['base']}/30 against {h['reference']}"
        else:
            control, read = control_counts(root, task)
            if not read["sampled"]:
                return f"sampled control failed: task {task}, {control['sampled']}/{len(CONTROL_IDS)}"
    return None


def read_tasks(root) -> list[int]:
    """Tasks whose repeat control passed; the others leave the tests (README.md, step 8)."""
    return [t for t in TASKS if control_counts(root, t)[1]["repeat"]]


def load_screen(root, task, state, ids) -> int:
    """Hits on these noise ids of the screen. Refuses a missing episode, and an id the screen does not have."""
    rows = read_rows(task_root(root, task) / "screen" / f"per_episode_sampled_state{state}.csv")
    wanted = {EPISODE_SEED_BASE + k for k in ids}
    if not wanted <= set(rows) or not set(rows) <= {EPISODE_SEED_BASE + k for k in ALL_SCREEN_IDS}:
        raise ValueError(f"screen task {task} state {state}: {len(rows)} episodes do not cover {len(wanted)}; finish the run")
    return sum(rows[s] for s in wanted)


def second_states(root, read=None) -> list[tuple[int, int]]:
    """The states that go on to the screen's second part."""
    read = read_tasks(root) if read is None else read
    zero = {t: [s for s in SCREEN_STATES if load_screen(root, t, s, FIRST_IDS) == 0] for t in SCREEN_TASKS if t in read}
    return round_robin(zero, MAX_SECOND)


def eligible_order(root, read=None) -> list[tuple[int, int]]:
    """States at 0 of 32, in selection order."""
    read = read_tasks(root) if read is None else read
    states = second_states(root, read) + [(t, s) for t, v in DIRECT.items() if t in read for s in v]
    by_task = {}
    for t, s in states:
        if load_screen(root, t, s, ALL_SCREEN_IDS) == 0:
            by_task.setdefault(t, []).append(s)
    return round_robin(by_task)


def scene_of(root, task, state) -> bool | None:
    """Whether the state gave one scene; None if not checked. Refuses a file for another state."""
    path = task_root(root, task) / "scene" / f"scene_state{state}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r["task_id"] != task or r["init_state"] != state:
        raise ValueError(f"{path}: another state")
    return bool(r["same"])


def selection(root, read=None) -> tuple[list, tuple | None]:
    """(states for test 2, next state to check or None). A state that failed is passed over, as the README says."""
    chosen = []
    for key in eligible_order(root, read):
        if len(chosen) == MAX_SELECTED:
            break
        same = scene_of(root, *key)
        if same is None:
            return chosen, key
        if same:
            chosen.append(key)
    return chosen, None


def test_plan(root) -> dict:
    """{'test1': (task, state) or None, 'test2': [(task, state, ids)]}; refuses an unfinished selection."""
    read = read_tasks(root)
    chosen, pending = selection(root, read)
    if pending is not None:
        raise ValueError(f"scene check of {pending} is missing")
    first = None
    if TEST1[0] in read:
        same = scene_of(root, *TEST1)
        if same is None:
            raise ValueError("scene check of 6/8 is missing")
        first = TEST1 if same else None
    second = [(t, s, test2_ids(i)) for i, (t, s) in enumerate(chosen)] if len(chosen) >= MIN_SELECTED else []
    return {"test1": first, "test2": second}


def state_result(root, task, state, ids) -> dict:
    folder = task_root(root, task) / "test"
    wins = {f: load_family(folder, f, state, ids) for f in FAMILIES}
    counts = {f: sum(w.values()) for f, w in wins.items()}
    hits = hits_to_confirm(wins["repeat"])
    return {"task": task, "state": state, "episodes": len(ids), **counts,
            "sampled_interval": clopper_pearson(counts["sampled"], len(ids)),
            "repeat_interval": clopper_pearson(counts["repeat"], len(ids)),
            "sampled_bound_if_zero": zero_upper(len(ids)) if counts["sampled"] == 0 else None,
            "confirmed": {str(k): c for k, c in confirm_counts(folder, "repeat", state, hits).items()}}


def analyse(root) -> dict:
    harness = {t: harness_check(root, t) for t in TASKS}
    out = {"provenance": check_provenance(root), "harness": harness, "control": {}, "tests": {}, "states": []}
    if not all(h["ok"] for h in harness.values()):
        out["decision"] = "not read: a harness failed"
        return out
    counts = {t: control_counts(root, t) for t in TASKS}
    out["control"] = {t: c[0] for t, c in counts.items()}
    if not all(c[1]["sampled"] for c in counts.values()):
        out["decision"] = "not read: a sampled control failed"
        return out
    read = [t for t in TASKS if counts[t][1]["repeat"]]
    plan = test_plan(root)
    order = eligible_order(root, read)
    out["read_tasks"] = read
    out["test1_status"] = ("run" if plan["test1"] else
                           "not run: task 6's repeat control failed" if TEST1[0] not in read else "not run: the scene moved")
    out["screen"] = {"first_part_zero": {t: sum(load_screen(root, t, s, FIRST_IDS) == 0 for s in SCREEN_STATES)
                                         for t in SCREEN_TASKS if t in read},
                     "second_part": len(second_states(root, read)), "eligible": [list(k) for k in order],
                     "scene_failed": [list(k) for k in order if scene_of(root, *k) is False]}
    ps = {}
    if plan["test1"]:
        r = state_result(root, *plan["test1"], TEST1_IDS)
        r["group"] = "test 1"
        out["states"].append(r)
        ps["test 1"] = fisher_greater(r["repeat"], r["sampled"], r["episodes"])
    fresh = [state_result(root, t, s, ids) for t, s, ids in plan["test2"]]
    for r in fresh:
        r["group"] = "test 2"
    out["states"] += fresh
    if fresh:
        n = sum(r["episodes"] for r in fresh)
        repeat, sampled = sum(r["repeat"] for r in fresh), sum(r["sampled"] for r in fresh)
        ps["test 2"] = fisher_greater(repeat, sampled, n)
        out["pooled"] = {"repeat": repeat, "sampled": sampled, "episodes": n,
                         "repeat_interval": clopper_pearson(repeat, n), "sampled_interval": clopper_pearson(sampled, n),
                         "states_with_a_sampled_hit": sum(r["sampled"] > 0 for r in fresh), "states": len(fresh)}
    passed = holm(ps) if ps else {}
    out["tests"] = {k: {"p": ps[k], "passes": passed[k]} for k in ps}
    out["jittered"] = {"successes": sum(c for r in out["states"] for c in r["confirmed"].values()),
                       "episodes": 4 * sum(len(r["confirmed"]) for r in out["states"])}
    out["decision"] = reading(passed.get("test 1"), passed.get("test 2"))
    return out


def format_report(r: dict) -> str:
    lines = [f"step 8: fixed noise on states that resist sampling; commit {r['provenance']['commit']}"]
    for t, h in r["harness"].items():
        lines.append(f"harness task {t}: base {h['base']}/30 against {h['reference']} (gap {h['gap']:+d}, "
                     f"{'ok' if h['ok'] else 'FAILED'})")
    for t, c in r["control"].items():
        lines.append(f"control task {t}, state {CONTROL[t]}, of {len(CONTROL_IDS)}: sampled {c['sampled']}, repeat {c['repeat']}")
    if "screen" in r:
        s = r["screen"]
        lines.append(f"tasks read: {r['read_tasks']}; states at 0 of 8, of {len(SCREEN_STATES)}: {s['first_part_zero']}; "
                     f"second part {s['second_part']} states; eligible at 0 of 32: {s['eligible']}; failed the scene check: "
                     f"{s['scene_failed']}; test 1 {r['test1_status']}")
    for v in r["states"]:
        a, b, c, d = (100 * x for x in v["sampled_interval"] + v["repeat_interval"])
        lines.append(f"{v['group']}, task {v['task']} state {v['state']}: sampled {v['sampled']}/{v['episodes']} ({a:.1f}-{b:.1f}%), "
                     f"repeat {v['repeat']}/{v['episodes']} ({c:.1f}-{d:.1f}%), jittered {v['confirmed']}"
                     + (f"; sampled one-sided 95% bound {100 * v['sampled_bound_if_zero']:.2f}%" if v["sampled_bound_if_zero"] else ""))
    if "pooled" in r:
        q = r["pooled"]
        a, b, c, d = (100 * x for x in q["sampled_interval"] + q["repeat_interval"])
        lines.append(f"test 2 pooled: sampled {q['sampled']}/{q['episodes']} ({a:.1f}-{b:.1f}%), repeat {q['repeat']}/{q['episodes']} "
                     f"({c:.1f}-{d:.1f}%); {q['states_with_a_sampled_hit']} of {q['states']} states have a sampled hit")
    for k, v in r["tests"].items():
        lines.append(f"{k}: one-sided Fisher p {v['p']:.3g}, Holm {'passes' if v['passes'] else 'does not pass'}")
    if "jittered" in r:
        lines.append(f"jittered successes (descriptive): {r['jittered']['successes']}/{r['jittered']['episodes']}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--gate", choices=("harness", "control"), help="exit 1 if the run fails this gate")
    p.add_argument("--second", action="store_true")
    p.add_argument("--order", action="store_true")
    p.add_argument("--tests", action="store_true")
    p.add_argument("--hits", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.gate:
        reason = gate(a.root, a.gate)
        print(f"STOP: {reason}" if reason else f"gate {a.gate} passed")
        return 1 if reason else 0
    if a.second or a.order:
        for t, s in (second_states(a.root) if a.second else eligible_order(a.root)):
            print(t, s)
        return 0
    if a.tests or a.hits:
        plan = test_plan(a.root)
        states = ([(*plan["test1"], TEST1_IDS)] if plan["test1"] else []) + plan["test2"]
        for t, s, ids in states:
            if a.tests:
                print(t, s, f"{ids[0]}-{ids[-1]}")
            else:
                for k in hits_to_confirm(load_family(task_root(a.root, t) / "test", "repeat", s, ids)):
                    print(t, s, k)
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
