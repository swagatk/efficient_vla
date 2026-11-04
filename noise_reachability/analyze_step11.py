#!/usr/bin/env python3
"""Step 11 analysis (README.md, step 11): does step 8's test 2 replicate on step 9's eight held-out states?
Standard library only.

Step 9 ran fresh `repeat` noise ids 6000-6049 on eight resistant states (committed in
week6/noise_reachability_step9_kaggle/run). Step 11 runs the same ids with `sampled` noise on the same states and seeds,
so each id is a pair of episodes on one scene. Test A: the eight states pooled; test B: without task 7. One-sided Fisher
as step 8's test 2, Holm over A and B at 0.05.

    python3 -S noise_reachability/analyze_step11.py --root <run folder> --out step11_report.txt
    python3 -S noise_reachability/analyze_step11.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step11.py --root <run folder> --jobs           # task state ids, per job

<run folder>/task_<t>/ holds harness/seed<s>/, scene/ and test/sampled/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import load_family  # noqa: E402
from analyze_step7 import ALPHA, POLICY_REVISION, clopper_pearson, fisher_greater  # noqa: E402
from analyze_step8 import TASKS, check_provenance, harness_check, holm, scene_of, task_root  # noqa: E402
from analyze_step10 import mcnemar_greater  # noqa: E402

HELD_OUT = {0: (14, 27), 4: (18, 26), 6: (14, 20), 7: (38, 47)}   # step 9's states
IDS = tuple(range(6000, 6050))                                     # step 9's random arm
STEP9 = Path(__file__).resolve().parent.parent / "week6/noise_reachability_step9_kaggle/run"
STEP9_COMMIT = "f2a067e629557b44996b423c08835647544178b9"
WITHOUT = 7                                                        # test B leaves this task out
# sha256 of the committed step-9 files this step pairs with (week6/noise_reachability_step9_kaggle/run, commit 9cb2fc6)
STEP9_SHA256 = {
    "task_0/test/random/per_episode_repeat_state14.csv": "b977adc273c66517aea12592d0fcc8ca854a48c6f4fdb12331f08aa372e3698c",
    "task_0/test/random/per_episode_repeat_state27.csv": "0416676f43e0318afc48561a9cc5ac479960d852ad95e8504abe06349db1326a",
    "task_4/test/random/per_episode_repeat_state18.csv": "043f9404422cdb3bf021188f24ecb92cf130c27b2bedcce795eb6b4cc906ba03",
    "task_4/test/random/per_episode_repeat_state26.csv": "ea6b54901168e70ebeb0712e2679dedbf9bc568069ef45342bdc61450c4152f5",
    "task_6/test/random/per_episode_repeat_state14.csv": "2c049ae02301bf2ae092e6f246fd42e94a8067badf1f37664f36047faa692b13",
    "task_6/test/random/per_episode_repeat_state20.csv": "012fbf04a807a67b212be6019c53ff7f3baf4beb645854531cc6493bf97e98d1",
    "task_7/test/random/per_episode_repeat_state38.csv": "94ec76e110eaf7dc6a0bade97e19352539c742d82df85931668d3de9d8ba4bd2",
    "task_7/test/random/per_episode_repeat_state47.csv": "a435e82b7c4e7e9ed2a911c2f02698c0475103b70664fcd9fdda2e9e0ea732d7",
}


def ids_text() -> str:
    return ",".join(str(k) for k in IDS)


def check_step9(step9, pinned=None) -> dict:
    """Step 9's repeat arm on ids 6000-6049: its commit, clean trees and settings; sha256 of every file read, which must
    equal ``pinned`` when given (the committed files)."""
    files, hashes, packages = 0, {}, set()
    for t in TASKS:
        folder = Path(step9) / f"task_{t}" / "test" / "random"
        for line in open(folder / "provenance.jsonl"):
            r = json.loads(line)
            st = r["settings"]
            if r["git_commit"] != STEP9_COMMIT or r["git_dirty"] is not False:
                raise ValueError(f"step 9 provenance: {r['git_commit']} dirty={r['git_dirty']} in {folder}")
            if (st["task_id"], st["family"], st["n_action_steps"]) != (t, "repeat", 10) or st["noise_ids"] != ids_text():
                raise ValueError(f"step 9 provenance: settings {st}")
            if (st["max_steps"] != 520 or POLICY_REVISION not in st["policy_path"] or st["init_state"] not in HELD_OUT[t]
                    or r.get("git_dirty_noise_reachability") is not False):
                raise ValueError(f"step 9 provenance: settings {st}")
            packages.add(json.dumps(r.get("packages"), sort_keys=True))
            files += 1
        for s in HELD_OUT[t]:
            p = folder / f"per_episode_repeat_state{s}.csv"
            hashes[str(p.relative_to(step9))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if pinned is not None and hashes != pinned:
        raise ValueError(f"step 9 files differ from the committed ones: {sorted(k for k in pinned if hashes.get(k) != pinned[k])}")
    if len(packages) != 1:
        raise ValueError(f"step 9 provenance: packages changed between records: {packages}")
    return {"records": files, "commit": STEP9_COMMIT, "sha256": hashes, "pinned": pinned is not None,
            "packages": packages.pop()}


def check_settings(root) -> None:
    """Every record of the sampled arm: family sampled, ids 6000-6049, its task and a held-out state, no jitter."""
    for t in TASKS:
        f = task_root(root, t) / "test" / "sampled" / "provenance.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            st = json.loads(line)["settings"]
            if st["family"] != "sampled" or st["noise_ids"] != ids_text() or st.get("jitter_seeds") is not None:
                raise ValueError(f"sampled arm settings {st}")
            if st["init_state"] not in HELD_OUT[t]:
                raise ValueError(f"sampled arm: state {st['init_state']} is not a held-out state of task {t}")


def gate(root) -> str | None:
    """Why the run must stop after the harness, or None. Provenance is checked too."""
    check_provenance(root)
    check_settings(root)
    for t in TASKS:
        h = harness_check(root, t)
        if not h["ok"]:
            return f"harness failed: task {t}, base {h['base']}/30 against {h['reference']}"
    return None


def read_states(root) -> list[tuple[int, int]]:
    """Held-out states that passed this run's scene check; refuses a state not yet checked."""
    out = []
    for t in TASKS:
        for s in HELD_OUT[t]:
            same = scene_of(root, t, s)
            if same is None:
                raise ValueError(f"scene check of {t}/{s} is missing")
            if same:
                out.append((t, s))
    return out


def jobs(root) -> list[tuple[int, int, str]]:
    return [(t, s, ids_text()) for t, s in read_states(root)]


def pooled_test(rows: list[dict]) -> dict:
    n = sum(r["episodes"] for r in rows)
    a, b = sum(r["repeat"] for r in rows), sum(r["sampled"] for r in rows)
    first, second = sum(r["repeat_only"] for r in rows), sum(r["sampled_only"] for r in rows)
    return {"states": [[r["task"], r["state"]] for r in rows], "episodes": n, "repeat": a, "sampled": b,
            "repeat_interval": clopper_pearson(a, n) if n else None,
            "sampled_interval": clopper_pearson(b, n) if n else None,
            "p": fisher_greater(a, b, n) if n else None,
            "mcnemar_cross_check": mcnemar_greater(first, second) if n else None}


def reading(harness_ok: bool, a: bool | None, b: bool | None) -> str:
    if not harness_ok:
        return "not read: a harness failed"
    if a is None:
        return "not read: no held-out state passed the scene check"
    if a and b:
        return "replicated: fixed noise beats sampling on eight new resistant states chosen in advance, and without task 7"
    if a and b is None:
        return "replicated pooled; test B had no state to read"
    if a:
        return "replicated pooled, but not without task 7: the effect rests on task 7 again"
    if b:
        return "replicated without task 7 only"
    return "not replicated: step 8's test 2 does not extend to these states"


def own_packages(root) -> str:
    return next(json.dumps(json.loads(line).get("packages"), sort_keys=True)
                for f in sorted(Path(root).rglob("provenance.jsonl")) for line in open(f))


def analyse(root, step9=STEP9) -> dict:
    harness = {t: harness_check(root, t) for t in TASKS}
    pinned = STEP9_SHA256 if Path(step9).resolve() == STEP9.resolve() else None
    out = {"provenance": check_provenance(root), "step9": check_step9(step9, pinned), "harness": harness, "states": []}
    out["packages_match_step9"] = own_packages(root) == out["step9"]["packages"]   # reported, not a refusal
    check_settings(root)
    if not all(h["ok"] for h in harness.values()):
        out["decision"] = reading(False, None, None)
        return out
    states = read_states(root)
    out["scene_failed"] = [[t, s] for t in TASKS for s in HELD_OUT[t] if (t, s) not in states]
    for t, s in states:
        rep = load_family(Path(step9) / f"task_{t}" / "test" / "random", "repeat", s, IDS)
        smp = load_family(task_root(root, t) / "test" / "sampled", "sampled", s, IDS)
        out["states"].append({"task": t, "state": s, "episodes": len(IDS), "repeat": sum(rep.values()),
                              "sampled": sum(smp.values()),
                              "repeat_only": sum(rep[k] and not smp[k] for k in IDS),
                              "sampled_only": sum(smp[k] and not rep[k] for k in IDS)})
    if not states:
        out["decision"] = reading(True, None, None)
        return out
    out["test A"] = pooled_test(out["states"])
    out["test B"] = pooled_test([r for r in out["states"] if r["task"] != WITHOUT])
    ps = {k: out[k]["p"] for k in ("test A", "test B") if out[k]["p"] is not None}
    passed = holm(ps)
    for k in ("test A", "test B"):
        out[k]["passes"] = passed.get(k)
    out["decision"] = reading(True, out["test A"]["passes"], out["test B"]["passes"])
    return out


def format_report(r: dict) -> str:
    lines = [f"step 11: does fixed noise beat sampling on step 9's eight held-out states? commit {r['provenance']['commit']}",
             f"repeat arm: step 9 at {r['step9']['commit'][:7]}, {r['step9']['records']} provenance records, "
             f"files {'match the committed sha256' if r['step9']['pinned'] else 'NOT the committed folder'}; "
             f"package versions {'the same as' if r['packages_match_step9'] else 'DIFFERENT from'} step 9's"]
    for t, h in r["harness"].items():
        lines.append(f"harness task {t}: base {h['base']}/30 against {h['reference']} (gap {h['gap']:+d}, "
                     f"{'ok' if h['ok'] else 'FAILED'})")
    if "scene_failed" in r:
        lines.append(f"held-out states that failed the scene check: {r['scene_failed']}")
    for v in r["states"]:
        a, b = clopper_pearson(v["repeat"], v["episodes"]), clopper_pearson(v["sampled"], v["episodes"])
        lines.append(f"{v['task']}/{v['state']}: repeat {v['repeat']}/{v['episodes']} ({100 * a[0]:.1f}-{100 * a[1]:.1f}%), "
                     f"sampled {v['sampled']}/{v['episodes']} ({100 * b[0]:.1f}-{100 * b[1]:.1f}%), "
                     f"pairs repeat only {v['repeat_only']}, sampled only {v['sampled_only']}")
    for k in ("test A", "test B"):
        if k not in r:
            continue
        v = r[k]
        if v["p"] is None:
            lines.append(f"{k}: no states")
            continue
        a, b = v["repeat_interval"], v["sampled_interval"]
        lines.append(f"{k} ({len(v['states'])} states): repeat {v['repeat']}/{v['episodes']} ({100 * a[0]:.1f}-{100 * a[1]:.1f}%), "
                     f"sampled {v['sampled']}/{v['episodes']} ({100 * b[0]:.1f}-{100 * b[1]:.1f}%), one-sided Fisher p "
                     f"{v['p']:.3g}, {'passes' if v['passes'] else 'does not pass'} (Holm at {ALPHA}); "
                     f"paired McNemar cross-check {v['mcnemar_cross_check']:.3g}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--step9", default=str(STEP9), help="step 9's run folder (the committed one by default)")
    p.add_argument("--gate", choices=("harness",), help="exit 1 if the run fails this gate")
    p.add_argument("--jobs", action="store_true", help="print task, state, comma-separated noise ids per job")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.gate:
        why = gate(a.root)
        print(why or f"gate {a.gate} passed")
        return 1 if why else 0
    if a.jobs:
        for t, s, ids in jobs(a.root):
            print(t, s, ids)
        return 0
    r = analyse(a.root, a.step9)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
