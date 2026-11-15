#!/usr/bin/env python3
"""Colab T4 reproduction check (README.md, "Colab T4 check"): does a Colab T4 reproduce Kaggle T4 episodes exactly?
Standard library.

    python3 -S noise_reachability/analyze_colab_t4_check.py --root <run folder> --out colab_t4_check.txt

<run folder>/task_<t>/<policy>/per_episode_<base|lora>.csv for policy "base" (the frozen checkpoint) and "N0" (step
10's adapter N<t>_0), each on step 12's harness episodes: state HARNESS_STATE[t] x seeds 16-19. Both were run by step 10
on Kaggle T4 (week6/noise_reachability_step10_kaggle/run/task_<t>/eval/<base|N0>/). An episode reproduces if success and
number of steps are equal. Rule: Colab T4 is interchangeable with Kaggle T4 for paired comparisons if, for each policy,
at least MIN_SAME of the 12 episodes reproduce (the threshold step 12 used for its own sessions).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step12 import HARNESS_SEEDS, HARNESS_STATE, STEP10, TASKS, _rows, task_root  # noqa: E402
from analyze_step7 import POLICY_REVISION  # noqa: E402

POLICIES = {"base": ("base", "base"), "N0": ("lora", "lora")}  # name: (--eval_policies, csv suffix)
MIN_SAME = 11
N_ACTION_STEPS, MAX_STEPS = 10, 520


def want(t) -> set[tuple[int, int]]:
    return {(s, HARNESS_STATE[t]) for s in HARNESS_SEEDS}


def step10_rows(t, policy, step10=STEP10) -> dict:
    """Step 10's Kaggle T4 episodes of this policy on the harness episodes."""
    old = {}
    for p in sorted(glob.glob(str(task_root(step10, t) / "eval" / policy / "part*" / "per_episode_*.csv"))):
        for k, v in _rows(p).items():
            if k in old and old[k] != v:
                raise ValueError(f"step 10 task {t} {policy}: episode {k} twice with different outcomes")
            old[k] = v
    missing = want(t) - set(old)
    if missing:
        raise ValueError(f"step 10 task {t} {policy}: no episodes {sorted(missing)}")
    return {k: old[k] for k in want(t)}


def compare(root, step10=STEP10) -> dict:
    """Per policy and task: episodes equal to step 10's in success and steps; refuses missing or extra episodes."""
    out = {}
    for policy, (_, suffix) in POLICIES.items():
        per, same = {}, 0
        for t in TASKS:
            path = Path(root) / f"task_{t}" / policy / f"per_episode_{suffix}.csv"
            new = _rows(path) if path.exists() else {}
            if set(new) != want(t):
                raise ValueError(f"task {t} {policy}: has {sorted(new)}, expected {sorted(want(t))}; finish the run")
            old = step10_rows(t, policy, step10)
            k = sum(new[e] == old[e] for e in want(t))
            per[t] = {"same": k, "episodes": len(want(t)), "successes_now": sum(new[e][0] for e in want(t)),
                      "successes_step10": sum(old[e][0] for e in want(t)),
                      "differ": sorted([list(e), list(new[e]), list(old[e])] for e in want(t) if new[e] != old[e])}
            same += k
        n = len(TASKS) * len(HARNESS_SEEDS)
        out[policy] = {"per_task": per, "same": same, "episodes": n, "passes": same >= MIN_SAME}
    return out


def provenance(root, step10=STEP10) -> dict:
    """One commit, clean trees, T4 GPUs, the harness settings in every folder; packages compared with step 10's."""
    records = []
    for t in TASKS:
        for policy, (flag, _) in POLICIES.items():
            path = Path(root) / f"task_{t}" / policy / "provenance.jsonl"
            if not path.exists():
                raise ValueError(f"{path}: missing")
            for line in path.read_text().splitlines():
                r = json.loads(line)
                st = r["settings"]
                checks = {"stage": "eval", "task_id": t, "eval_policies": flag, "n_action_steps": N_ACTION_STEPS,
                          "max_steps": MAX_STEPS, "eval_init_states": str(HARNESS_STATE[t]),
                          "eval_seeds": f"{HARNESS_SEEDS[0]}-{HARNESS_SEEDS[-1]}"}
                bad = {k: st.get(k) for k, v in checks.items() if st.get(k) != v}
                if bad:
                    raise ValueError(f"{path}: settings {bad}, expected {checks}")
                if POLICY_REVISION not in st["policy_path"]:
                    raise ValueError(f"{path}: policy {st['policy_path']}")
                if flag == "lora" and Path(st["lora_dir"]).parent.name != f"N{t}_0":
                    raise ValueError(f"{path}: adapter {st['lora_dir']}, expected step 10's N{t}_0")
                if r.get("git_dirty") is not False or r.get("git_dirty_sfbc") is not False or not r.get("git_commit"):
                    raise ValueError(f"{path}: commit {r.get('git_commit')}, dirty {r.get('git_dirty')}")
                if not r.get("gpus") or any("T4" not in g for g in r["gpus"]):
                    raise ValueError(f"{path}: GPUs {r.get('gpus')}, expected T4")
                records.append(r)
    commits = {r["git_commit"] for r in records}
    if len(commits) != 1:
        raise ValueError(f"provenance: commits {commits}")
    packages = {json.dumps(r.get("packages"), sort_keys=True) for r in records}
    if len(packages) != 1:
        raise ValueError("provenance: packages changed between records")
    with open(task_root(step10, 0) / "eval" / "base" / "part0" / "provenance.jsonl") as f:
        old = json.loads(f.readline())
    ours = records[0].get("packages")
    return {"records": len(records), "commit": commits.pop(), "gpus": sorted({g for r in records for g in r["gpus"]}),
            "packages": ours, "step10_packages": old.get("packages"),
            "same_packages_as_step10": ours == old.get("packages")}


def report(root, step10=STEP10) -> dict:
    prov = provenance(root, step10)
    cmp_ = compare(root, step10)
    passes = all(v["passes"] for v in cmp_.values())
    return {"provenance": prov, "compare": cmp_, "passes": passes,
            "decision": ("Colab T4 reproduces Kaggle T4 episodes: interchangeable for paired comparisons" if passes else
                         "Colab T4 does not reproduce Kaggle T4 episodes: use it only for self-contained experiments")}


def text(r) -> str:
    p = r["provenance"]
    lines = [f"Colab T4 check: commit {p['commit']}, {p['records']} provenance records, GPUs {p['gpus']}, "
             f"packages {'the same as' if p['same_packages_as_step10'] else 'DIFFERENT from'} step 10's"]
    for policy, v in r["compare"].items():
        per = ", ".join(f"task {t} {x['same']}/{x['episodes']}" for t, x in v["per_task"].items())
        lines.append(f"{policy}: {v['same']} of {v['episodes']} episodes equal to step 10's in success and steps "
                     f"({per}); {'ok' if v['passes'] else 'FAILS'} (at least {MIN_SAME})")
        for t, x in v["per_task"].items():
            for e, new, old in x["differ"]:
                lines.append(f"  task {t} seed {e[0]} state {e[1]}: Colab (success, steps) {tuple(new)}, step 10 {tuple(old)}")
    lines.append("decision: " + r["decision"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    r = report(args.root)
    Path(args.out).write_text(text(r))
    Path(args.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    print(text(r), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
