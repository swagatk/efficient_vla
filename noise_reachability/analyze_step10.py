#!/usr/bin/env python3
"""Step 10 analysis (README.md, step 10): can constant-noise successes be learned into the weights? Standard library.

    python3 -S noise_reachability/analyze_step10.py --root <run folder> --out step10_report.txt
    python3 -S noise_reachability/analyze_step10.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step10.py --replay_jobs                        # the replays to record

<run folder>/task_<t>/ holds harness/seed<s>/ (the base, as step 8), replay/ (deep_noise_search.py output of the
replays; their recordings are in replay_data/episodes/) and eval/<policy>/part<j>/per_episode_{base,lora}.csv
(self_filtered_bc.py --stage eval). Arm A's collection is task_0/collect_data/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import read_rows  # noqa: E402
from analyze_step7 import ALPHA, POLICY_REVISION, clopper_pearson  # noqa: E402
from analyze_step8 import harness_check, task_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WEEK6 = REPO_ROOT / "week6"
TASKS = (0, 6, 7)
EPISODE_SEED_BASE = 1000
# Where each task's constant-noise successes on resistant states were found: (result folder, state).
REPLAY_SOURCES = {
    0: [("noise_reachability_step5a_colab/run", s) for s in (2, 5, 6, 7, 8)]
       + [("noise_reachability_step8_kaggle/run/task_0/test", 11)],
    6: [("noise_reachability_step7_kaggle/run/task_6", 8), ("noise_reachability_step8_kaggle/run/task_6/test", 8),
        ("noise_reachability_step8_kaggle/run/task_6/test", 12)],
    7: [("noise_reachability_step8_kaggle/run/task_7/test", 33)],
}
FIXED_FAMILIES = ("full", "repeat")
MIN_REPLAYS = 20                                   # successful replays for a task to be read
TRAINED = {0: (2, 5, 6, 7, 8, 11), 6: (8, 12), 7: (33,)}
HELD_OUT = {0: (14, 27), 6: (14, 20), 7: (38, 47)}
EASY = {0: (0, 1, 3, 4, 9), 6: (0,), 7: (4,)}
SYSTEMATIC_0 = (2, 5, 6, 7, 8)                     # test 2: task 0's systematic states, step 4's episodes
EVAL_STATES = {0: tuple(range(10)) + (11, 14, 27), 6: (0, 8, 12, 14, 20), 7: (4, 33, 38, 47)}
EVAL_SEEDS = {0: tuple(range(16, 32)), 6: tuple(range(16, 48)), 7: tuple(range(16, 48))}
# How the notebook passes them, per process (checked against the provenance of every evaluation part).
EVAL_STATE_SPECS = {0: "0-9,11,14,27", 6: "0,8,12,14,20", 7: "4,33,38,47"}
EVAL_SEED_PARTS = {t: [f"{a}-{a + 3}" for a in range(EVAL_SEEDS[t][0], EVAL_SEEDS[t][-1] + 1, 4)] for t in EVAL_SEEDS}
COLLECT_PARTS = ("10-19", "20-29", "30-39", "40-49")   # arm A's collection, as step 4
TRAIN_SETTINGS = {"train_steps": 2000, "batch_size": 16, "learning_rate": 1e-4, "weight_decay": 0.0, "warmup_steps": 100,
                  "grad_clip": 1.0, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "lora_targets": "all",
                  "train_action_heads": 0, "bf16": 1, "selection": "success"}   # step 4's
TRAIN_SEEDS = (0, 1, 2)
MIN_SEEDS = 2                                      # of the 3 training seeds, for test 1


# ---- provenance ------------------------------------------------------------------------------------------------

DIRTY_KEYS = ("git_dirty_noise_reachability", "git_dirty_sfbc")   # deep_noise_search / eval_noise_selection, self_filtered_bc


def adapter_task(name: str) -> int:
    """A<s> trains on task 0; N<t>_<s> on task t."""
    return 0 if name.startswith("A") else int(name[1:].split("_")[0])


def adapter_of(task: int, policy: str) -> str:
    """The adapter an evaluated policy loads: N<s> on task t is N<t>_<s>; A<s> is A<s>."""
    return policy if policy.startswith("A") else f"N{task}_{policy[1:]}"


def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def check_settings(root, t: int, f: Path, r: dict) -> None:
    """The protocol's settings for the kind of record its folder holds; refuses a record in an unknown folder."""
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    if rel[0] == "runs":
        name = rel[1]
        seed = int(name[1:]) if name.startswith("A") else int(name.split("_")[1])
        data = "task_0/collect_data" if name.startswith("A") else f"task_{t}/replay_data"
        _require(st.get("stage") == "train" and st["seed"] == seed, f, "not a training run of this seed")
        _require(str(st["data_dir"]).endswith(data), f, f"data {st['data_dir']}")
        _require(st["val_init_states"] == ("45-49" if name.startswith("A") else ""), f, "validation states")
        _require(all(st.get(k) == v for k, v in TRAIN_SETTINGS.items()), f, "training settings")
    elif rel[1:2] == ("harness",):
        return
    elif rel[1:2] == ("replay",):
        _require(st.get("family") in FIXED_FAMILIES and st["frame_stride"] == 2, f, "replay settings")
        _require(str(st.get("record_dir")).endswith(f"task_{t}/replay_data"), f, f"record_dir {st.get('record_dir')}")
        _require(r.get("git_dirty_sfbc") is False, f, "no clean record of self_filtered_bc (the recorder)")
    elif rel[1:2] == ("collect",) and t == 0:
        k = int(rel[2].removeprefix("part"))
        _require(st.get("stage") == "collect" and st["hard_reset"] == 0 and st["collect_seeds"] == "0-2"
                 and st["frame_stride"] == 2 and st["init_states"] == COLLECT_PARTS[k], f, "collection settings")
    elif rel[1:2] == ("eval",):
        policy, j = rel[2], int(rel[3].removeprefix("part"))
        _require(st.get("stage") == "eval" and st["eval_init_states"] == EVAL_STATE_SPECS[t]
                 and st["eval_seeds"] == EVAL_SEED_PARTS[t][j], f, "evaluation episodes")
        if policy == "base":
            _require(st["eval_policies"] == "base" and st.get("policy") == "base", f, "not the base")
        else:
            _require(st["eval_policies"] == "lora" and st.get("policy") == "lora"
                     and str(st["lora_dir"]).endswith(f"runs/{adapter_of(t, policy)}/adapter"), f,
                     f"adapter {st.get('lora_dir')} for {policy}")
    else:
        raise ValueError(f"provenance: a record in an unexpected folder {f}")


def check_provenance(root) -> dict:
    """One commit, clean trees, n_action_steps 10, 520 steps and one checkpoint in every record of every task folder and
    training run; each record belongs to its folder's task. Accepts either entry point's own dirty flag."""
    files = [(t, f) for t in TASKS for f in sorted(task_root(root, t).rglob("provenance.jsonl"))]
    files += [(adapter_task(f.parent.name), f) for f in sorted(Path(root, "runs").glob("*/provenance.jsonl"))]
    records = []
    for t, f in files:
        for line in open(f):
            r = json.loads(line)
            if r["settings"]["task_id"] != t:
                raise ValueError(f"provenance: task {r['settings']['task_id']} in {f}")
            check_settings(root, t, f, r)
            records.append(r)
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"provenance: commits {commits}")
    for r in records:
        st = r["settings"]
        flags = [r[k] for k in DIRTY_KEYS if k in r]
        if r.get("git_dirty") is not False or not flags or any(v is not False for v in flags):
            raise ValueError("provenance: a run on a dirty tree")
        if st["n_action_steps"] != 10 or st["max_steps"] != 520:
            raise ValueError(f"provenance: settings {st}")
    packages = {json.dumps(r.get("packages"), sort_keys=True) for r in records}
    if len(packages) != 1:
        raise ValueError(f"provenance: packages changed between records: {packages}")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1 or POLICY_REVISION not in next(iter(paths)):
        raise ValueError(f"provenance: policy paths {paths}")
    return {"records": len(records), "commit": commits.pop(), "policy_path": paths.pop()}


# ---- the replays ------------------------------------------------------------------------------------------------

def _wins(path) -> list[int]:
    return sorted(s for s, v in read_rows(path).items() if v)


def replay_plan(week6=WEEK6) -> dict[int, list[tuple[str, int, int, int | None]]]:
    """{task: [(family, state, noise id, jitter seed or None)]}: every successful fixed-noise episode of the sources."""
    plan = {}
    for t, sources in REPLAY_SOURCES.items():
        eps = []
        for folder, s in sources:
            for f in FIXED_FAMILIES:
                p = Path(week6, folder, f"per_episode_{f}_state{s}.csv")
                if p.exists():
                    eps += [(f, s, seed - EPISODE_SEED_BASE, None) for seed in _wins(p)]
                for c in sorted(glob.glob(str(Path(week6, folder, f"confirm_{f}_n*_state{s}.csv")))):
                    k = int(re.search(r"_n(\d+)_state", c).group(1))
                    eps += [(f, s, k, j) for j in _wins(c)]
        if len(set(eps)) != len(eps):
            raise ValueError(f"task {t}: a replay appears twice")
        plan[t] = sorted(eps, key=lambda e: (e[0], e[1], e[2], -1 if e[3] is None else e[3]))
    return plan


def source_steps(week6=WEEK6) -> dict[tuple, int]:
    """{(task, family, state, noise id, jitter seed or None): steps} of the original successful episodes."""
    out = {}
    for t, sources in REPLAY_SOURCES.items():
        for folder, s in sources:
            for f in FIXED_FAMILIES:
                p = Path(week6, folder, f"per_episode_{f}_state{s}.csv")
                files = ([(p, None)] if p.exists() else []) + [
                    (Path(c), int(re.search(r"_n(\d+)_state", c).group(1)))
                    for c in sorted(glob.glob(str(Path(week6, folder, f"confirm_{f}_n*_state{s}.csv"))))]
                for path, k in files:
                    for r in csv.DictReader(open(path)):
                        if int(r["success"]):
                            seed = int(r["seed"])
                            key = (t, f, s, seed - EPISODE_SEED_BASE, None) if k is None else (t, f, s, k, seed)
                            out[key] = int(r["steps"])
    return out


def replay_jobs(plan) -> list[tuple[int, str, int, str, str | None]]:
    """(task, family, state, noise ids, jitter seeds or None): one deep_noise_search.py call each."""
    jobs = []
    for t, eps in plan.items():
        plain = {}
        for f, s, k, j in eps:
            if j is None:
                plain.setdefault((f, s), []).append(k)
        for (f, s), ks in sorted(plain.items()):
            jobs.append((t, f, s, ",".join(map(str, ks)), None))
        jittered = {}
        for f, s, k, j in eps:
            if j is not None:
                jittered.setdefault((f, s, k), []).append(j)
        for (f, s, k), js in sorted(jittered.items()):
            jobs.append((t, f, s, str(k), ",".join(map(str, js))))
    return jobs


def replay_counts(root, plan, steps=None) -> dict[int, dict]:
    """Per task: replays planned, replays that succeeded again, and (with ``steps``, the originals' step counts) how many
    of those took the original's number of steps, a proxy for reproducing it exactly. Refuses a missing or extra replay."""
    out = {}
    for t, eps in plan.items():
        folder = task_root(root, t) / "replay"
        got = {}
        for f, s, k, j in eps:
            name = f"per_episode_{f}_state{s}.csv" if j is None else f"confirm_{f}_n{k}_state{s}.csv"
            rows = got.setdefault(name, read_rows(folder / name) if (folder / name).exists() else {})
            seed = EPISODE_SEED_BASE + k if j is None else j
            if seed not in rows:
                raise ValueError(f"task {t}: replay {f} {s} {k} {j} is missing; finish the run")
        want = {}
        for f, s, k, j in eps:
            name = f"per_episode_{f}_state{s}.csv" if j is None else f"confirm_{f}_n{k}_state{s}.csv"
            want.setdefault(name, set()).add(EPISODE_SEED_BASE + k if j is None else j)
        if any(set(got[n]) != want[n] for n in want):
            raise ValueError(f"task {t}: replay files hold episodes that were not planned")
        out[t] = {"planned": len(eps), "succeeded": sum(sum(rows.values()) for rows in got.values())}
        if steps is not None:
            same = 0
            for f, s, k, j in eps:
                name = f"per_episode_{f}_state{s}.csv" if j is None else f"confirm_{f}_n{k}_state{s}.csv"
                seed = EPISODE_SEED_BASE + k if j is None else j
                row = next(r for r in csv.DictReader(open(folder / name)) if int(r["seed"]) == seed)
                same += int(row["success"]) == 1 and int(row["steps"]) == steps[(t, f, s, k, j)]
            out[t]["same_steps"] = same
    return out


# ---- the evaluation ---------------------------------------------------------------------------------------------

def policies(task: int) -> list[str]:
    return ["base"] + [f"N{i}" for i in TRAIN_SEEDS] + ([f"A{i}" for i in TRAIN_SEEDS] if task == 0 else [])


def load_eval(root, task: int, policy: str) -> dict[tuple[int, int], int]:
    """{(seed, init state): success} of one policy on one task, merged over its parts; refuses gaps and repeats."""
    name = "base" if policy == "base" else "lora"
    out = {}
    for path in sorted(glob.glob(str(task_root(root, task) / "eval" / policy / "part*" / f"per_episode_{name}.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["seed"]), int(r["init_state"]))
                if int(r["task_id"]) != task:
                    raise ValueError(f"{path}: an episode of task {r['task_id']}")
                if key in out:
                    raise ValueError(f"{path}: episode {key} twice")
                out[key] = int(r["success"])
    want = {(s, i) for s in EVAL_SEEDS[task] for i in EVAL_STATES[task]}
    if set(out) != want:
        raise ValueError(f"task {task} {policy}: {len(set(out) & want)} of {len(want)} episodes and "
                         f"{len(set(out) - want)} others; finish the run")
    return out


def mcnemar_greater(first_only: int, second_only: int) -> float:
    """One-sided exact McNemar: P(X >= first_only), X ~ Bin(first_only + second_only, 1/2)."""
    n = first_only + second_only
    return sum(comb(n, j) for j in range(first_only, n + 1)) / 2 ** n


def paired(first: dict, second: dict, keys) -> dict:
    """``first`` against ``second`` on ``keys``: successes, discordant counts and the one-sided p."""
    keys = list(keys)
    a = sum(1 for k in keys if first[k] and not second[k])
    b = sum(1 for k in keys if second[k] and not first[k])
    return {"n": len(keys), "first": sum(first[k] for k in keys), "second": sum(second[k] for k in keys),
            "first_only": a, "second_only": b, "p": mcnemar_greater(a, b)}


def keys_of(task: int, states) -> list[tuple[int, int]]:
    return [(s, i) for s in EVAL_SEEDS[task] for i in states]


def reading(harness_ok: bool, read: list[int], test1: bool | None, test2: bool | None) -> str:
    if not harness_ok:
        return "not read: a harness failed"
    if not read:
        return "not read: no task has enough successful replays"
    if not test1:
        return ("not shown: one round of filtered BC on constant-noise successes did not make these states solvable "
                "by sampling")
    if test2:
        return "constant-noise successes make resistant states learnable, beyond what the policy's own successes give"
    return "learnable, but not shown better than training on its own successes"


def gate(root) -> str | None:
    check_provenance(root)
    for t in TASKS:
        h = harness_check(root, t)
        if not h["ok"]:
            return f"harness failed: task {t}, base {h['base']}/30 against {h['reference']}"
    return None


def analyse(root, plan=None, steps=None) -> dict:
    if plan is None:
        plan, steps = replay_plan(), source_steps()
    harness = {t: harness_check(root, t) for t in TASKS}
    out = {"provenance": check_provenance(root), "harness": harness, "replays": {}, "states": [], "tests": {}}
    if not all(h["ok"] for h in harness.values()):
        out["decision"] = reading(False, [], None, None)
        return out
    out["replays"] = replay_counts(root, plan, steps)
    read = [t for t in TASKS if out["replays"][t]["succeeded"] >= MIN_REPLAYS]
    out["read_tasks"] = read
    if not read:
        out["decision"] = reading(True, [], None, None)
        return out
    ev = {t: {p: load_eval(root, t, p) for p in policies(t)} for t in read}
    for t in read:
        for i in EVAL_STATES[t]:
            role = "trained" if i in TRAINED[t] else "held out" if i in HELD_OUT[t] else "easy"
            row = {"task": t, "state": i, "role": role, "episodes": len(EVAL_SEEDS[t])}
            for p in policies(t):
                if p.startswith("A") and i not in range(10):
                    continue                      # A's collection used states 10-49: A is read on 0-9 only
                k = sum(ev[t][p][(s, i)] for s in EVAL_SEEDS[t])
                row[p] = [k, clopper_pearson(k, len(EVAL_SEEDS[t]))]
            out["states"].append(row)
    per_seed = {}
    for i in TRAIN_SEEDS:
        first, second, keys = {}, {}, []
        for t in read:
            for k in keys_of(t, TRAINED[t]):
                first[(t, *k)], second[(t, *k)] = ev[t][f"N{i}"][k], ev[t]["base"][k]
                keys.append((t, *k))
        per_seed[i] = paired(first, second, keys)
        per_seed[i]["passes"] = per_seed[i]["p"] < ALPHA
    test1 = sum(v["passes"] for v in per_seed.values()) >= MIN_SEEDS
    out["tests"]["test 1"] = {"per_seed": per_seed, "passes": test1}
    test2 = None
    if test1 and 0 in read:
        first, second, keys = {}, {}, []
        for i in TRAIN_SEEDS:
            for k in keys_of(0, SYSTEMATIC_0):
                first[(i, *k)], second[(i, *k)] = ev[0][f"N{i}"][k], ev[0][f"A{i}"][k]
                keys.append((i, *k))
        r = paired(first, second, keys)
        test2 = r["passes"] = r["p"] < ALPHA
        out["tests"]["test 2"] = r
    reported = {}
    for t in read:
        for role, states in (("held out", HELD_OUT[t]), ("easy", EASY[t])):
            for i in TRAIN_SEEDS:
                reported[f"task {t} {role}, N{i} against base"] = paired(ev[t][f"N{i}"], ev[t]["base"], keys_of(t, states))
        if t == 0:
            for i in TRAIN_SEEDS:
                reported[f"task 0 systematic, A{i} against base"] = paired(ev[0][f"A{i}"], ev[0]["base"],
                                                                           keys_of(0, SYSTEMATIC_0))
    out["reported"] = reported
    out["decision"] = reading(True, read, test1, test2)
    return out


def format_report(r: dict) -> str:
    lines = [f"step 10: can constant-noise successes be learned into the weights? commit {r['provenance']['commit']}"]
    for t, h in r["harness"].items():
        lines.append(f"harness task {t}: base {h['base']}/30 against {h['reference']} (gap {h['gap']:+d}, "
                     f"{'ok' if h['ok'] else 'FAILED'})")
    for t, v in r["replays"].items():
        same = f"; {v['same_steps']} in the original's number of steps" if "same_steps" in v else ""
        lines.append(f"replays task {t}: {v['succeeded']} of {v['planned']} succeeded again{same}")
    if "read_tasks" in r:
        lines.append(f"tasks read: {r['read_tasks']}")
    for v in r["states"]:
        cells = ", ".join(f"{p} {v[p][0]} ({100 * v[p][1][0]:.1f}-{100 * v[p][1][1]:.1f}%)"
                          for p in policies(v["task"]) if p in v)
        lines.append(f"task {v['task']} state {v['state']} ({v['role']}), of {v['episodes']}: {cells}")
    if "test 1" in r["tests"]:
        for i, v in r["tests"]["test 1"]["per_seed"].items():
            lines.append(f"test 1, N{i} against base on trained states: {v['first']} against {v['second']} of {v['n']} "
                         f"(only N {v['first_only']}, only base {v['second_only']}), one-sided p {v['p']:.3g}")
        lines.append(f"test 1 {'passes' if r['tests']['test 1']['passes'] else 'does not pass'} "
                     f"(p < {ALPHA} in at least {MIN_SEEDS} of {len(TRAIN_SEEDS)} seeds)")
    if "test 2" in r["tests"]:
        v = r["tests"]["test 2"]
        lines.append(f"test 2, N against A on task 0's systematic states, pooled over seeds: {v['first']} against "
                     f"{v['second']} of {v['n']} (only N {v['first_only']}, only A {v['second_only']}), one-sided p "
                     f"{v['p']:.3g}, {'passes' if v['passes'] else 'does not pass'}")
    for name, v in r.get("reported", {}).items():
        lines.append(f"reported, {name}: {v['first']} against {v['second']} of {v['n']}, one-sided p {v['p']:.3g}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root")
    p.add_argument("--gate", choices=("harness",), help="exit 1 if the run fails this gate")
    p.add_argument("--replay_jobs", action="store_true", help="print task, family, state, noise ids, jitter seeds or -")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.replay_jobs:
        for t, f, s, ids, js in replay_jobs(replay_plan()):
            print(t, f, s, ids, js or "-")
        return 0
    if not a.root:
        p.error("--root is required")
    if a.gate:
        reason = gate(a.root)
        print(f"STOP: {reason}" if reason else f"gate {a.gate} passed")
        return 1 if reason else 0
    r = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1, default=list) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
