#!/usr/bin/env python3
"""Step 12 analysis (README.md, step 12): does a second round of constant-noise search and training improve on the
first? Standard library.

    python3 -S noise_reachability/analyze_step12.py --root <run folder> --out step12_report.txt
    python3 -S noise_reachability/analyze_step12.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step12.py --selection                          # states from step 10's report

<run folder>/task_<t>/ holds harness/ (round 1's N<t>_0 on four of step 10's episodes per task), search/ (the round-1
explorer with fixed ``repeat`` noises; recordings in search_data/episodes/), round2_data/ (round 1's recordings and
round 2's, which R trains on), and eval/<policy>/part<j>/per_episode_lora.csv for N0-N2 (step 10's adapters) and
R0-R2 (round 2's). runs/R<t>_<s>/ holds round 2's training runs.
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
from analyze_step7 import POLICY_REVISION, clopper_pearson  # noqa: E402
from analyze_step10 import DIRTY_KEYS, TRAIN_SETTINGS, mcnemar_greater  # noqa: E402
from deep_noise_search import record_name  # noqa: E402  (standard library at import time)

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP10 = REPO_ROOT / "week6" / "noise_reachability_step10_kaggle" / "run"
TASKS = (0, 6, 7)
SELECT_MAX = 0.30          # a resistant state is searched when round 1's N0-N2 pooled success is at most this
SEARCHED = {0: (2, 5, 6, 11), 6: (8, 12, 14), 7: (38,)}
HELD_OUT = {0: (), 6: (20,), 7: (47,)}          # selected like the searched states but never searched: transfer
CHECK = {0: (8,), 6: (), 7: (33,)}              # round 1's best trained states: does round 2 keep them?
SEARCH_FAMILY = "repeat"
SEARCH_IDS = tuple(range(7000, 7050))           # fresh: earlier steps used ids below 1,500 and 6000-6049
EPISODE_SEED_BASE = 1000                        # deep_noise_search: noise id k runs with episode seed 1000 + k
EXPLORER = 0                                    # round 1's adapter N<t>_0 searches
TRAIN_SEEDS = (0, 1, 2)
MIN_SEEDS = 2
EVAL_SEEDS = {0: tuple(range(48, 64)), 6: tuple(range(48, 80)), 7: tuple(range(48, 80))}   # fresh, not step 10's
EVAL_SEED_PARTS = {t: [f"{a}-{a + 3}" for a in range(EVAL_SEEDS[t][0], EVAL_SEEDS[t][-1] + 1, 4)] for t in EVAL_SEEDS}
HARNESS_STATE = {0: 8, 6: 8, 7: 33}             # N<t>_0 on step 10's episodes of these states, seeds 16-19
HARNESS_SEEDS = (16, 17, 18, 19)
HARNESS_MIN_SAME = 11                           # of 12 episodes identical to step 10 in success and steps
POLICIES = [f"N{s}" for s in TRAIN_SEEDS] + [f"R{s}" for s in TRAIN_SEEDS]


def eval_states(task: int) -> tuple[int, ...]:
    return tuple(sorted(SEARCHED[task] + HELD_OUT[task] + CHECK[task]))


EVAL_STATE_SPECS = {t: ",".join(map(str, eval_states(t))) for t in TASKS}


def episodes_per_seed(task: int) -> int:
    """Episodes per state per policy."""
    return len(EVAL_SEEDS[task])


def adapter_of(task: int, policy: str) -> str:
    """N<s> is step 10's adapter N<t>_<s>; R<s> is round 2's R<t>_<s>."""
    return f"{policy[0]}{task}_{policy[1:]}"


def task_root(root, task) -> Path:
    return Path(root) / f"task_{task}"


# ---- the states, from step 10 ----------------------------------------------------------------------------------

def select_states(report: dict) -> tuple[dict, dict]:
    """(searched, held out) by the rule fixed before the run: resistant states (step 10's trained and held-out roles)
    of the tasks read with N0-N2 pooled success at most SELECT_MAX; per task, the highest-numbered such held-out state
    stays held out and the rest are searched. Nothing is chosen on easy states."""
    searched = {t: [] for t in TASKS}
    low_held = {t: [] for t in TASKS}
    for s in report["states"]:
        t, i = s["task"], s["state"]
        if t not in TASKS or s["role"] not in ("trained", "held out"):
            continue
        rate = sum(s[f"N{k}"][0] for k in TRAIN_SEEDS) / (len(TRAIN_SEEDS) * s["episodes"])
        if rate <= SELECT_MAX:
            (searched if s["role"] == "trained" else low_held)[t].append(i)
    held = {t: tuple(sorted(low_held[t])[-1:]) for t in TASKS}
    for t in TASKS:
        searched[t] = tuple(sorted(searched[t] + sorted(low_held[t])[:-1]))
    return searched, held


# ---- provenance ------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def check_settings(root, t: int, f: Path, r: dict) -> None:
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    if rel[0] == "runs":
        name = rel[1]
        _require(re.fullmatch(r"R\d_\d", name) is not None and st.get("stage") == "train"
                 and st["seed"] == int(name[-1]), f, "not a round-2 training run of this seed")
        _require(str(st["data_dir"]).endswith(f"task_{t}/round2_data"), f, f"data {st['data_dir']}")
        _require(st["val_init_states"] == "", f, "validation states")
        _require(all(st.get(k) == v for k, v in TRAIN_SETTINGS.items()), f, "training settings")
        return
    part = rel[1]
    if part == "harness":
        _require(st.get("stage") == "eval" and st["eval_policies"] == "lora"
                 and str(st["lora_dir"]).endswith(f"step10/runs/N{t}_{EXPLORER}/adapter")
                 and st["eval_init_states"] == str(HARNESS_STATE[t])
                 and st["eval_seeds"] == f"{HARNESS_SEEDS[0]}-{HARNESS_SEEDS[-1]}", f, "harness settings")
    elif part == "search":
        _require(st.get("family") == SEARCH_FAMILY and st.get("jitter_seeds") is None
                 and st["noise_ids"] == f"{SEARCH_IDS[0]}-{SEARCH_IDS[-1]}"
                 and st["init_state"] in SEARCHED[t]
                 and str(st.get("lora_dir")).endswith(f"step10/runs/N{t}_{EXPLORER}/adapter")
                 and str(st.get("record_dir")).endswith(f"task_{t}/search_data")
                 and st.get("record_successes_only") == 1 and st.get("frame_stride") == 2, f, "search settings")
        _require(r.get("git_dirty_sfbc") is False, f, "no clean record of self_filtered_bc (the recorder)")
    elif part == "eval":
        policy, j = rel[2], int(rel[3].removeprefix("part"))
        want = (f"step10/runs/{adapter_of(t, policy)}/adapter" if policy.startswith("N")
                else f"step12/runs/{adapter_of(t, policy)}/adapter")
        _require(policy in POLICIES and st.get("stage") == "eval" and st["eval_policies"] == "lora"
                 and st["eval_init_states"] == EVAL_STATE_SPECS[t] and st["eval_seeds"] == EVAL_SEED_PARTS[t][j]
                 and str(st["lora_dir"]).endswith(want), f, f"evaluation settings ({policy}, {st.get('lora_dir')})")
    else:
        raise ValueError(f"provenance: a record in an unexpected folder {f}")


def check_provenance(root) -> dict:
    """One commit, clean trees, n_action_steps 10, 520 steps and one checkpoint in every record; each record belongs
    to its folder's task and has the protocol's settings for that folder."""
    files = [(t, f) for t in TASKS for f in sorted(task_root(root, t).rglob("provenance.jsonl"))
             if "round2_data" not in f.parts and "search_data" not in f.parts]
    files += [(int(f.parent.name[1]), f) for f in sorted(Path(root, "runs").glob("*/provenance.jsonl"))]
    records = []
    for t, f in files:
        for line in open(f):
            r = json.loads(line)
            st = r["settings"]
            if st["task_id"] != t:
                raise ValueError(f"provenance: task {st['task_id']} in {f}")
            check_settings(root, t, f, r)
            flags = [r[k] for k in DIRTY_KEYS if k in r]
            if r.get("git_dirty") is not False or not flags or any(v is not False for v in flags):
                raise ValueError(f"provenance: a run on a dirty tree in {f}")
            if st["n_action_steps"] != 10 or st["max_steps"] != 520:
                raise ValueError(f"provenance: settings {st} in {f}")
            records.append(r)
    if not records:
        raise ValueError("provenance: no records")
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"provenance: commits {commits}")
    if len({json.dumps(r.get("packages"), sort_keys=True) for r in records}) != 1:
        raise ValueError("provenance: packages changed between records")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1 or POLICY_REVISION not in next(iter(paths)):
        raise ValueError(f"provenance: policy paths {paths}")
    return {"records": len(records), "commit": commits.pop()}


# ---- harness ---------------------------------------------------------------------------------------------------

def _rows(path) -> dict[tuple[int, int], tuple[int, int]]:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["seed"]), int(r["init_state"]))
            if key in out:
                raise ValueError(f"{path}: episode {key} twice")
            out[key] = (int(r["success"]), int(r["steps"]))
    return out


def harness(root, step10=STEP10) -> dict:
    """Round 1's N<t>_0 re-run on step 10's episodes: how many match step 10 exactly (success and steps)."""
    same, total, per = 0, 0, {}
    for t in TASKS:
        old = {}
        for p in sorted(glob.glob(str(task_root(step10, t) / "eval" / "N0" / "part*" / "per_episode_lora.csv"))):
            old.update(_rows(p))
        path = task_root(root, t) / "harness" / "per_episode_lora.csv"
        new = _rows(path) if path.exists() else {}
        want = {(s, HARNESS_STATE[t]) for s in HARNESS_SEEDS}
        if set(new) != want:
            raise ValueError(f"task {t}: harness has {sorted(new)}, expected {sorted(want)}; finish the run")
        k = sum(new[e] == old[e] for e in want)
        per[t] = {"same": k, "episodes": len(want), "successes_now": sum(new[e][0] for e in want),
                  "successes_step10": sum(old[e][0] for e in want)}
        same, total = same + k, total + len(want)
    return {"per_task": per, "same": same, "episodes": total, "passes": same >= HARNESS_MIN_SAME}


def gate(root) -> str | None:
    h = harness(root)
    return None if h["passes"] else f"harness: {h['same']} of {h['episodes']} episodes reproduce step 10"


# ---- search ----------------------------------------------------------------------------------------------------

def search_counts(root) -> dict[tuple[int, int], dict]:
    """{(task, state): {"hits", "episodes", "ids"}} of the explorer's fixed-noise search; refuses gaps and strays."""
    out = {}
    want = {EPISODE_SEED_BASE + k for k in SEARCH_IDS}
    for t in TASKS:
        for s in SEARCHED[t]:
            path = task_root(root, t) / "search" / f"per_episode_{SEARCH_FAMILY}_state{s}.csv"
            rows = read_rows(path) if path.exists() else {}
            if set(rows) != want:
                raise ValueError(f"task {t} state {s}: {len(set(rows) & want)} of {len(want)} search episodes and "
                                 f"{len(set(rows) - want)} others; finish the run")
            hits = sorted(k - EPISODE_SEED_BASE for k, v in rows.items() if v)
            out[(t, s)] = {"hits": len(hits), "episodes": len(want), "ids": hits}
    return out


def train_episodes(provenance: Path) -> list[str]:
    return json.loads(Path(provenance).read_text().splitlines()[0])["settings"]["train_episodes"]


def check_training_data(root, search: dict, step10=STEP10) -> dict:
    """Every round-2 run trained on exactly round 1's training episodes of its seed (step 10's N<t>_<s>, committed)
    plus a recording of every search success of its task, nothing else. Returns {run: (round 1, round 2) counts}."""
    out = {}
    for t in TASKS:
        new = {record_name(SEARCH_FAMILY, k, None, s, EPISODE_SEED_BASE + k) for (task, s), v in search.items()
               if task == t for k in v["ids"]}
        for i in TRAIN_SEEDS:
            name = f"R{t}_{i}"
            path = Path(root, "runs", name, "provenance.jsonl")
            if not path.exists():
                raise ValueError(f"{name}: no training run; finish the run")
            old = set(train_episodes(Path(step10, "runs", f"N{t}_{i}", "provenance.jsonl")))
            got = train_episodes(path)
            if len(got) != len(set(got)) or set(got) != old | new:
                raise ValueError(f"{name}: trained on {len(got)} episodes; expected round 1's {len(old)} plus "
                                 f"{len(new)} search successes (missing {sorted((old | new) - set(got))[:5]}, "
                                 f"extra {sorted(set(got) - old - new)[:5]})")
            out[name] = (len(old), len(new))
    return out


def training_data(root) -> dict[str, dict]:
    """Per round-2 run: training episodes from round 1 (step 10's recordings) and from round 2's search, per state."""
    out = {}
    for f in sorted(Path(root, "runs").glob("R*/provenance.jsonl")):
        names = train_episodes(f)
        per = {}
        for name in names:
            m = re.search(r"_n(\d+)_", name)
            k = int(m.group(1)) if m else None
            state = int(re.search(r"_init(\d+)[_.]", name).group(1))
            which = "round 2" if k is not None and k >= SEARCH_IDS[0] else "round 1"
            per.setdefault(which, {}).setdefault(state, 0)
            per[which][state] += 1
        out[f.parent.name] = per
    return out


# ---- evaluation ------------------------------------------------------------------------------------------------

def load_eval(root, task: int, policy: str) -> dict[tuple[int, int], int]:
    out = {}
    for path in sorted(glob.glob(str(task_root(root, task) / "eval" / policy / "part*" / "per_episode_lora.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["seed"]), int(r["init_state"]))
                if int(r["task_id"]) != task:
                    raise ValueError(f"{path}: an episode of task {r['task_id']}")
                if key in out:
                    raise ValueError(f"{path}: episode {key} twice")
                out[key] = int(r["success"])
    want = set(keys_of(task, eval_states(task)))
    if set(out) != want:
        raise ValueError(f"task {task} {policy}: {len(set(out) & want)} of {len(want)} episodes and "
                         f"{len(set(out) - want)} others; finish the run")
    return out


def keys_of(task: int, states) -> list[tuple[int, int]]:
    return [(s, i) for s in EVAL_SEEDS[task] for i in states]


def mcnemar_two_sided(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def paired(first: dict, second: dict, keys) -> dict:
    keys = list(keys)
    a = sum(1 for k in keys if first[k] and not second[k])
    b = sum(1 for k in keys if second[k] and not first[k])
    return {"n": len(keys), "first": sum(first[k] for k in keys), "second": sum(second[k] for k in keys),
            "first_only": a, "second_only": b, "p": mcnemar_greater(a, b), "p_two_sided": mcnemar_two_sided(a, b)}


def pooled(ev: dict, task_states: dict, first: str, second: str) -> dict:
    """``first``<s> against ``second``<s> pooled over seeds and tasks on ``task_states``: pairs keyed (seed index, task,
    episode)."""
    a, b = {}, {}
    for i in TRAIN_SEEDS:
        for t, states in task_states.items():
            for k in keys_of(t, states):
                a[(i, t) + k] = ev[t][f"{first}{i}"][k]
                b[(i, t) + k] = ev[t][f"{second}{i}"][k]
    return paired(a, b, a)


def reading(harness_ok: bool, test1: bool) -> str:
    if not harness_ok:
        return "not read: the harness failed"
    if test1:
        return ("a second round of constant-noise search and training improves on the first on the states the first "
                "left weak")
    return "not shown: a second round did not improve on the first on the states it left weak"


def analyse(root, step10=STEP10) -> dict:
    prov = check_provenance(root)
    h = harness(root, step10)
    search = search_counts(root)
    ev = {t: {p: load_eval(root, t, p) for p in POLICIES} for t in TASKS}
    per_seed = {}
    for i in TRAIN_SEEDS:
        r_ = {(t,) + k: ev[t][f"R{i}"][k] for t in TASKS for k in keys_of(t, SEARCHED[t])}
        n_ = {(t,) + k: ev[t][f"N{i}"][k] for t in TASKS for k in keys_of(t, SEARCHED[t])}
        per_seed[str(i)] = paired(r_, n_, r_)
        per_seed[str(i)]["passes"] = per_seed[str(i)]["p"] < 0.05
    test1 = sum(v["passes"] for v in per_seed.values()) >= MIN_SEEDS
    states = []
    for t in TASKS:
        for s in eval_states(t):
            role = "searched" if s in SEARCHED[t] else ("held out" if s in HELD_OUT[t] else "check")
            row = {"task": t, "state": s, "role": role, "episodes": episodes_per_seed(t)}
            for p in POLICIES:
                k = sum(ev[t][p][e] for e in keys_of(t, (s,)))
                row[p] = [k, list(clopper_pearson(k, episodes_per_seed(t)))]
            if s in SEARCHED[t]:
                row["search_hits"] = search[(t, s)]["hits"]
            states.append(row)
    reported = {"held out, R against N, seeds pooled": pooled(ev, {t: HELD_OUT[t] for t in TASKS if HELD_OUT[t]}, "R", "N"),
                "check states, R against N, seeds pooled": pooled(ev, {t: CHECK[t] for t in TASKS if CHECK[t]}, "R", "N"),
                "searched, R against N, seeds pooled": pooled(ev, SEARCHED, "R", "N")}
    return {"provenance": prov, "harness": h, "search": {f"{t}/{s}": v for (t, s), v in search.items()},
            "training_data": training_data(root), "training_sets": check_training_data(root, search, step10),
            "states": states,
            "tests": {"test 1": {"per_seed": per_seed, "passes": test1}}, "reported": reported,
            "decision": reading(h["passes"], test1)}


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"step 12: a second round of constant-noise search and training? commit {r['provenance']['commit']}",
             f"harness: {h['same']} of {h['episodes']} of round 1's episodes reproduce step 10 "
             f"({'ok' if h['passes'] else 'FAILED'}; per task "
             + ", ".join(f"{t} {v['same']}/{v['episodes']}" for t, v in h["per_task"].items()) + ")"]
    for k, v in r["search"].items():
        lines.append(f"search {k}: {v['hits']} of {v['episodes']} fixed noises succeeded")
    for name, v in r["training_data"].items():
        lines.append(f"training data {name}: " + "; ".join(f"{w} {d}" for w, d in sorted(v.items())))
    for s in r["states"]:
        cells = ", ".join(f"{p} {s[p][0]} ({100 * s[p][1][0]:.1f}-{100 * s[p][1][1]:.1f}%)" for p in POLICIES)
        lines.append(f"task {s['task']} state {s['state']} ({s['role']}), of {s['episodes']}: {cells}")
    for i, v in r["tests"]["test 1"]["per_seed"].items():
        lines.append(f"test 1, R{i} against N{i} on searched states: {v['first']} against {v['second']} of {v['n']} "
                     f"(only R {v['first_only']}, only N {v['second_only']}), one-sided p {v['p']:.3g}")
    lines.append(f"test 1 {'passes' if r['tests']['test 1']['passes'] else 'does not pass'} "
                 f"(p < 0.05 in at least {MIN_SEEDS} of {len(TRAIN_SEEDS)} seeds)")
    for k, v in r["reported"].items():
        lines.append(f"reported, {k}: {v['first']} against {v['second']} of {v['n']} (only R {v['first_only']}, "
                     f"only N {v['second_only']}), one-sided p {v['p']:.3g}, two-sided p {v['p_two_sided']:.3g}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--gate", choices=["harness"])
    ap.add_argument("--selection", action="store_true", help="print the states the rule picks from step 10's report")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.selection:
        searched, held = select_states(json.loads((STEP10 / "step10_report.txt.json").read_text()))
        print(json.dumps({"searched": searched, "held_out": held}))
        return 0 if (searched, held) == ({t: SEARCHED[t] for t in TASKS}, {t: HELD_OUT[t] for t in TASKS}) else 1
    if a.gate:
        g = gate(a.root)
        print(f"gate {a.gate} " + ("passed" if g is None else f"FAILED: {g}"))
        return 0 if g is None else 1
    r = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
