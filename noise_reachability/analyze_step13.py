#!/usr/bin/env python3
"""Step 13 analysis (README.md, step 13): do the distilled adapters change success on initial states they never trained
on? Standard library.

    python3 -S noise_reachability/analyze_step13.py --root <run folder> --out step13_report.txt
    python3 -S noise_reachability/analyze_step13.py --root <run folder> --gate harness   # exit 1 stops the session

<run folder> holds harness/session<k>/task_<t>/ (N<t>_0 on the Colab T4 check's step-10 episodes, every session; the
notebook runs it in harness_pending/ and moves it here only when it passes, or to harness_failed/, and a session adds
evaluation episodes only after its harness passed) and
eval/<policy>/task_<t>/part<j>/per_episode_<base|lora>.csv for the base, N0-N2 and R0-R2 (R not on task 7: R7_s's weights
are N7_s's, so its episodes are N7_s's).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_colab_t4_check import step10_rows  # noqa: E402
from analyze_step7 import POLICY_REVISION, clopper_pearson  # noqa: E402
from analyze_step10 import DIRTY_KEYS  # noqa: E402
from analyze_step12 import HARNESS_SEEDS, HARNESS_STATE, STEP10, _rows, paired, train_episodes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP12 = REPO_ROOT / "week6" / "noise_reachability_step12_kaggle" / "run"
TASKS = (0, 6, 7)
RANGE = tuple(range(30, 50))
TRAINED = {0: (2, 5, 6, 7, 8, 11), 6: (8, 12), 7: (33,)}   # every state in an N or R training set (committed provenance)
STATES = {t: tuple(s for s in RANGE if s not in TRAINED[t]) for t in TASKS}
SEEDS = tuple(range(200, 208))                              # no (task, state, seed) episode of step 13 ran before
PARTS = ["200-203", "204-207"]
TRAIN_SEEDS = (0, 1, 2)
MIN_SEEDS = 2
POLICIES = ["base"] + [f"{x}{s}" for x in "NR" for s in TRAIN_SEEDS]
SHARED = {7: "N"}                  # task: R<t>_s's weights are this family's, so R's episodes there are N's
HARNESS_MIN_SAME = 11              # of 12 episodes identical to step 10 in success and steps, every session
# sha256 of the N and R adapters' files (archive MANIFEST.sha256 of noise-reachability-step-10 and
# expert-iteration-step-12, 2026-11-15); the notebook checks the Drive copies against these
ADAPTER_SHA256 = {
    "N0_0": {
        "adapter_config.json": "4f512e8d34de097adadb1d1e76324556a58d5a486dd98169929a772801a146fd",
        "adapter_model.safetensors": "aa163591bfc41a54da6d711908d828f7fe5b9a8d7d9f8895f1392c07e3e2d48d",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "N0_1": {
        "adapter_config.json": "384a41facbd1070d02158a334ec5b3d4ddebcdbd9a66361755450a2a6ab09ff7",
        "adapter_model.safetensors": "7b7c89e575a72cdee3084260acb3a9642f8a6a6ef4511147ecdb4da43123c57b",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "N0_2": {
        "adapter_config.json": "4f6630bd090b14e00deab8a0a79a37a924c64a28aa23f0d58ac743372799b404",
        "adapter_model.safetensors": "295237a3a12988fdfceaa5643e8c58a78cb5f1c16d7c84ddc1f5ee9b655e4e41",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "N6_0": {
        "adapter_config.json": "6e3e0f49f27fc0fef5a0d76761a971b2c5c4feec91ea81ee8cd41de657c69cc3",
        "adapter_model.safetensors": "1eb3b7d5321c1a1d7455bc491920a2f2dfcf9542b81a91751163139c16eb0498",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "N6_1": {
        "adapter_config.json": "4cb696c73129fc0e437255d80aa7e69a902d9742ac3c51657e0f5697dac0002e",
        "adapter_model.safetensors": "89b9a705b7d1ebdea8b4c180e5618f95cf4fbc24d9ce3d67c2a6bd167c931c52",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "N6_2": {
        "adapter_config.json": "ac688cf422ae58c00b8e992d41fbd1056e397124100b198e8f1866dba7ed1432",
        "adapter_model.safetensors": "71e65d76a9b74ac50b0748ceddeb1c7526cf89e03959ed87b4c666a0ef7fdbe7",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "N7_0": {
        "adapter_config.json": "33f536c16b513ab6673fa7106a9e38be80df456a246ccef0131f4d85aefdc91e",
        "adapter_model.safetensors": "49c56ea1ea44cb6d4dd50c6362f4304531af410c355b03917c5cafa94bd00c40",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
    "N7_1": {
        "adapter_config.json": "903aae691044be234e4887dd43dce0c59962576684ce91f10451380009f5b7cd",
        "adapter_model.safetensors": "497fa7e300c2a3650a3091016f92f6607400a0cdd13ad56f065e4efdb1347366",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
    "N7_2": {
        "adapter_config.json": "cc5ca0f30c9e85ef175afaa668532da3c7bcd92b60fa6e164ef1212a5e57b5c0",
        "adapter_model.safetensors": "41a132f12b07fd7652b0d0b8692051eb1a8a4887e578552a832122aab3db7ea3",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
    "R0_0": {
        "adapter_config.json": "c05ec2f485970f353ad97fe5989a05d430168e7609149b7b1940daf5b9b0e827",
        "adapter_model.safetensors": "5eef27693909dfafe5b49335fbb1398a6f73110eeb4eb4064009bb5843ae6d0c",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "R0_1": {
        "adapter_config.json": "d41a94803aa215de7fa29329bc944827e83ea9f293750141bc836437e4714270",
        "adapter_model.safetensors": "26351fd57db533b16b798cc3b61eb48b0bdb10ee3a98b328a94461cd0ea65774",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "R0_2": {
        "adapter_config.json": "312d708ea0eae4b6aad0858546f53199630038619c3434927e5ee8ba8756a93a",
        "adapter_model.safetensors": "d9d605d7d32d493c82c41a00df1091ba16d62abd60889c92b17d7e6fd399eebb",
        "run_config.json": "faf912a80ab9fd77669633970d54325c0d51fbe8f6bad968e1e34d7448cd0f01",
    },
    "R6_0": {
        "adapter_config.json": "a8d2f91f3aa49ec89d1b7bc46cd66a22406f738c10db9ae414fc6e5154b3062e",
        "adapter_model.safetensors": "2fad4bb1d99d12fca8235a699f26e7f271e925b9b76b0feb41f38cefa098bfb4",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "R6_1": {
        "adapter_config.json": "b412ef733b3becd20f2bb3f5dd7e7708f6686326bd6043f306f7a46ea38a559c",
        "adapter_model.safetensors": "9d2dc443e06ebca0e034f9bc8465b2092ed4f49351eee708758c39c89c5fb8ab",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "R6_2": {
        "adapter_config.json": "284e6f8fe7b8a98c6a18ad39b1656aafe5fb3af8ad3ef6e45ab4b2ad83e018ca",
        "adapter_model.safetensors": "4a98d39be880b2f717ef77e9a954af63032a5d90ca141fb02e7dfd0cade345f7",
        "run_config.json": "16af5136a897ace291ef2b255db723f3614b81d318b7df73ee16977a1e7bed8c",
    },
    "R7_0": {
        "adapter_config.json": "94e9c4ac6d237e221e114d5dd044c413072aa1218c398a09de28996aad039460",
        "adapter_model.safetensors": "49c56ea1ea44cb6d4dd50c6362f4304531af410c355b03917c5cafa94bd00c40",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
    "R7_1": {
        "adapter_config.json": "5337ac193ddaafc0e7b30b601eede1b8edfcb4fab7a0731048e5897a9438a377",
        "adapter_model.safetensors": "497fa7e300c2a3650a3091016f92f6607400a0cdd13ad56f065e4efdb1347366",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
    "R7_2": {
        "adapter_config.json": "2f5511492a8adfbac42c40088f04d8b8180d471c4c15eec8a1d3af9be858d0d3",
        "adapter_model.safetensors": "41a132f12b07fd7652b0d0b8692051eb1a8a4887e578552a832122aab3db7ea3",
        "run_config.json": "550aad6fc781c7bf130daede67413757343cbe29c1b4ebab4ccbe7047c0f3f1a",
    },
}
# the identity SHARED relies on
WEIGHTS_SHA256 = {k: v["adapter_model.safetensors"] for k, v in ADAPTER_SHA256.items()}


def run_policies(t: int) -> list[str]:
    """The policies evaluated on task t (R is not run where its weights are another family's)."""
    return [p for p in POLICIES if not (p[0] == "R" and t in SHARED)]


def adapter_path(t: int, policy: str) -> str:
    return {"N": f"step10/runs/N{t}_{policy[1:]}/adapter", "R": f"step12/runs/R{t}_{policy[1:]}/adapter"}[policy[0]]


def suffix(policy: str) -> str:
    return "base" if policy == "base" else "lora"


def spec(states) -> str:
    return ",".join(map(str, states))


def trained_states(step10=STEP10, step12=STEP12) -> dict[int, tuple[int, ...]]:
    """Every initial state in a training set of N<t>_s (step 10) or R<t>_s (step 12), from their committed provenance."""
    out = {}
    for t in TASKS:
        names = []
        for s in TRAIN_SEEDS:
            names += train_episodes(Path(step10, "runs", f"N{t}_{s}", "provenance.jsonl"))
            names += train_episodes(Path(step12, "runs", f"R{t}_{s}", "provenance.jsonl"))
        out[t] = tuple(sorted({int(re.search(r"_init(\d+)[_.]", n).group(1)) for n in names}))
    return out


def shared_weights_identical() -> bool:
    return all(WEIGHTS_SHA256[f"R{t}_{s}"] == WEIGHTS_SHA256[f"{SHARED[t]}{t}_{s}"] for t in SHARED for s in TRAIN_SEEDS)


# ---- provenance ------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def check_settings(root, f: Path, r: dict) -> None:
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    _require(st.get("stage") == "eval", f, "not an evaluation")
    if rel[0] == "harness":
        _require(re.fullmatch(r"session\d+", rel[1]) is not None and rel[2] == f"task_{st['task_id']}", f, "folder")
        t = st["task_id"]
        _require(st["eval_policies"] == "lora" and str(st["lora_dir"]).endswith(f"step10/runs/N{t}_0/adapter")
                 and st["eval_init_states"] == str(HARNESS_STATE[t])
                 and st["eval_seeds"] == f"{HARNESS_SEEDS[0]}-{HARNESS_SEEDS[-1]}", f, "harness settings")
        return
    _require(rel[0] == "eval" and len(rel) == 4, f, "an unexpected folder")
    policy, task, j = rel[1], rel[2], int(rel[3].removeprefix("part"))
    t = int(task.removeprefix("task_"))
    _require(st["task_id"] == t and t in TASKS and policy in run_policies(t) and 0 <= j < len(PARTS), f,
             f"policy {policy} on task {t}")
    _require(st["eval_init_states"] == spec(STATES[t]) and st["eval_seeds"] == PARTS[j], f, "states or seeds")
    if policy == "base":
        _require(st["eval_policies"] == "base", f, "the base must run without an adapter")
    else:
        _require(st["eval_policies"] == "lora" and str(st["lora_dir"]).endswith(adapter_path(t, policy)), f,
                 f"adapter {st.get('lora_dir')}")


def check_provenance(root) -> dict:
    """One commit, clean trees, T4 GPUs only, n_action_steps 10, 520 steps and one checkpoint in every record; each
    record has the protocol's settings for its folder."""
    records = []
    files = sorted(f for top in ("harness", "eval") for f in Path(root, top).rglob("provenance.jsonl"))
    for f in files:
        for line in open(f):
            r = json.loads(line)
            check_settings(root, f, r)
            flags = [r[k] for k in DIRTY_KEYS if k in r]
            _require(r.get("git_dirty") is False and flags and all(v is False for v in flags), f, "a dirty tree")
            _require(r["settings"]["n_action_steps"] == 10 and r["settings"]["max_steps"] == 520, f,
                     "n_action_steps or max_steps")
            _require(bool(r.get("gpus")) and all("T4" in g for g in r["gpus"]), f, f"GPUs {r.get('gpus')}")
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

def harness_session(folder: Path, step10=STEP10) -> dict:
    """N<t>_0 on the 12 episodes of the Colab T4 check, compared with step 10's rows."""
    same, per = 0, {}
    for t in TASKS:
        path = folder / f"task_{t}" / "per_episode_lora.csv"
        new = _rows(path) if path.exists() else {}
        old = step10_rows(t, "N0", step10)
        if set(new) != set(old):
            raise ValueError(f"{folder.name} task {t}: harness has {sorted(new)}, expected {sorted(old)}")
        per[t] = sum(new[e] == old[e] for e in old)
        same += per[t]
    return {"same": same, "episodes": len(TASKS) * len(HARNESS_SEEDS), "per_task": per,
            "passes": same >= HARNESS_MIN_SAME}


def harness(root, step10=STEP10) -> dict:
    sessions = sorted(Path(root, "harness").glob("session*"), key=lambda p: int(p.name.removeprefix("session")))
    if not sessions:
        raise ValueError("no harness session")
    out = {p.name: harness_session(p, step10) for p in sessions}
    return {"sessions": out, "passes": all(v["passes"] for v in out.values())}


def packages_of(folder: Path) -> set[str]:
    return {json.dumps(json.loads(line).get("packages"), sort_keys=True)
            for f in sorted(folder.rglob("provenance.jsonl")) for line in open(f)}


def gate(root, folder: str) -> str | None:
    """One session's harness, in ``folder`` (relative to the run folder, e.g. harness_pending/session3): it must
    reproduce step 10, and its package versions must equal those of the sessions already accepted, so a session on
    another PyTorch build stops before it adds episodes (the analysis refuses mixed packages)."""
    h = harness_session(Path(root, folder))
    if not h["passes"]:
        return f"harness {folder}: {h['same']} of {h['episodes']} episodes reproduce step 10"
    here, before = packages_of(Path(root, folder)), packages_of(Path(root, "harness"))
    if len(here) != 1 or (before and here != before):
        return f"harness {folder}: package versions {sorted(here)} differ from the accepted sessions' {sorted(before)}"
    return None


def failed_sessions(root) -> list[str]:
    return sorted(p.name for p in Path(root, "harness_failed").glob("session*"))


# ---- evaluation ------------------------------------------------------------------------------------------------

def load_eval(root, policy: str, t: int) -> dict[tuple[int, int], int]:
    out = {}
    for path in sorted(glob.glob(str(Path(root, "eval", policy, f"task_{t}", "part*", f"per_episode_{suffix(policy)}.csv")))):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["seed"]), int(r["init_state"]))
                if int(r["task_id"]) != t:
                    raise ValueError(f"{path}: an episode of task {r['task_id']}")
                if key in out:
                    raise ValueError(f"{path}: episode {key} twice")
                out[key] = int(r["success"])
    want = {(s, i) for s in SEEDS for i in STATES[t]}
    if set(out) != want:
        raise ValueError(f"{policy} task {t}: {len(set(out) & want)} of {len(want)} episodes and "
                         f"{len(set(out) - want)} others; finish the run")
    return out


def load_all(root) -> dict[str, dict]:
    """{policy: {(task, seed, state): success}} over all tasks; R on a SHARED task takes that family's episodes."""
    if SHARED and not shared_weights_identical():
        raise ValueError("R's weights differ from the shared family's: R must be run on that task")
    for t in SHARED:
        if any(Path(root, "eval", f"R{s}", f"task_{t}").exists() for s in TRAIN_SEEDS):
            raise ValueError(f"R was run on task {t}, where the protocol uses {SHARED[t]}'s episodes")
    out = {p: {} for p in POLICIES}
    for t in TASKS:
        for p in POLICIES:
            src = f"{SHARED[t]}{p[1:]}" if (p[0] == "R" and t in SHARED) else p
            for k, v in load_eval(root, src, t).items():
                out[p][(t,) + k] = v
    return out


def changes(ev: dict, family: str) -> dict:
    """``family``<s> against the base per training seed, two-sided; a change needs 2 of 3 seeds with one sign."""
    per = {}
    for s in TRAIN_SEEDS:
        v = paired(ev[f"{family}{s}"], ev["base"], sorted(ev["base"]))
        v["significant"] = v["p_two_sided"] < 0.05
        v["sign"] = (v["first_only"] > v["second_only"]) - (v["first_only"] < v["second_only"])
        per[str(s)] = v
    ups = sum(v["significant"] and v["sign"] > 0 for v in per.values())
    downs = sum(v["significant"] and v["sign"] < 0 for v in per.values())
    direction = "helps" if ups >= MIN_SEEDS else ("hurts" if downs >= MIN_SEEDS else None)
    return {"per_seed": per, "changes": direction is not None, "direction": direction}


def reading(harness_ok: bool, n: dict, r: dict) -> str:
    if not harness_ok:
        return "not read: a session's harness failed"
    parts = []
    for name, v in (("N (round 1)", n), ("R (round 2)", r)):
        parts.append(f"{name} {v['direction']} task success on unseen initial states" if v["changes"] else
                     f"{name}: no change of 10 points or more shown on unseen initial states")
    return "; ".join(parts)


def analyse(root, step10=STEP10) -> dict:
    prov = check_provenance(root)
    h = harness(root, step10)
    ev = load_all(root)
    n, r = changes(ev, "N"), changes(ev, "R")
    per_task = {}
    for t in TASKS:
        keys = [k for k in ev["base"] if k[0] == t]
        per_task[str(t)] = {}
        for p in POLICIES:
            k = sum(ev[p][x] for x in keys)
            per_task[str(t)][p] = [k, len(keys), list(clopper_pearson(k, len(keys)))]
    states = []
    for t in TASKS:
        for i in STATES[t]:
            keys = [(t, s, i) for s in SEEDS]
            row = {"task": t, "state": i}
            for p in POLICIES:
                k = sum(ev[p][x] for x in keys)
                row[p] = [k, list(clopper_pearson(k, len(keys)))]
            states.append(row)
    r_vs_n = {}
    for s in TRAIN_SEEDS:
        v = paired(ev[f"R{s}"], ev[f"N{s}"], sorted(ev["base"]))
        r_vs_n[str(s)] = {k: v[k] for k in ("n", "first", "second", "first_only", "second_only", "p_two_sided")}
    pooled = {x: sum(sum(ev[f"{x}{s}"].values()) for s in TRAIN_SEEDS) for x in "NR"}
    pooled["base x 3"] = 3 * sum(ev["base"].values())
    h["failed_sessions"] = failed_sessions(root)
    return {"provenance": prov, "harness": h, "trained_states": {str(t): list(v) for t, v in TRAINED.items()},
            "tests": {"N against base": n, "R against base": r}, "per_task": per_task, "states": states,
            "reported": {"R against N": r_vs_n, "pooled counts (no pooled p: the seeds share their episodes)": pooled},
            "decision": reading(h["passes"], n, r)}


def format_report(r: dict) -> str:
    lines = [f"step 13: do the distilled adapters change success on unseen initial states? commit "
             f"{r['provenance']['commit']}",
             "harness: " + ", ".join(f"{k} {v['same']}/{v['episodes']}" for k, v in r["harness"]["sessions"].items())
             + f" ({'ok' if r['harness']['passes'] else 'FAILED'}; at least {HARNESS_MIN_SAME} each); sessions stopped by a "
             f"failed harness, without episodes: {', '.join(r['harness']['failed_sessions']) or 'none'}"]
    for t, v in r["per_task"].items():
        lines.append(f"task {t}: " + ", ".join(f"{p} {k}/{n}" for p, (k, n, _) in v.items()))
    for name, v in r["tests"].items():
        for s, x in v["per_seed"].items():
            lines.append(f"{name}, seed {s}: {x['first']} against {x['second']} of {x['n']} (only adapter "
                         f"{x['first_only']}, only base {x['second_only']}), two-sided p {x['p_two_sided']:.3g}")
        lines.append(f"{name}: " + (f"changes ({v['direction']})" if v["changes"] else "no change shown")
                     + f" (p < 0.05 in at least {MIN_SEEDS} of {len(TRAIN_SEEDS)} seeds, one sign)")
    for s, x in r["reported"]["R against N"].items():
        lines.append(f"reported, R{s} against N{s}: {x['first']} against {x['second']} of {x['n']}, two-sided p "
                     f"{x['p_two_sided']:.3g}")
    lines.append("reported, pooled counts: " + ", ".join(
        f"{k} {v}" for k, v in r["reported"]["pooled counts (no pooled p: the seeds share their episodes)"].items()))
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", help="check one session's harness folder, e.g. harness_pending/session3")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.gate:
        try:
            g = gate(a.root, a.gate)
        except ValueError as e:              # incomplete or malformed: not a verdict on the stack (exit 2, not 1)
            print(f"gate {a.gate} could not be read: {e}")
            return 2
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
