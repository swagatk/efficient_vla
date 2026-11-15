#!/usr/bin/env python3
"""Step 14 analysis (README.md, step 14): is round 2's loss on 0/8 real, and is it dilution? Standard library.

    python3 -S noise_reachability/analyze_step14.py --root <run folder> --out step14_report.txt
    python3 -S noise_reachability/analyze_step14.py --root <run folder> --gate harness   # exit 1 stops the run

<run folder> holds harness/<N0|R0>/ (step 10's N0_0 and step 12's R0_0 on their own committed episodes of 0/8),
runs/L0_<s>/ (round 2's training set, trained for L_STEPS), eval_check/<policy>/part<j>/ (0/8, N0-N2, R0-R2, L0-L2) and
eval_searched/<policy>/part<j>/ (step 12's searched task-0 states, R0-R2 and L0-L2), each with per_episode_lora.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step7 import POLICY_REVISION, clopper_pearson  # noqa: E402
from analyze_step10 import DIRTY_KEYS, TRAIN_SETTINGS  # noqa: E402
from analyze_step12 import STEP10, _rows, paired, train_episodes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP12 = REPO_ROOT / "week6" / "noise_reachability_step12_kaggle" / "run"
TASK = 0
CHECK_STATE = 8                                  # round 1's best trained state of task 0, where round 2 lost ground
SEARCHED = (2, 5, 6, 11)                         # step 12's searched task-0 states: does longer training keep the gain?
TRAIN_SEEDS = (0, 1, 2)
MIN_SEEDS = 2
L_STEPS = 2671                                   # 2,000 x 7,309 / 5,473 frames: round 1's samples per frame on R's set
CHECK_SEEDS = tuple(range(80, 176))              # fresh: step 10 used 16-47, step 12 48-79
SEARCHED_SEEDS = tuple(range(80, 96))
CHECK_PARTS = [f"{a}-{a + 15}" for a in range(CHECK_SEEDS[0], CHECK_SEEDS[-1] + 1, 16)]
SEARCHED_PARTS = [f"{a}-{a + 3}" for a in range(SEARCHED_SEEDS[0], SEARCHED_SEEDS[-1] + 1, 4)]
CHECK_POLICIES = [f"{x}{s}" for x in "NRL" for s in TRAIN_SEEDS]
SEARCHED_POLICIES = [f"{x}{s}" for x in "RL" for s in TRAIN_SEEDS]
HARNESS = {"N0": ("16-19", STEP10, "N0"), "R0": ("48-51", STEP12, "R0")}   # policy: (seeds, committed run, its folder)
HARNESS_MIN_SAME = 7                             # of 8 episodes identical to the committed rows in success and steps
SETS = {"eval_check": ((CHECK_STATE,), CHECK_SEEDS, CHECK_PARTS, CHECK_POLICIES),
        "eval_searched": (SEARCHED, SEARCHED_SEEDS, SEARCHED_PARTS, SEARCHED_POLICIES)}


def spec(states) -> str:
    return ",".join(map(str, states))


def adapter_path(policy: str) -> str:
    """The adapter folder a policy's evaluation must use (suffix of its lora_dir)."""
    s = policy[1:]
    return {"N": f"step10/runs/N{TASK}_{s}/adapter", "R": f"step12/runs/R{TASK}_{s}/adapter",
            "L": f"step14/runs/L{TASK}_{s}/adapter"}[policy[0]]


def seeds_of(part: str) -> list[int]:
    a, b = part.split("-")
    return list(range(int(a), int(b) + 1))


# ---- provenance ------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def check_settings(root, f: Path, r: dict) -> None:
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    if rel[0] == "runs":
        name = rel[1]
        _require(name in {f"L{TASK}_{s}" for s in TRAIN_SEEDS} and st.get("stage") == "train"
                 and st["seed"] == int(name[-1]), f, "not a step-14 training run of this seed")
        _require(str(st["data_dir"]).endswith(f"task_{TASK}/round2_data"), f, f"data {st['data_dir']}")
        _require(st["val_init_states"] == "", f, "validation states")
        want = dict(TRAIN_SETTINGS, train_steps=L_STEPS)
        _require(all(st.get(k) == v for k, v in want.items()), f, "training settings")
        _require(sorted(st.get("train_episode_sha256") or {}) == sorted(st["train_episodes"]), f,
                 "a sha256 per training episode")
        return
    _require(st.get("stage") == "eval" and st["eval_policies"] == "lora", f, "not a LoRA evaluation")
    if rel[0] == "harness":
        seeds, _, folder = HARNESS[rel[1]]
        _require(st["eval_init_states"] == str(CHECK_STATE) and st["eval_seeds"] == seeds
                 and str(st["lora_dir"]).endswith(adapter_path(folder)), f, "harness settings")
        return
    _require(rel[0] in SETS, f, "an unexpected folder")
    states, _, parts, policies = SETS[rel[0]]
    policy, j = rel[1], int(rel[2].removeprefix("part"))
    _require(policy in policies and 0 <= j < len(parts) and st["eval_init_states"] == spec(states)
             and st["eval_seeds"] == parts[j] and str(st["lora_dir"]).endswith(adapter_path(policy)), f,
             f"evaluation settings ({rel[0]}, {policy}, {st.get('lora_dir')})")


def check_provenance(root) -> dict:
    """One commit, clean trees, task 0, n_action_steps 10, 520 steps and one checkpoint in every record; each record has
    the protocol's settings for its folder."""
    files = sorted(f for f in Path(root).rglob("provenance.jsonl") if "round2_data" not in f.parts)
    records = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            st = r["settings"]
            _require(st["task_id"] == TASK, f, f"task {st['task_id']}")
            check_settings(root, f, r)
            flags = [r[k] for k in DIRTY_KEYS if k in r]
            _require(r.get("git_dirty") is False and flags and all(v is False for v in flags), f, "a dirty tree")
            _require(st["n_action_steps"] == 10 and st["max_steps"] == 520, f, "n_action_steps or max_steps")
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


# ---- harness and training data ---------------------------------------------------------------------------------

def committed_rows(run: Path, folder: str, seeds: str) -> dict:
    """A committed run's rows of 0/8 for these seeds (step 10's or step 12's evaluation of the adapter)."""
    out = {}
    for p in sorted(glob.glob(str(run / f"task_{TASK}" / "eval" / folder / "part*" / "per_episode_lora.csv"))):
        out.update(_rows(p))
    want = {(s, CHECK_STATE) for s in seeds_of(seeds)}
    if not want <= set(out):
        raise ValueError(f"{run}: no committed rows for {folder} on {sorted(want - set(out))}")
    return {k: out[k] for k in want}


def harness(root, step10=STEP10, step12=STEP12) -> dict:
    """N0_0 and R0_0 re-run on their own committed episodes of 0/8: how many match exactly (success and steps)."""
    runs = {"N0": step10, "R0": step12}
    per, same, total = {}, 0, 0
    for policy, (seeds, _, folder) in HARNESS.items():
        path = Path(root, "harness", policy, "per_episode_lora.csv")
        new = _rows(path) if path.exists() else {}
        old = committed_rows(runs[policy], folder, seeds)
        if set(new) != set(old):
            raise ValueError(f"harness {policy}: has {sorted(new)}, expected {sorted(old)}; finish the run")
        k = sum(new[e] == old[e] for e in old)
        per[policy] = {"same": k, "episodes": len(old)}
        same, total = same + k, total + len(old)
    return {"per_policy": per, "same": same, "episodes": total, "passes": same >= HARNESS_MIN_SAME}


def gate(root) -> str | None:
    h = harness(root)
    return None if h["passes"] else f"harness: {h['same']} of {h['episodes']} episodes reproduce the committed rows"


def check_training_data(root, step12=STEP12) -> dict:
    """Every L0_s trained on exactly R0_s's training episodes (step 12's committed provenance), nothing else, and the
    three read identical files (the sha256 each run recorded). Returns {run: episodes} and the files' sha256."""
    out, hashes = {}, []
    for s in TRAIN_SEEDS:
        path = Path(root, "runs", f"L{TASK}_{s}", "provenance.jsonl")
        if not path.exists():
            raise ValueError(f"L{TASK}_{s}: no training run; finish the run")
        want = train_episodes(Path(step12, "runs", f"R{TASK}_{s}", "provenance.jsonl"))
        got = train_episodes(path)
        if len(got) != len(set(got)) or set(got) != set(want):
            raise ValueError(f"L{TASK}_{s}: trained on {len(got)} episodes, R{TASK}_{s} on {len(want)} (missing "
                             f"{sorted(set(want) - set(got))[:5]}, extra {sorted(set(got) - set(want))[:5]})")
        out[f"L{TASK}_{s}"] = len(got)
        hashes.append(json.loads(path.read_text().splitlines()[0])["settings"]["train_episode_sha256"])
    if any(h != hashes[0] for h in hashes):
        raise ValueError("L0_0-L0_2 read files with different contents under the same names")
    return {"episodes": out, "sha256": hashes[0]}


# ---- evaluation ------------------------------------------------------------------------------------------------

def load_eval(root, which: str, policy: str) -> dict[tuple[int, int], int]:
    states, seeds, _, _ = SETS[which]
    out = {}
    for path in sorted(glob.glob(str(Path(root, which, policy, "part*", "per_episode_lora.csv")))):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r["seed"]), int(r["init_state"]))
                if int(r["task_id"]) != TASK:
                    raise ValueError(f"{path}: an episode of task {r['task_id']}")
                if key in out:
                    raise ValueError(f"{path}: episode {key} twice")
                out[key] = int(r["success"])
    want = {(s, i) for s in seeds for i in states}
    if set(out) != want:
        raise ValueError(f"{which} {policy}: {len(set(out) & want)} of {len(want)} episodes and "
                         f"{len(set(out) - want)} others; finish the run")
    return out


def per_seed(ev: dict, first: str, second: str) -> dict:
    """``first``<s> against ``second``<s> per training seed on the same episodes; p < 0.05 marks a pass."""
    out = {}
    for s in TRAIN_SEEDS:
        a, b = ev[f"{first}{s}"], ev[f"{second}{s}"]
        v = paired(a, b, sorted(a))
        v["passes"] = v["p"] < 0.05
        out[str(s)] = v
    return out


def reading(harness_ok: bool, a: bool, b: bool) -> str:
    if not harness_ok:
        return "not read: the harness failed"
    if a and b:
        return ("round 2's loss on 0/8 is real, and training long enough to keep round 1's passes per episode reduces "
                "it: dilution is a cause")
    if a:
        return "round 2's loss on 0/8 is real; longer training was not shown to remove it"
    if b:
        return ("round 2's loss on 0/8 was not replicated at this budget; longer training raises 0/8 above R, which "
                "does not show that R lost anything")
    return "round 2's loss on 0/8 was not replicated at this budget, and longer training was not shown to help"


def analyse(root, step10=STEP10, step12=STEP12) -> dict:
    prov = check_provenance(root)
    h = harness(root, step10, step12)
    data = check_training_data(root, step12)
    ev = {w: {p: load_eval(root, w, p) for p in SETS[w][3]} for w in SETS}
    chk, srch = ev["eval_check"], ev["eval_searched"]
    test_a, test_b = per_seed(chk, "N", "R"), per_seed(chk, "L", "R")
    a_pass = sum(v["passes"] for v in test_a.values()) >= MIN_SEEDS
    b_pass = sum(v["passes"] for v in test_b.values()) >= MIN_SEEDS
    states = []
    for w, (sts, seeds, _, policies) in SETS.items():
        for i in sts:
            row = {"state": i, "set": w, "episodes": len(seeds)}
            for p in policies:
                k = sum(ev[w][p][(s, i)] for s in seeds)
                row[p] = [k, list(clopper_pearson(k, len(seeds)))]
            states.append(row)
    pooled_counts = {f"{w}, {x} seeds pooled": sum(sum(ev[w][f"{x}{s}"].values()) for s in TRAIN_SEEDS)
                     for w in SETS for x in sorted({p[0] for p in SETS[w][3]})}
    return {"provenance": prov, "harness": h, "training_sets": data, "states": states,
            "tests": {"test A": {"per_seed": test_a, "passes": a_pass}, "test B": {"per_seed": test_b, "passes": b_pass}},
            "reported": {"L against N on 0/8": per_seed(chk, "L", "N"),
                         "L against R on the searched states": per_seed(srch, "L", "R"),
                         "pooled counts (no pooled p: the seeds share their episodes)": pooled_counts},
            "decision": reading(h["passes"], a_pass, b_pass)}


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"step 14: is round 2's loss on 0/8 real, and is it dilution? commit {r['provenance']['commit']}",
             f"harness: {h['same']} of {h['episodes']} committed episodes reproduce ({'ok' if h['passes'] else 'FAILED'}; "
             + ", ".join(f"{p} {v['same']}/{v['episodes']}" for p, v in h["per_policy"].items()) + ")",
             "training sets: " + ", ".join(f"{k} {v} episodes (as R)" for k, v in r["training_sets"]["episodes"].items())
             + f"; the same {len(r['training_sets']['sha256'])} files (sha256) for all three"]
    for s in r["states"]:
        cells = ", ".join(f"{p} {s[p][0]} ({100 * s[p][1][0]:.1f}-{100 * s[p][1][1]:.1f}%)"
                          for p in s if p[0] in "NRL" and p[1:].isdigit())
        lines.append(f"0/{s['state']} ({s['set']}), of {s['episodes']}: {cells}")
    for name, label in (("test A", ("N", "R")), ("test B", ("L", "R"))):
        for s, v in r["tests"][name]["per_seed"].items():
            lines.append(f"{name}, {label[0]}{s} against {label[1]}{s} on 0/8: {v['first']} against {v['second']} of "
                         f"{v['n']} (only {label[0]} {v['first_only']}, only {label[1]} {v['second_only']}), "
                         f"one-sided p {v['p']:.3g}")
        lines.append(f"{name} {'passes' if r['tests'][name]['passes'] else 'does not pass'} "
                     f"(p < 0.05 in at least {MIN_SEEDS} of {len(TRAIN_SEEDS)} seeds)")
    for k in ("L against N on 0/8", "L against R on the searched states"):
        for s, v in r["reported"][k].items():
            lines.append(f"reported, {k}, seed {s}: {v['first']} against {v['second']} of {v['n']}, two-sided p "
                         f"{v['p_two_sided']:.3g}")
    lines.append("reported, " + ", ".join(f"{k} {v}" for k, v in
                                          r["reported"]["pooled counts (no pooled p: the seeds share their episodes)"].items()))
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", choices=["harness"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
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
