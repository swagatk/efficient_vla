#!/usr/bin/env python3
"""Step 15 analysis (README.md, step 15): are 6/14 and 6/20 unreachable, and did distillation lose 7/38's
constant-noise solutions? Standard library only.

    python3 -S noise_reachability/analyze_step15.py --root <run folder> --out step15_report.txt
    python3 -S noise_reachability/analyze_step15.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step15.py --root <run folder> --replay_jobs    # task state ids, per replay

<run folder> holds
  task_<t>/scene/scene_state<i>.json         check_scene.py, for 6/14, 6/20 and 7/38;
  harness/base/                              the base on 10 of step 9's 7/38 episodes (deep_noise_search, repeat);
  harness/N0/                                step 10's N7_0 on its own 7/33 episodes, seeds 16-19 (self_filtered_bc eval);
  task_6/search/part<j>/                     test A: the base, `repeat` ids A_PARTS[j], on 6/14 and 6/20;
  task_6/replay/                             test A: every search success re-run once with its id;
  task_7/search/<base|N0|N1|N2>/             test B: the base and N7_0-N7_2, `repeat` ids B_IDS, on 7/38.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import read_rows  # noqa: E402
from analyze_step7 import POLICY_REVISION, clopper_pearson  # noqa: E402
from analyze_step8 import scene_of  # noqa: E402
from analyze_step10 import DIRTY_KEYS  # noqa: E402
from analyze_step11 import STEP9, STEP9_SHA256  # noqa: E402
from analyze_step12 import STEP10, _rows, paired  # noqa: E402
from analyze_step13 import ADAPTER_SHA256  # noqa: E402

EPISODE_SEED_BASE = 1000                         # deep_noise_search: noise id k runs with episode seed 1000 + k
TASK_A, STATES_A = 6, (14, 20)                   # never solved by any noise in the committed runs
TASK_B, STATE_B = 7, 38                          # solved by the base's constant noises (step 9), not by N7_0's (step 12)
A_IDS = tuple(range(9000, 9400))                 # fresh: committed repeat ids are 0-69, 100-399, 1000-1599, 6000-6049, 7000-7049
A_PARTS = ("9000-9199", "9200-9399")             # two jobs per state, so the four GPU slots stay busy
B_IDS = tuple(range(9000, 9200))
TRAIN_SEEDS = (0, 1, 2)
B_POLICIES = ["base"] + [f"N{s}" for s in TRAIN_SEEDS]
MIN_SEEDS = 2
ALPHA = 0.05
HARNESS_BASE_IDS = (6000, 6001, 6002, 6017, 6019, 6021, 6023, 6025, 6030, 6040)   # step 9's seven 7/38 successes + 3
HARNESS_N_STATE, HARNESS_N_SEEDS = 33, "16-19"   # step 10's N7_0 episodes, re-run as step 12's harness
HARNESS_MIN_SAME = 13                            # of 14 episodes identical to the committed rows in success and steps
STEP9_FILE = f"task_{TASK_B}/test/random/per_episode_repeat_state{STATE_B}.csv"
N7_SHA256 = {f"N{s}": ADAPTER_SHA256[f"N{TASK_B}_{s}"] for s in TRAIN_SEEDS}   # step 10's archive manifest
SCENE_STATES = {TASK_A: STATES_A, TASK_B: (STATE_B,)}


def ids_spec(ids) -> str:
    """'9000-9399' for a run of consecutive ids, else comma-separated runs ('6000-6002,6017,...')."""
    ids, runs, i = sorted(ids), [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        runs.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(runs)


def parse_ids(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def adapter_suffix(policy: str) -> str:
    return f"step10/runs/N{TASK_B}_{policy[1:]}/adapter"


# ---- provenance ------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def _search_settings(f, st, task, states, lora) -> None:
    _require(st.get("family") == "repeat" and st.get("jitter_seeds") is None and st.get("record_dir") is None, f,
             "not a plain repeat search")
    _require(st["task_id"] == task and st["init_state"] in states, f, f"task {st['task_id']} state {st['init_state']}")
    if lora is None:
        _require(st.get("lora_dir") is None, f, "an adapter on the base")
    else:
        _require(str(st.get("lora_dir")).rstrip("/").endswith(lora), f, f"adapter {st.get('lora_dir')}")


def check_settings(root, f: Path, r: dict) -> None:
    """The protocol's settings for the folder ``f`` is in; refuses a folder the protocol does not have."""
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    if len(rel) == 2 and rel[1] == "scene" and rel[0] in {f"task_{t}" for t in SCENE_STATES}:
        t = int(rel[0].removeprefix("task_"))
        _require(st["task_id"] == t and st["init_state"] in SCENE_STATES[t] and "family" not in st, f, "scene check")
        return
    if rel == ("harness", "base"):
        _search_settings(f, st, TASK_B, (STATE_B,), None)
        _require(st["noise_ids"] == ids_spec(HARNESS_BASE_IDS), f, f"harness ids {st['noise_ids']}")
        return
    if rel == ("harness", "N0"):
        _require(st.get("stage") == "eval" and st["eval_policies"] == "lora" and st["task_id"] == TASK_B
                 and st["eval_init_states"] == str(HARNESS_N_STATE) and st["eval_seeds"] == HARNESS_N_SEEDS
                 and str(st["lora_dir"]).rstrip("/").endswith(adapter_suffix("N0")), f, "harness settings")
        return
    if len(rel) == 3 and rel[:2] == (f"task_{TASK_A}", "search") and rel[2] in {f"part{j}" for j in range(len(A_PARTS))}:
        _search_settings(f, st, TASK_A, STATES_A, None)
        _require(st["noise_ids"] == A_PARTS[int(rel[2].removeprefix("part"))], f, f"ids {st['noise_ids']}")
        return
    if rel == (f"task_{TASK_A}", "replay"):
        _search_settings(f, st, TASK_A, STATES_A, None)
        _require(set(parse_ids(st["noise_ids"])) <= set(A_IDS), f, f"replay ids {st['noise_ids']}")
        return
    if len(rel) == 3 and rel[:2] == (f"task_{TASK_B}", "search") and rel[2] in B_POLICIES:
        _search_settings(f, st, TASK_B, (STATE_B,), None if rel[2] == "base" else adapter_suffix(rel[2]))
        _require(st["noise_ids"] == ids_spec(B_IDS), f, f"ids {st['noise_ids']}")
        return
    raise ValueError(f"provenance: an unexpected folder {f.parent}")


def check_provenance(root) -> dict:
    """One commit, clean trees, T4 GPUs only, n_action_steps 10, 520 steps, one checkpoint and one package set in every
    record; each record has the protocol's settings for its folder."""
    records = []
    for f in sorted(Path(root).rglob("provenance.jsonl")):
        for line in open(f):
            r = json.loads(line)
            st = r["settings"]
            check_settings(root, f, r)
            flags = [r[k] for k in DIRTY_KEYS if k in r]
            _require(r.get("git_dirty") is False and flags and all(v is False for v in flags), f, "a dirty tree")
            _require(st["n_action_steps"] == 10 and st["max_steps"] == 520, f, "n_action_steps or max_steps")
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

def committed_harness(step9=STEP9, step10=STEP10) -> dict:
    """{policy: {(seed, state): (success, steps)}} of the committed episodes the harness re-runs. Step 9's file must be
    the committed one (sha256) when the committed folder is read."""
    path = Path(step9) / STEP9_FILE
    if Path(step9).resolve() == STEP9.resolve():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != STEP9_SHA256[STEP9_FILE]:
            raise ValueError(f"{path}: sha256 {got} is not the committed file's")
    rows = _rows(path)
    base = {(EPISODE_SEED_BASE + k, STATE_B): rows[(EPISODE_SEED_BASE + k, STATE_B)] for k in HARNESS_BASE_IDS}
    old = {}
    for p in sorted((Path(step10) / f"task_{TASK_B}" / "eval" / "N0").glob("part*/per_episode_lora.csv")):
        old.update(_rows(p))
    lo, hi = (int(x) for x in HARNESS_N_SEEDS.split("-"))
    want = {(s, HARNESS_N_STATE) for s in range(lo, hi + 1)}
    if not want <= set(old):
        raise ValueError(f"{step10}: no committed N7_0 rows for {sorted(want - set(old))}")
    return {"base": base, "N0": {k: old[k] for k in want}}


def harness(root, step9=STEP9, step10=STEP10) -> dict:
    """How many of the 14 committed episodes this run reproduces exactly (success and steps)."""
    files = {"base": Path(root, "harness", "base", f"per_episode_repeat_state{STATE_B}.csv"),
             "N0": Path(root, "harness", "N0", "per_episode_lora.csv")}
    per, same, total = {}, 0, 0
    for policy, old in committed_harness(step9, step10).items():
        new = _rows(files[policy]) if files[policy].exists() else {}
        if set(new) != set(old):
            raise ValueError(f"harness {policy}: has {sorted(new)}, expected {sorted(old)}; finish the run")
        k = sum(new[e] == old[e] for e in old)
        per[policy] = {"same": k, "episodes": len(old), "differ": sorted(str(e) for e in old if new[e] != old[e])}
        same, total = same + k, total + len(old)
    return {"per_policy": per, "same": same, "episodes": total, "passes": same >= HARNESS_MIN_SAME}


def gate(root) -> str | None:
    check_provenance(root)
    h = harness(root)
    return None if h["passes"] else f"harness: {h['same']} of {h['episodes']} episodes reproduce the committed rows"


# ---- scenes, search and replay ---------------------------------------------------------------------------------

def scenes(root) -> dict[tuple[int, int], bool]:
    """{(task, state): passed} for the three states; refuses a state not yet checked."""
    out = {}
    for t, states in SCENE_STATES.items():
        for s in states:
            same = scene_of(root, t, s)
            if same is None:
                raise ValueError(f"scene check of {t}/{s} is missing; finish the run")
            out[(t, s)] = same
    return out


def _by_id(path: Path, ids, what: str) -> dict[int, int]:
    rows = read_rows(path) if path.exists() else {}
    got = {s - EPISODE_SEED_BASE: v for s, v in rows.items()}
    if set(got) != set(ids):
        raise ValueError(f"{what}: {len(set(got) & set(ids))} of {len(ids)} episodes and {len(set(got) - set(ids))} "
                         "others; finish the run")
    return got


def _steps(path: Path, wins) -> dict[int, int]:
    """{id: steps} of the successful ids in one result file (read after _by_id checked the file)."""
    return {s - EPISODE_SEED_BASE: v[1] for (s, _), v in _rows(path).items() if s - EPISODE_SEED_BASE in wins}


def a_file(root, j: int, state: int) -> Path:
    return Path(root, f"task_{TASK_A}", "search", f"part{j}", f"per_episode_repeat_state{state}.csv")


def search_a(root, state: int) -> dict[int, int]:
    """{id: success} of the base on one test-A state, merged over the parts (each part must hold exactly its ids)."""
    out = {}
    for j, part in enumerate(A_PARTS):
        rows = _by_id(a_file(root, j, state), parse_ids(part), f"test A {TASK_A}/{state} part {j}")
        out.update(rows)
    return out


def replay_ids(root, state: int) -> list[int]:
    return sorted(k for k, v in search_a(root, state).items() if v)


def replay_jobs(root) -> list[tuple[int, int, str]]:
    """(task, state, ids) of the successes to replay, for the states that passed the scene check."""
    ok = scenes(root)
    return [(TASK_A, s, ids_spec(ids)) for s in STATES_A if ok[(TASK_A, s)] and (ids := replay_ids(root, s))]


def test_a(root, state: int) -> dict:
    found = search_a(root, state)
    wins = sorted(k for k, v in found.items() if v)
    path = Path(root, f"task_{TASK_A}", "replay", f"per_episode_repeat_state{state}.csv")
    replay = _by_id(path, wins, f"replay of {TASK_A}/{state}") if wins else {}
    replayed = [k for k in wins if replay[k]]
    n = len(A_IDS)
    steps = {}
    for j in range(len(A_PARTS)):
        steps.update(_steps(a_file(root, j, state), set(wins)))
    replay_steps = _steps(path, set(wins)) if wins else {}
    return {"state": state, "episodes": n, "successes": len(wins), "success_ids": wins, "replayed_ids": replayed,
            "success_steps": {str(k): [steps[k], replay_steps.get(k)] for k in wins},
            "found_interval": list(clopper_pearson(len(wins), n)),
            "interval": list(clopper_pearson(len(replayed), n)), "passes": bool(replayed),
            "upper_bound_if_none": 1 - ALPHA ** (1 / n)}


def test_b(root) -> dict:
    files = {p: Path(root, f"task_{TASK_B}", "search", p, f"per_episode_repeat_state{STATE_B}.csv") for p in B_POLICIES}
    ev = {p: _by_id(files[p], B_IDS, f"test B {p}") for p in B_POLICIES}
    per_seed = {}
    for s in TRAIN_SEEDS:
        v = paired(ev["base"], ev[f"N{s}"], B_IDS)
        v["passes"] = v["p"] < ALPHA
        per_seed[str(s)] = v
    counts = {p: [sum(ev[p].values()), list(clopper_pearson(sum(ev[p].values()), len(B_IDS)))] for p in B_POLICIES}
    steps = {p: sorted(_steps(files[p], {k for k, v in ev[p].items() if v}).values()) for p in B_POLICIES}
    return {"per_seed": per_seed, "counts": counts, "success_steps": steps, "episodes": len(B_IDS),
            "passes": sum(v["passes"] for v in per_seed.values()) >= MIN_SEEDS}


def reading_a(r: dict | None) -> str:
    if r is None:
        return "not read: the state failed the scene check"
    if r["passes"]:
        return "rare, not unreachable: a constant noise solves it"
    return (f"not solved: below {100 * r['upper_bound_if_none']:.2f}% per noise (one-sided 95%), a limit the paper "
            "states")


def reading_b(r: dict | None) -> str:
    if r is None:
        return "not read: 7/38 failed the scene check"
    if r["passes"]:
        return ("distillation on 7/33 removed constant-noise solutions of 7/38 that the base had: later rounds should "
                "also search with the base")
    return "step 12's 0 of 50 is not shown to be more than chance at this budget"


def analyse(root, step9=STEP9, step10=STEP10) -> dict:
    prov = check_provenance(root)
    h = harness(root, step9, step10)
    out = {"provenance": prov, "harness": h}
    if not h["passes"]:
        out["decision"] = "not read: the harness failed"
        return out
    ok = scenes(root)
    out["scene_failed"] = [list(k) for k, v in ok.items() if not v]
    out["test A"] = {str(s): (test_a(root, s) if ok[(TASK_A, s)] else None) for s in STATES_A}
    out["test B"] = test_b(root) if ok[(TASK_B, STATE_B)] else None
    out["decision"] = {**{f"{TASK_A}/{s}": reading_a(out["test A"][str(s)]) for s in STATES_A},
                       f"{TASK_B}/{STATE_B}": reading_b(out["test B"])}
    both = [out["test A"][str(s)] for s in STATES_A]
    if all(r is not None and r["passes"] for r in both):
        out["decision"]["both task-6 states"] = "every resistant state searched so far has been solved by a constant noise"
    return out


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"step 15: are 6/14 and 6/20 unreachable, and did distillation lose 7/38's solutions? "
             f"commit {r['provenance']['commit']}",
             f"harness: {h['same']} of {h['episodes']} committed episodes reproduce ({'ok' if h['passes'] else 'FAILED'}; "
             + ", ".join(f"{p} {v['same']}/{v['episodes']}" for p, v in h["per_policy"].items()) + ")"]
    if not h["passes"]:
        lines.append(f"decision: {r['decision']}")
        return "\n".join(lines) + "\n"
    if r["scene_failed"]:
        lines.append(f"states that failed the scene check (dropped): {r['scene_failed']}")
    for s, v in r["test A"].items():
        if v is None:
            continue
        lo, hi = v["interval"]
        lines.append(f"test A, {TASK_A}/{s}: base {v['successes']} of {v['episodes']} fresh repeat noises succeeded "
                     f"(ids {v['success_ids']}), {len(v['replayed_ids'])} replayed ({100 * lo:.2f}-{100 * hi:.2f}%); "
                     f"{'passes' if v['passes'] else 'does not pass'}")
        if v["success_steps"]:
            lines.append(f"test A, {TASK_A}/{s}: steps of the successes (search, replay): {v['success_steps']}")
    b = r["test B"]
    if b is not None:
        lines.append(f"test B, {TASK_B}/{STATE_B}, of {b['episodes']} paired noises: " + ", ".join(
            f"{p} {k} ({100 * iv[0]:.1f}-{100 * iv[1]:.1f}%)" for p, (k, iv) in b["counts"].items()))
        lines.append(f"test B, steps of the successes: {b['success_steps']}")
        for s, v in b["per_seed"].items():
            lines.append(f"test B, base against N{TASK_B}_{s}: {v['first']} against {v['second']} (only base "
                         f"{v['first_only']}, only N {v['second_only']}), one-sided p {v['p']:.3g}, two-sided "
                         f"{v['p_two_sided']:.3g}")
        lines.append(f"test B {'passes' if b['passes'] else 'does not pass'} (p < {ALPHA} in at least {MIN_SEEDS} of "
                     f"{len(TRAIN_SEEDS)} seeds)")
    for k, v in r["decision"].items():
        lines.append(f"decision, {k}: {v}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", choices=["harness"])
    ap.add_argument("--replay_jobs", action="store_true", help="print task, state, ids of the successes to replay")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.gate:
        g = gate(a.root)
        print(f"gate {a.gate} " + ("passed" if g is None else f"FAILED: {g}"))
        return 0 if g is None else 1
    if a.replay_jobs:
        for t, s, ids in replay_jobs(a.root):
            print(t, s, ids)
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
