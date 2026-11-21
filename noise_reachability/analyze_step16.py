#!/usr/bin/env python3
"""Step 16 analysis (README.md, step 16): the crossed constant-noise map. Standard library only.

This script checks the run and writes its table. The tests T1 and T2 are step16_g1.py's, run locally on that table.

    python3 -S noise_reachability/analyze_step16.py --root <run folder> --gate harness   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step16.py --root <run folder> --parts 1 --out step16_report.txt
    python3 -S noise_reachability/analyze_step16.py --root <run folder> --parts 4 --table step16_table.csv

<run folder> holds
  task_<t>/scene/scene_state<i>.json         check_scene.py, for every map state;
  harness/base/                              the base on 12 of step 15's 7/38 episodes (deep_noise_search, repeat);
  task_<t>/map/part<j>/                      the base, `repeat` ids PARTS[j], on every map state that passed its scene check.

``--parts n`` reads parts 0..n-1 (all must be complete) and reports part 1's gate when n is 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_step5a import read_rows  # noqa: E402
from analyze_step7 import POLICY_REVISION  # noqa: E402
from analyze_step8 import scene_of  # noqa: E402
from analyze_step10 import DIRTY_KEYS  # noqa: E402
from analyze_step12 import _rows  # noqa: E402
from analyze_step15 import ids_spec, parse_ids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODE_SEED_BASE = 1000                         # deep_noise_search: noise id k runs with episode seed 1000 + k
TASKS = (0, 4, 6, 7)                             # the weak tasks with a fixed scene (task 8's stove moves)
MAP_STATES = tuple(range(30))                    # the search split; states 30-49 are held out for later steps
HELD_OUT_STATES = {t: tuple(s for s in range(30, 50) if not (t == 7 and s in (33, 38, 47))) for t in TASKS}
PARTS = ("10000-10009", "10010-10019", "10020-10029", "10030-10039", "10040-10049")   # fresh: committed repeat ids are 0-69,
#                                     100-399, 1000-1599, 6000-6049, 7000-7049, 9000-9399
MIN_INFORMATIVE_FRACTION = 0.10                  # part 1's gate: states with both a success and a failure
IDS_PER_PROCESS = 10                             # deep_noise_search's default: one process per (state, part)
STEP15 = REPO_ROOT / "week6" / "noise_reachability_step15_kaggle" / "run"
STEP15_FILE = "task_7/search/base/per_episode_repeat_state38.csv"
STEP15_SHA256 = "472d8f0b81377e78ff98dcfeb8ac516bbde6e399f084436b3ac8aab72b4949d7"
HARNESS_TASK, HARNESS_STATE = 7, 38
HARNESS_IDS = (9000, 9001, 9002, 9003, 9004, 9005, 9007, 9008, 9012, 9019, 9022, 9045)   # 6 failures, 6 successes
HARNESS_MIN_SAME = 11                            # of 12 episodes identical to the committed rows in success and steps


def map_dir(root, task: int, part: int) -> Path:
    return Path(root, f"task_{task}", "map", f"part{part}")


def map_file(root, task: int, part: int, state: int) -> Path:
    return map_dir(root, task, part) / f"per_episode_repeat_state{state}.csv"


# ---- provenance ------------------------------------------------------------------------------------------------

def _require(cond: bool, f, what: str) -> None:
    if not cond:
        raise ValueError(f"provenance: {what} in {f}")


def _search_settings(f, st, task, states) -> None:
    _require(st.get("family") == "repeat" and st.get("jitter_seeds") is None and st.get("record_dir") is None
             and st.get("lora_dir") is None, f, "not a plain repeat search of the base")
    _require(st["task_id"] == task and st["init_state"] in states, f, f"task {st['task_id']} state {st['init_state']}")


def check_settings(root, f: Path, r: dict) -> None:
    """The protocol's settings for the folder ``f`` is in; refuses a folder the protocol does not have."""
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    tasks = {f"task_{t}": t for t in TASKS}
    if len(rel) == 2 and rel[0] in tasks and rel[1] == "scene":
        _require(st["task_id"] == tasks[rel[0]] and st["init_state"] in MAP_STATES and "family" not in st, f,
                 "scene check")
        return
    if rel == ("harness", "base"):
        _search_settings(f, st, HARNESS_TASK, (HARNESS_STATE,))
        _require(st["noise_ids"] == ids_spec(HARNESS_IDS), f, f"harness ids {st['noise_ids']}")
        return
    if len(rel) == 3 and rel[0] in tasks and rel[1] == "map" and rel[2] in {f"part{j}" for j in range(len(PARTS))}:
        _search_settings(f, st, tasks[rel[0]], MAP_STATES)
        _require(st["noise_ids"] == PARTS[int(rel[2].removeprefix("part"))], f, f"ids {st['noise_ids']}")
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

def committed_harness(step15=STEP15) -> dict:
    """{(seed, state): (success, steps)} of the committed step-15 episodes the harness re-runs (sha256-checked when the
    committed folder is read)."""
    path = Path(step15) / STEP15_FILE
    if Path(step15).resolve() == STEP15.resolve():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != STEP15_SHA256:
            raise ValueError(f"{path}: sha256 {got} is not the committed file's")
    rows = _rows(path)
    want = [(EPISODE_SEED_BASE + k, HARNESS_STATE) for k in HARNESS_IDS]
    missing = [e for e in want if e not in rows]
    if missing:
        raise ValueError(f"{path}: no committed rows for {missing}")
    return {e: rows[e] for e in want}


def harness(root, step15=STEP15) -> dict:
    """How many of the 12 committed episodes this run reproduces exactly (success and steps)."""
    old = committed_harness(step15)
    path = Path(root, "harness", "base", f"per_episode_repeat_state{HARNESS_STATE}.csv")
    new = _rows(path) if path.exists() else {}
    if set(new) != set(old):
        raise ValueError(f"harness: has {sorted(new)}, expected {sorted(old)}; finish the run")
    same = sum(new[e] == old[e] for e in old)
    return {"same": same, "episodes": len(old), "differ": sorted(str(e) for e in old if new[e] != old[e]),
            "passes": same >= HARNESS_MIN_SAME}


def gate(root) -> str | None:
    check_provenance(root)
    h = harness(root)
    return None if h["passes"] else f"harness: {h['same']} of {h['episodes']} episodes reproduce the committed rows"


# ---- scenes and the map ----------------------------------------------------------------------------------------

def scenes(root) -> dict[tuple[int, int], bool]:
    """{(task, state): passed} for every map state; refuses a state not yet checked."""
    out = {}
    for t in TASKS:
        for s in MAP_STATES:
            same = scene_of(root, t, s)
            if same is None:
                raise ValueError(f"scene check of {t}/{s} is missing; finish the run")
            out[(t, s)] = same
    return out


def part_rows(root, task: int, part: int, state: int) -> dict[int, tuple[int, int]]:
    """{noise id: (success, steps)} of one state in one part; refuses a file without exactly that part's ids."""
    path = map_file(root, task, part, state)
    ids = parse_ids(PARTS[part])
    rows = _rows(path) if path.exists() else {}
    read_rows(path) if path.exists() else None          # refuses a success other than 0/1
    if any(st != state for _, st in rows):
        raise ValueError(f"{path}: rows of another state")
    got = {s - EPISODE_SEED_BASE: v for (s, _), v in rows.items()}
    if set(got) != set(ids):
        raise ValueError(f"map {task}/{state} part {part}: {len(set(got) & set(ids))} of {len(ids)} episodes and "
                         f"{len(set(got) - set(ids))} others; finish the run")
    return got


def map_table(root, n_parts: int) -> dict[tuple[int, int, int], tuple[int, int]]:
    """{(task, state, id): (success, steps)} over parts 0..n_parts-1 of every state that passed its scene check."""
    if not 1 <= n_parts <= len(PARTS):
        raise ValueError(f"--parts must be 1-{len(PARTS)}")
    ok = scenes(root)
    out = {}
    for (t, s), passed in ok.items():
        if not passed:
            continue
        for j in range(n_parts):
            for k, v in part_rows(root, t, j, s).items():
                out[(t, s, k)] = v
    return out


def informative(table: dict) -> dict[tuple[int, int], tuple[int, int]]:
    """{(task, state): (successes, episodes)} with both a success and a failure."""
    per: dict = {}
    for (t, s, _), (y, _) in table.items():
        n, k = per.get((t, s), (0, 0))
        per[(t, s)] = (n + y, k + 1)
    return {ts: v for ts, v in per.items() if 0 < v[0] < v[1]}


def part1_gate(table: dict, n_states: int = len(TASKS) * len(MAP_STATES)) -> dict:
    """Parts 2-5 run only if at least MIN_INFORMATIVE_FRACTION of all map states (dropped ones count as not
    informative) have both a success and a failure."""
    inf = informative(table)
    need = MIN_INFORMATIVE_FRACTION * n_states
    return {"informative_states": len(inf), "map_states": n_states, "needed": need, "passes": len(inf) >= need}


def write_table(table: dict, out) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "init_state", "noise_id", "success", "steps"])
        for (t, s, k), (y, steps) in sorted(table.items()):
            w.writerow([t, s, k, y, steps])


def analyse(root, n_parts: int, step15=STEP15) -> tuple[dict, dict | None]:
    """(report, table); the table is None when the harness failed."""
    prov = check_provenance(root)
    h = harness(root, step15)
    out = {"provenance": prov, "harness": h, "parts": n_parts}
    if not h["passes"]:
        out["decision"] = "not read: the harness failed"
        return out, None
    ok = scenes(root)
    out["scene_failed"] = [list(k) for k, v in ok.items() if not v]
    table = map_table(root, n_parts)
    out["episodes"] = len(table)
    out["successes"] = sum(y for y, _ in table.values())
    out["per_task"] = {str(t): [sum(y for (tt, _, _), (y, _) in table.items() if tt == t),
                                sum(1 for (tt, _, _) in table if tt == t)] for t in TASKS}
    if n_parts == 1:
        g = part1_gate(table)
        out["part1_gate"] = g
        out["decision"] = ("part 1's gate passes: run parts 2-5" if g["passes"] else
                           "part 1's gate fails: too few states with both outcomes to learn from; stop and rethink")
    else:
        out["decision"] = f"parts 1-{n_parts} complete: run step16_g1.py on the table" + (
            "" if n_parts == len(PARTS) else " only after every part is done")
    return out, table


def format_report(r: dict) -> str:
    h = r["harness"]
    lines = [f"step 16: the crossed constant-noise map, parts 1-{r['parts']}, commit {r['provenance']['commit']}",
             f"harness: {h['same']} of {h['episodes']} committed episodes reproduce ({'ok' if h['passes'] else 'FAILED'})"]
    if not h["passes"]:
        lines.append(f"decision: {r['decision']}")
        return "\n".join(lines) + "\n"
    if r["scene_failed"]:
        lines.append(f"states that failed the scene check (dropped): {r['scene_failed']}")
    lines.append(f"map: {r['episodes']} episodes, {r['successes']} successes; per task (successes, episodes): "
                 f"{r['per_task']}")
    if "part1_gate" in r:
        g = r["part1_gate"]
        lines.append(f"part 1 gate: {g['informative_states']} of {g['map_states']} states have both outcomes, "
                     f"{g['needed']:.0f} needed ({'passes' if g['passes'] else 'fails'})")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", choices=["harness"])
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--table")
    a = ap.parse_args(argv)
    if a.gate:
        g = gate(a.root)
        print(f"gate {a.gate} " + ("passed" if g is None else f"FAILED: {g}"))
        return 0 if g is None else 1
    r, table = analyse(a.root, a.parts)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    if a.table and table is not None:
        write_table(table, a.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
