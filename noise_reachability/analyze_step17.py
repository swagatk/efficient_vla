#!/usr/bin/env python3
"""Step 17 analysis (README.md, step 17): check the feature run and write the features file. Standard library only.

The test U1 is step17_g1b.py's, run locally on the features file this script writes.

    python3 -S noise_reachability/analyze_step17.py --root <run folder> --gate features   # exit 1 stops the run
    python3 -S noise_reachability/analyze_step17.py --root <run folder> --out step17_report.txt \
        --features step17_features.json

<run folder> holds task_<t>/features/: deep_noise_search.py --family repeat --feature_dir <that folder>/json, one process
per map state, id FEATURE_ID on every state of TASKS x MAP_STATES and also DUPLICATE_ID on DUPLICATE_STATES.

The feature check (the run's harness):

- every map state's FEATURE_ID episode equals step 16's committed row (success and steps) on at least MIN_SAME of the
  120 states; a state that differs is dropped (its features are not written);
- on DUPLICATE_STATES the DUPLICATE_ID episode also equals its committed row, and its features equal FEATURE_ID's bit
  for bit (the prefix pass does not depend on the noise);
- every feature record matches its episode's row, has EXPECTED_DIM finite values, and saw at least one inference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_step16 as a16  # noqa: E402
from analyze_step12 import _rows  # noqa: E402
from analyze_step15 import ids_spec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS, MAP_STATES = a16.TASKS, a16.MAP_STATES
EPISODE_SEED_BASE = a16.EPISODE_SEED_BASE
FEATURE_ID = 10000
DUPLICATE_ID = 10001
DUPLICATE_STATES = ((0, 0), (4, 0), (6, 0), (7, 0))
MIN_SAME = 114                                   # of 120 map states, FEATURE_ID's episode equal to step 16's row
EXPECTED_DIM = 968                               # SmolVLM prefix hidden size 960 + 8 state entries (smoke run, 4090)
STEP16 = REPO_ROOT / "week6" / "noise_reachability_step16_kaggle" / "run"


def ids_of(task: int, state: int) -> tuple[int, ...]:
    return (FEATURE_ID, DUPLICATE_ID) if (task, state) in DUPLICATE_STATES else (FEATURE_ID,)


def out_dir(root, task: int) -> Path:
    return Path(root, f"task_{task}", "features")


def feature_path(root, task: int, state: int, noise_id: int) -> Path:
    return out_dir(root, task) / "json" / f"features_state{state}_id{noise_id}.json"


# ---- provenance ------------------------------------------------------------------------------------------------

def check_settings(root, f: Path, r: dict) -> None:
    """Step 17's folders: task_<t>/features only, a plain repeat search of the base with the feature recorder on."""
    st = r["settings"]
    rel = f.parent.relative_to(Path(root)).parts
    tasks = {f"task_{t}": t for t in TASKS}
    a16._require(len(rel) == 2 and rel[0] in tasks and rel[1] == "features", f, f"an unexpected folder {f.parent}")
    t = tasks[rel[0]]
    a16._search_settings(f, st, t, MAP_STATES)
    a16._require(st["noise_ids"] == ids_spec(ids_of(t, st["init_state"])), f, f"ids {st['noise_ids']}")
    a16._require(str(st.get("feature_dir") or "").rstrip("/").endswith(f"task_{t}/features/json"), f,
                 f"feature_dir {st.get('feature_dir')}")


def check_provenance(root) -> dict:
    return a16.check_provenance(root, settings_check=check_settings)


# ---- the feature check -----------------------------------------------------------------------------------------

def committed_rows(step16=STEP16) -> dict[tuple[int, int, int], tuple[int, int]]:
    """{(task, state, id): (success, steps)} of step 16's committed part-1 episodes for the ids step 17 re-runs."""
    out = {}
    for t in TASKS:
        for s in MAP_STATES:
            rows = a16.part_rows(step16, t, 0, s)
            for k in ids_of(t, s):
                out[(t, s, k)] = rows[k]
    return out


def run_rows(root, task: int, state: int) -> dict[int, tuple[int, int]]:
    path = out_dir(root, task) / f"per_episode_repeat_state{state}.csv"
    rows = _rows(path) if path.exists() else {}
    if any(st != state for _, st in rows):
        raise ValueError(f"{path}: rows of another state")
    got = {sd - EPISODE_SEED_BASE: v for (sd, _), v in rows.items()}
    if set(got) != set(ids_of(task, state)):
        raise ValueError(f"features {task}/{state}: ids {sorted(got)}, expected {list(ids_of(task, state))}; finish the run")
    return got


def read_record(root, task: int, state: int, noise_id: int, row: tuple[int, int]) -> tuple[dict, list[float]]:
    path = feature_path(root, task, state, noise_id)
    if not path.exists():
        raise ValueError(f"{path}: missing, but its episode has a row")
    rec = json.loads(path.read_text())
    feats = rec.pop("features")
    want = {"task_id": task, "init_state": state, "noise_id": noise_id, "seed": EPISODE_SEED_BASE + noise_id,
            "success": row[0], "steps": row[1]}
    if any(rec.get(k) != v for k, v in want.items()) or rec.get("dtype") != "float32":
        raise ValueError(f"{path}: {rec} does not match its episode {want}")
    if len(feats) != rec["dim"] or rec["dim"] != EXPECTED_DIM or rec["inferences"] < 1:
        raise ValueError(f"{path}: {len(feats)} values for dim {rec['dim']} (expected {EXPECTED_DIM}), "
                         f"{rec['inferences']} inferences")
    if not all(isinstance(x, float) and math.isfinite(x) for x in feats):
        raise ValueError(f"{path}: a feature value that is not a finite float")
    return rec, feats


def feature_check(root, step16=STEP16) -> dict:
    old = committed_rows(step16)
    same, differ, dup_bad, feats, dims = [], [], [], {}, set()
    for t in TASKS:
        for s in MAP_STATES:
            new = run_rows(root, t, s)
            recs = {k: read_record(root, t, s, k, new[k]) for k in ids_of(t, s)}
            dims |= {r["dim"] for r, _ in recs.values()}
            (same if new[FEATURE_ID] == old[(t, s, FEATURE_ID)] else differ).append((t, s))
            feats[(t, s)] = recs[FEATURE_ID][1]
            if DUPLICATE_ID in recs and (new[DUPLICATE_ID] != old[(t, s, DUPLICATE_ID)]
                                         or recs[DUPLICATE_ID][1] != recs[FEATURE_ID][1]):
                dup_bad.append((t, s))
    if len(dims) != 1:
        raise ValueError(f"feature dimensions differ between records: {sorted(dims)}")
    n = len(TASKS) * len(MAP_STATES)
    return {"same": len(same), "states": n, "differ": [list(x) for x in differ], "duplicates_bad": [list(x) for x in dup_bad],
            "dim": dims.pop(), "passes": len(same) >= MIN_SAME and not dup_bad,
            "kept": [list(x) for x in same], "_features": {x: feats[x] for x in same}}


def gate(root) -> str | None:
    check_provenance(root)
    c = feature_check(root)
    if c["passes"]:
        return None
    return (f"feature check: {c['same']} of {c['states']} states reproduce step 16 (need {MIN_SAME}); "
            f"duplicate states failing: {c['duplicates_bad']}")


def analyse(root, step16=STEP16) -> tuple[dict, dict | None]:
    """(report, features file content); the features are None when the feature check failed."""
    prov = check_provenance(root)
    c = feature_check(root, step16)
    feats = c.pop("_features")
    out = {"provenance": prov, "feature_check": c}
    if not c["passes"]:
        out["decision"] = "not read: the feature check failed"
        return out, None
    out["decision"] = "feature check passes: run step17_g1b.py on the features file"
    rows = [{"task_id": t, "init_state": s, "features": f} for (t, s), f in sorted(feats.items())]
    return out, {"commit": prov["commit"], "dim": c["dim"], "rows": rows}


def format_report(r: dict) -> str:
    c = r["feature_check"]
    lines = [f"step 17: first-inference features, commit {r['provenance']['commit']}, {r['provenance']['records']} "
             f"provenance records",
             f"feature check: {c['same']} of {c['states']} states reproduce step 16's id {FEATURE_ID} row (need {MIN_SAME}); "
             f"duplicate id {DUPLICATE_ID} on {len(DUPLICATE_STATES)} states: "
             + ("all identical" if not c["duplicates_bad"] else f"FAILED on {c['duplicates_bad']}")
             + f"; feature dimension {c['dim']} ({'ok' if c['passes'] else 'FAILED'})"]
    if c["differ"]:
        lines.append(f"states that differ (dropped): {c['differ']}")
    lines.append(f"decision: {r['decision']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", choices=["features"])
    ap.add_argument("--out")
    ap.add_argument("--features")
    a = ap.parse_args(argv)
    if a.gate:
        g = gate(a.root)
        print(f"gate {a.gate} " + ("passed" if g is None else f"FAILED: {g}"))
        return 0 if g is None else 1
    r, feats = analyse(a.root)
    text = format_report(r)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(r, indent=1) + "\n")
    if a.features and feats is not None:
        Path(a.features).write_text(json.dumps(feats) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
