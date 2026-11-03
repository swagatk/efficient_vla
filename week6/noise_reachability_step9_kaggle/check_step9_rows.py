"""Independent recount of step 9's test files; shares no code with analyze_step9.py.

Checks every test file has its arm's noise ids exactly once, on the right task, state and seed (1000 + id), with
provenance from one clean commit and the protocol's settings; that every scene check passed; then recounts the
successes and estimates the permutation p by random relabelling (the analysis computes it exactly).

    python -I check_step9_rows.py <run folder> [--perms 200000 --seed 0]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COMMIT = "f2a067e629557b44996b423c08835647544178b9"
SNAPSHOT = "a24f1f380461b665158d52acd140152c79e8b36d"
HELD_OUT = {0: (14, 27), 4: (18, 26), 6: (14, 20), 7: (38, 47)}
# Typed from README.md, step 9, "Library, per task".
LIBRARY = {
    0: [1, 6, 10, 12, 15, 23, 30, 35, 38, 41, 44, 46, 53, 55, 57, 60, 62, 1055, 1130, 1148],
    4: [23, 27, 49, 1239, 1276, 1286],
    6: [30, 42, 69, 125, 150, 164, 167, 190, 256, 293, 313, 350, 381, 1324, 1334, 1356, 1368, 1389, 1396, 1439],
    7: [1459, 1467, 1473, 1483, 1484, 1492, 1509, 1510, 1518, 1519, 1522, 1536, 1538, 1540, 1545, 1548, 1557, 1561,
        1562, 1567, 1581, 1586, 1595, 1596, 1599],
}
RANDOM = list(range(6000, 6050))


def check_provenance(folder, task, ids):
    recs = [json.loads(l) for l in (folder / "provenance.jsonl").read_text().splitlines() if l.strip()]
    assert recs, folder
    for r in recs:
        assert r["git_commit"] == COMMIT and r["git_dirty"] is False, folder
        s = r["settings"]
        assert s["task_id"] == task and s["family"] == "repeat" and s["n_action_steps"] == 10, folder
        assert s["max_steps"] == 520 and s["jitter_seeds"] is None and SNAPSHOT in s["policy_path"], folder
        assert sorted(int(k) for k in s["noise_ids"].split(",")) == sorted(ids), folder
    return len(recs)


def read_file(path, task, state, ids):
    rows = list(csv.DictReader(path.open()))
    seeds = [int(r["seed"]) for r in rows]
    assert sorted(seeds) == sorted(1000 + k for k in ids), (path, "ids not exactly once")
    for r in rows:
        assert int(r["task_id"]) == task and int(r["init_state"]) == state, path
        assert r["success"] in ("0", "1") and 0 < int(r["steps"]) <= 520, path
        assert r["success"] == "1" or int(r["steps"]) == 520, (path, "failure ended early")
    return {int(r["seed"]) - 1000: int(r["success"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--perms", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    episodes, records, per_task = 0, 0, {}
    for t, states in HELD_OUT.items():
        for s in states:
            scene = json.loads((a.root / f"task_{t}/scene/scene_state{s}.json").read_text())
            assert scene["task_id"] == t and scene["init_state"] == s and scene["same"] is True, (t, s)
        totals = {}
        for arm, ids in (("library", LIBRARY[t]), ("random", RANDOM)):
            folder = a.root / f"task_{t}/test/{arm}"
            records += check_provenance(folder, t, ids)
            assert sorted(p.name for p in folder.glob("*.csv")) == sorted(
                f"per_episode_repeat_state{s}.csv" for s in states), folder
            got = {s: read_file(folder / f"per_episode_repeat_state{s}.csv", t, s, ids) for s in states}
            episodes += sum(len(v) for v in got.values())
            for s in states:
                print(f"task {t} state {s} {arm}: {sum(got[s].values())}/{len(ids)}")
            totals[arm] = np.array([sum(got[s][k] for s in states) for k in ids])
        per_task[t] = totals
    observed = sum(int(v["library"].sum()) for v in per_task.values())
    lib_n = sum(len(v["library"]) * 2 for v in per_task.values())
    ran = sum(int(v["random"].sum()) for v in per_task.values())
    print(f"episodes {episodes}, provenance records {records}")
    print(f"pooled library {observed}/{lib_n}, random {ran}/{len(RANDOM) * 2 * len(HELD_OUT)}")
    null = np.zeros(a.perms, dtype=np.int64)
    for v in per_task.values():
        pool = np.concatenate([v["library"], v["random"]])
        k = len(v["library"])
        keys = rng.random((a.perms, len(pool)))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        null += pool[pick].sum(axis=1)
    p = (null >= observed).mean()
    se = (p * (1 - p) / a.perms) ** 0.5
    print(f"permutation p, {a.perms} random relabellings within task (seed {a.seed}): {p:.4f} +- {1.96 * se:.4f}")


if __name__ == "__main__":
    main()
