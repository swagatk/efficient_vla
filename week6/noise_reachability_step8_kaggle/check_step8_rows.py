"""Separate recount of step 8 from the raw CSVs, sharing no code with analyze_step8.py.
    python -I -S check_step8_rows.py <run root>
Checks every episode of the harness, control and tests is there exactly once, then recounts hits, the two
one-sided Fisher tests, Holm at .05 and the jittered confirmations."""
import csv, glob, json, os, re, sys
from math import comb
root = sys.argv[1]
TASKS = (0, 4, 6, 7)
CONTROL = {0: 9, 4: 5, 6: 0, 7: 4}
REF = {0: 7, 4: 14, 6: 20, 7: 18}
TEST2 = [(0, 11), (4, 11), (6, 12), (7, 33)]  # selection order printed by --tests in both sessions
def rows(p): return list(csv.DictReader(open(p)))
def win(x): return str(x["success"]).strip().lower() in ("1", "true", "1.0")
def fisher(a, b, n):
    """P(repeat >= a | a + b hits over two arms of n), one-sided."""
    t = a + b
    return sum(comb(n, k) * comb(n, t - k) for k in range(a, min(t, n) + 1) if t - k <= n) / comb(2 * n, t)
def family(path, task, state, ids):
    r = rows(path)
    got = [int(x["seed"]) - 1000 for x in r]  # noise id k is episode seed 1000 + k
    assert sorted(got) == list(ids), (path, len(got), len(ids))
    assert all(int(x["task_id"]) == task and int(x["init_state"]) == state for x in r), path
    return sorted(int(x["seed"]) - 1000 for x in r if win(x))
for t in TASKS:
    h = [x for p in glob.glob(f"{root}/task_{t}/harness/seed*/per_episode_base.csv") for x in rows(p)]
    assert len(h) == 30 and len({(x["seed"], x["init_state"]) for x in h}) == 30
    n = sum(map(win, h))
    c = {f: len(family(f"{root}/task_{t}/control/per_episode_{f}_state{CONTROL[t]}.csv", t, CONTROL[t], range(35, 70)))
         for f in ("sampled", "repeat")}
    print(f"task {t}: harness {n}/30 vs {REF[t]} (gap {n - REF[t]:+d}); control state {CONTROL[t]} of 35 {c}")
tests = [("test 1", 6, 8, range(100, 400))] + [("test 2", t, s, range(1000 + 150 * i, 1150 + 150 * i)) for i, (t, s) in enumerate(TEST2)]
pooled = {"sampled": 0, "repeat": 0, "n": 0}
ps, jit = {}, [0, 0]
for g, t, s, ids in tests:
    hits = {f: family(f"{root}/task_{t}/test/per_episode_{f}_state{s}.csv", t, s, ids) for f in ("sampled", "repeat")}
    n = len(ids)
    print(f"{g} {t}/{s} ids {ids[0]}-{ids[-1]}: sampled {len(hits['sampled'])}/{n} {hits['sampled']}; "
          f"repeat {len(hits['repeat'])}/{n} {hits['repeat']}")
    files = sorted(glob.glob(f"{root}/task_{t}/test/confirm_repeat_n*_state{s}.csv"))
    named = sorted(int(re.search(r"_n(\d+)_", os.path.basename(p)).group(1)) for p in files)
    assert named == hits["repeat"][:3], (t, s, named, hits["repeat"][:3])  # the first three repeat hits
    for p in files:
        r = rows(p)
        assert sorted(int(x["seed"]) for x in r) == [5000, 5001, 5002, 5003], p
        assert all(int(x["task_id"]) == t and int(x["init_state"]) == s for x in r), p
        jit[0] += sum(map(win, r)); jit[1] += len(r)
        print(f"   jittered {os.path.basename(p)}: {sum(map(win, r))}/4")
    if g == "test 1":
        ps[g] = fisher(len(hits["repeat"]), len(hits["sampled"]), n)
    else:
        pooled["sampled"] += len(hits["sampled"]); pooled["repeat"] += len(hits["repeat"]); pooled["n"] += n
ps["test 2"] = fisher(pooled["repeat"], pooled["sampled"], pooled["n"])
print("test 2 pooled", pooled)
order = sorted(ps, key=lambda k: (ps[k], k))
passed, go = {}, True
for i, k in enumerate(order):
    go = go and ps[k] < 0.05 / (len(ps) - i)
    passed[k] = go
for k in ps:
    print(f"{k}: one-sided Fisher p {ps[k]:.4g}, Holm {'passes' if passed[k] else 'does not pass'}")
print(f"jittered successes {jit[0]}/{jit[1]}")
recs = [json.loads(l) for p in glob.glob(f"{root}/task_*/**/provenance.jsonl", recursive=True) for l in open(p)]
print("provenance records", len(recs), "commits", {r["git_commit"] for r in recs}, "dirty", {r["git_dirty"] for r in recs},
      "hosts", sorted({r["host"] for r in recs}), "n_action_steps", {r["settings"]["n_action_steps"] for r in recs},
      "max_steps", {r["settings"]["max_steps"] for r in recs})
