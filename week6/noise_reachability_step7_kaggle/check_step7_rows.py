"""Separate recount of step 7 from the raw CSVs, sharing no code with analyze_step7.py.
    python -I -S check_step7_rows.py <run root> <repo root>"""
import csv, glob, json, os, re, sys
from math import comb
root, repo = sys.argv[1], sys.argv[2]
def rows(p): return list(csv.DictReader(open(p)))
SEARCH = {4: 2, 6: 8, 8: 4}
CONTROL = {4: 5, 6: 0, 8: 8}
FAM = ("sampled", "full", "repeat")
def fisher(a, b, n):
    t = a + b
    return sum(comb(n, k) * comb(n, t - k) for k in range(a, min(t, n) + 1)) / comb(2 * n, t)
tot = {f: 0 for f in FAM}
for label, states, n in (("search", SEARCH, 70), ("control", CONTROL, 35)):
    for t, s in states.items():
        for f in FAM:
            r = rows(f"{root}/task_{t}/per_episode_{f}_state{s}.csv")
            ids = [int(x["seed"]) - 1000 for x in r]  # noise_id k is episode seed 1000 + k
            assert len(ids) == n and sorted(ids) == list(range(n)), (t, s, f, len(ids), ids[:5])
            assert all(int(x["task_id"]) == t and int(x["init_state"]) == s for x in r)
            hit = [(int(x["seed"]) - 1000, int(x["steps"])) for x in r if str(x["success"]).lower() in ("1", "true", "1.0")]
            print(f"{label} {t}/{s} {f}: {len(hit)}/{n} ids,steps {hit if label == 'search' else ''}")
            if label == "search":
                tot[f] += len(hit)
print("totals", tot, "repeat v sampled p", round(fisher(tot["repeat"], tot["sampled"], 210), 4))
for t in SEARCH:
    print("harness", t, sum(str(x["success"]).lower() in ("1", "true", "1.0") for p in glob.glob(f"{root}/task_{t}/harness/seed*/per_episode_base.csv") for x in rows(p)))
conf = sorted(glob.glob(f"{root}/task_*/confirm_*.csv"))
print(len(conf), "confirmation files")
for p in conf:
    r = rows(p)
    print(os.path.relpath(p, root), "ids", [x["seed"] for x in r], "successes", sum(str(x["success"]).lower() in ("1", "true", "1.0") for x in r))
recs = [json.loads(l) for p in glob.glob(f"{root}/task_*/provenance.jsonl") + glob.glob(f"{root}/task_*/harness/*/provenance.jsonl") for l in open(p)]
print("provenance records", len(recs), "commits", {r["git_commit"] for r in recs}, "dirty", {r["git_dirty"] for r in recs},
      "n_action_steps", {r["settings"].get("n_action_steps") for r in recs}, "gpus", {tuple(r["gpus"]) for r in recs})
paths = {str(r["settings"].get("policy_path")) for r in recs}
print("policy", {re.sub(r"^.*snapshots/", "", p)[:12] for p in paths})
# earlier base episodes on the search states, seeds 0-31
base = {}
for p in (f"{repo}/week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv", f"{repo}/week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv"):
    r = rows(p)
    print(os.path.basename(os.path.dirname(p)), "columns", list(r[0]))
    for x in r:
        if int(x["task_id"]) in SEARCH and int(x["init_state"]) == SEARCH[int(x["task_id"])]:
            base.setdefault(int(x["task_id"]), []).append(x)
for t, r in base.items():
    print("earlier base", t, len(r), "hits", [x["seed"] for x in r if str(x["success"]).lower() in ("1", "true", "1.0")])


# Exploratory, not part of the rule.
def fisher2(a, n, b, m):
    """one-sided: a/n at least this far above b/m"""
    t = a + b
    return sum(comb(n, k) * comb(m, t - k) for k in range(a, min(t, n) + 1) if t - k <= m) / comb(n + m, t)
def wins(p, **want):
    return sum(str(x["success"]).lower() in ("1", "true", "1.0") for x in rows(p)
               if all(int(x[k]) == v for k, v in want.items()))
step6 = f"{repo}/week6/noise_reachability_step6_kaggle/run/eval/per_episode_base.csv"
stepB = f"{repo}/week6/noise_reachability_step6_kaggle/run/eval/per_episode_B.csv"
for t, s in SEARCH.items():
    old = sum(str(x["success"]).lower() in ("1", "true", "1.0") for x in base[t]) + wins(step6, task_id=t, init_state=s)
    new = wins(f"{root}/task_{t}/per_episode_sampled_state{s}.csv")
    rep = wins(f"{root}/task_{t}/per_episode_repeat_state{s}.csv")
    ful = wins(f"{root}/task_{t}/per_episode_full_state{s}.csv")
    arm = wins(stepB, task_id=t, init_state=s)
    print(f"{t}/{s}: base seeds 0-31 {old}/32, step 7 sampled {new}/70, together {old + new}/102; "
          f"repeat {rep}/70 p {fisher2(rep, 70, old + new, 102):.3f}; full {ful}/70 p {fisher2(ful, 70, old + new, 102):.3f}; "
          f"B seeds 16-31 {arm}/16 p {fisher2(arm, 16, old + new, 102):.2g} (all against the /102)")
    conf = {f: [wins(p) for p in glob.glob(f"{root}/task_{t}/confirm_{f}_n*_state{s}.csv")] for f in ("repeat", "full")}
    jit = {f: (sum(v), 4 * len(v)) for f, v in conf.items()}
    print(f"   jittered {jit}; repeat against the /102: p {fisher2(*jit['repeat'], old + new, 102):.2g}")
ctl = {f: sum(wins(f"{root}/task_{t}/per_episode_{f}_state{s}.csv") for t, s in CONTROL.items()) for f in FAM}
print("controls pooled, of 105:", ctl)
