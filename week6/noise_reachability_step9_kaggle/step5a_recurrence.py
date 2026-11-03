"""Exploratory (step 9 motivation): do repeat/full hits on one task-0 state recur on other states with the same noise id?
    python -I -S step5a_recurrence.py week6/noise_reachability_step5a_colab/run
"""
import csv, sys, itertools, random
root = sys.argv[1]
def hits(f, s):
    r = list(csv.DictReader(open(f"{root}/per_episode_{f}_state{s}.csv")))
    return {int(x["seed"]) - 1000 for x in r if str(x["success"]).lower() in ("1", "true", "1.0")}, len(r)
for fam in ("repeat", "full", "sampled"):
    H = {s: hits(fam, s) for s in (2, 5, 6, 7, 8)}
    print(fam, {s: (sorted(h), n) for s, (h, n) in H.items()})
    # pairs of (state, state) sharing a hit noise id, against a shuffle of ids within each state
    def shared(sets): return sum(len(a & b) for a, b in itertools.combinations(sets, 2))
    obs = shared([h for h, _ in H.values()])
    rng = random.Random(0); ge = 0; N = 20000; tot = 0
    for _ in range(N):
        sim = [set(rng.sample(range(70), len(h))) for h, _ in H.values()]
        v = shared(sim); tot += v; ge += v >= obs
    print(f"  shared hit ids across state pairs: {obs}, expected {tot/N:.2f} under shuffled ids, p(>=) {ge/N:.4f}")
