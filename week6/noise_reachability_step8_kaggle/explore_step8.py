"""Exploratory, not part of the pre-registered rule."""
import json, sys
from math import comb
r = json.load(open(sys.argv[1]))
def fisher(a, b, n):
    t = a + b
    return sum(comb(n, k) * comb(n, t - k) for k in range(a, min(t, n) + 1) if t - k <= n) / comb(2 * n, t)
fresh = [s for s in r["states"] if s["group"] == "test 2"]
for s in r["states"]:
    print(f"{s['task']}/{s['state']}: repeat {s['repeat']} sampled {s['sampled']} of {s['episodes']}, per-state p {fisher(s['repeat'], s['sampled'], s['episodes']):.3g}")
for drop in fresh:
    rest = [s for s in fresh if s is not drop]
    a, b, n = sum(s["repeat"] for s in rest), sum(s["sampled"] for s in rest), sum(s["episodes"] for s in rest)
    print(f"test 2 without {drop['task']}/{drop['state']}: repeat {a}/{n} sampled {b}/{n} p {fisher(a, b, n):.3g}")
c = r["control"]
a, b = sum(v["repeat"] for v in c.values()), sum(v["sampled"] for v in c.values())
print(f"controls pooled of {35*len(c)}: repeat {a} sampled {b}; sampled>repeat one-sided p {fisher(b, a, 35*len(c)):.3g}")
print(f"task 6 control: sampled {c['6']['sampled']} > repeat {c['6']['repeat']} one-sided p {fisher(c['6']['sampled'], c['6']['repeat'], 35):.3g}")
