"""Exact power of step 8's two primary tests (README.md, step 8): one-sided Fisher, repeat against sampled, n episodes each.
Holm over 2 tests: the first needs p < 0.025, the second p < 0.05 once the first passed. Episodes are independent because
every state has its own noise ids."""
from math import comb, exp, lgamma, log

GRID = ((0.0, 0.02), (0.0, 0.03), (0.0, 0.043), (0.0, 0.057), (0.005, 0.03), (0.005, 0.043), (0.005, 0.057),
        (0.01, 0.03), (0.01, 0.043), (0.01, 0.057), (0.02, 0.043), (0.02, 0.057), (0.048, 0.076))


def binom(k, n, p):
    if p == 0:
        return float(k == 0)
    return exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p))


def fisher_greater(a, b, n):
    total = a + b
    return sum(comb(n, i) * comb(n, total - i) for i in range(a, min(total, n) + 1)) / comb(2 * n, total)


def threshold(b, n, alpha):
    """smallest repeat count that passes when sampled has b hits"""
    return next(a for a in range(n + 1) if fisher_greater(a, b, n) < alpha)


def power(sampled, repeat, n, alpha):
    top = min(n, int(n * sampled + 8 * (n * sampled + 1) ** 0.5) + 5)
    total = 0.0
    for b in range(top + 1):
        w = binom(b, n, sampled)
        if w < 1e-12:
            continue
        need = threshold(b, n, alpha)
        total += w * sum(binom(a, n, repeat) for a in range(need, n + 1))
    return total


for n, name in ((300, "6/8, 300 each"), (480, "fresh pooled, 4 states x 120 each"), (600, "fresh pooled, 4 states x 150 each"),
                (360, "fresh pooled, 3 states x 120 each (fewer states eligible)")):
    print(f"{name}: hits needed at sampled 0, 1, 2, 3: {[threshold(b, n, 0.05) for b in range(4)]} at alpha .05, "
          f"{[threshold(b, n, 0.025) for b in range(4)]} at alpha .025")
    print("  true sampled, true repeat, power at .025, power at .05")
    for s, r in GRID:
        print(f"  {s:.3f} {r:.3f} {power(s, r, n, 0.025):.3f} {power(s, r, n, 0.05):.3f}")
    print(f"  false pass at equal rates 0.01: {power(0.01, 0.01, n, 0.05):.4f}; 0.05: {power(0.05, 0.05, n, 0.05):.4f}")
print("upper one-sided 95% bound on sampled after 0 hits: n, bound")
for n in (32, 102, 300, 402, 480):
    print(n, round(1 - 0.05 ** (1 / n), 4))
print(f"chance a repeat control is under 3 of 35 at step 7's 6/0 rate, 5 of 35 (estimate): {sum(binom(k, 35, 5 / 35) for k in range(3)):.3f}")
print("chance a state passes the 0-of-32 screen, true sampled rate: value")
for p in (0.0, 0.01, 0.02, 0.05, 0.1, 0.3):
    print(p, round((1 - p) ** 32, 3))
