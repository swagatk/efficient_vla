"""Exact power of step 7's rate rule (README.md, step 7): one-sided Fisher, repeat against sampled, 210 episodes each."""
from itertools import combinations
from math import comb

N = 210


def binom(k, n, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def fisher_greater(a, b, n=N):
    """P(repeat has at least a successes) given a + b successes in all, two groups of n."""
    total = a + b
    return sum(comb(n, i) * comb(n, total - i) for i in range(a, min(total, n) + 1)) / comb(2 * n, total)


def pooled(rates):
    """{(repeat hits over 3 states, states with a hit): probability}, 70 noises per state, independent."""
    out = {(0, 0): 1.0}
    for rate in rates:
        state = [binom(k, 70, rate) for k in range(71)]
        next_ = {}
        for (total, spread), q in out.items():
            for k, w in enumerate(state):
                if q * w > 1e-13:
                    key = (total + k, spread + (k > 0))
                    next_[key] = next_.get(key, 0) + q * w
        out = next_
    return out


def power(rates, sampled):
    """(rate rule passes, rate rule passes with a repeat hit on at least 2 states)."""
    b = [binom(k, N, sampled) for k in range(40)]
    rule = spread_rule = 0.0
    for (a, spread), q in pooled(rates).items():
        for k, w in enumerate(b):
            if q * w > 1e-13 and fisher_greater(a, k) < 0.05:
                rule += q * w
                spread_rule += q * w * (spread >= 2)
    return rule, spread_rule


print(f"smallest repeat count that passes with sampled 0: {min(a for a in range(N) if fisher_greater(a, 0) < 0.05)}")
print(f"with sampled 1: {min(a for a in range(N) if fisher_greater(a, 1) < 0.05)}")
print(f"with sampled 2: {min(a for a in range(N) if fisher_greater(a, 2) < 0.05)}")
print("same true rate on each state: true repeat rate, true sampled rate, rate rule, rate rule and spread")
for sampled in (0.0, 0.01, 0.02):
    for rate in (0.01, 0.02, 0.03, 0.057):
        print("{:.3f} {:.3f} {:.3f} {:.3f}".format(rate, sampled, *power([rate] * 3, sampled)))
print("chance of at least one repeat hit on a state in 70 noises, true rate: value")
for rate in (1 / 70, 0.02, 0.03, 0.057):
    print(f"{rate:.3f}: {1 - (1 - rate) ** 70:.3f}")
print("false pass of the rate rule at equal true rates")
for rate in (0.01, 0.02):
    print("{:.3f}: {:.4f}".format(rate, power([rate] * 3, rate)[0]))
print("step 5a's repeat rates (1, 1, 7, 3, 8 of 70), mean over its 10 subsets of 3 states: true sampled rate, rate rule, rate rule and spread")
for sampled in (0.0, 0.01, 0.02):
    runs = [power([k / 70 for k in subset], sampled) for subset in combinations((1, 1, 7, 3, 8), 3)]
    print("{:.3f} {:.3f} {:.3f}".format(sampled, sum(r[0] for r in runs) / 10, sum(r[1] for r in runs) / 10))
