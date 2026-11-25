"""Power of step 18's rule (README.md, step 18), Monte Carlo with analyze_step18's own functions; an estimate, not a
result.

Each state's constant-noise (`repeat`) rate is its pooled committed count on fresh ids (from step18_states.py, Kaggle
T4 except step 5a's Colab episodes) times a shrink factor, since the states were picked for their high rates (winner's curse; the
factor is a guess, so a range). A tested condition succeeds at theta times that rate on every state. Episodes are
independent: every state has its own noise ids. Counts are binomial per state, pooled as the rule pools them.

    python3 -S noise_reachability/calibration/step18_power.py [--sims 1000] [--seed 18]
"""
import argparse
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_step18 as a  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step18_states as st  # noqa: E402

SHRINKS = (1.0, 0.7, 0.5)
N_PER_STATE = (50, 75, 100)
THETAS_S = (0.0, 0.25, 0.5, 0.75, 1.0)
THETAS_R = (0.5, 1.0)

a.fisher_greater = lru_cache(maxsize=None)(a.fisher_greater)     # pure function; the rule calls it by module name
a.ratio_interval = lru_cache(maxsize=None)(a.ratio_interval)


def binom(rng, n, p):
    return sum(rng.random() < p for _ in range(n))


def simulate(rng, rates, n, theta_s, theta_r):
    pooled = Counter()
    for p in rates:
        pooled["repeat"] += binom(rng, n, p)
        pooled["scaled_sigma0.5"] += binom(rng, n, theta_s * p)
        pooled["ar_repeat_rho0.9"] += binom(rng, n, theta_r * p)
    return a.rule(dict(pooled), n * len(rates))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=18)
    args = ap.parse_args(argv)
    rows = st.select(st.candidates(st.tally(st.ROOT / "week6", st.REPEAT), st.tally(st.ROOT / "week6", st.SAMPLED)))
    prior = {k: (rs, rn) for k, rs, rn, *_ in rows}           # repeat successes, episodes on fresh ids (README "States")
    print("command: python3 -S noise_reachability/calibration/step18_power.py " + " ".join(argv if argv is not None else sys.argv[1:]))
    print(f"step 18 power: {args.sims} simulations per cell, seed {args.seed}; states "
          + ", ".join(f"{t}/{s} {k}/{m}" for (t, s), (k, m) in prior.items()))
    for shrink in SHRINKS:
        rates = [shrink * k / m for k, m in prior.values()]
        print(f"\nshrink {shrink}: repeat rate pooled {sum(rates) / len(rates):.4f}")
        for n in N_PER_STATE:
            for ts in THETAS_S:
                for tr in THETAS_R:
                    rng = random.Random(f"{args.seed}-{shrink}-{n}-{ts}-{tr}")
                    res = [simulate(rng, rates, n, ts, tr) for _ in range(args.sims)]
                    read = [r for r in res if r["read"]]
                    f = lambda c, k: sum(bool(r["conditions"][c][k]) for r in read) / args.sims
                    readings = Counter(r["reading"].split(":")[0] for r in res)
                    print(f"  n {n}/state, theta sigma0.5 {ts:.2f}, rho0.9 {tr:.2f}: read {len(read) / args.sims:.3f}; "
                          f"sigma0.5 below {f('scaled_sigma0.5', 'below'):.3f} keeps half "
                          f"{f('scaled_sigma0.5', 'keeps_half'):.3f}; rho0.9 below {f('ar_repeat_rho0.9', 'below'):.3f} "
                          f"keeps half {f('ar_repeat_rho0.9', 'keeps_half'):.3f}; readings "
                          + ", ".join(f"{k} {v / args.sims:.2f}" for k, v in sorted(readings.items())))


if __name__ == "__main__":
    main()
