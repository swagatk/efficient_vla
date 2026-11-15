"""Power of step 14's tests A and B (README.md, step 14); a Monte Carlo estimate, not a result.

Both tests compare two adapters on state 0/8, per training seed, with an exact one-sided McNemar test on n paired
episodes, and pass with p < 0.05 in at least 2 of 3 seeds (as analyze_step14 reads them). The higher policy succeeds
at rate `hi`, the lower at `lo`, on every seed. Episodes are drawn independently for the two policies; on the same
scene they are positively correlated in practice, which gives fewer discordant pairs for the same gap and so more
power, so this is the cautious side. The rates start from step 12 on 0/8 (committed report): N0-N2 41 of 48 pooled,
R0-R2 32 of 48.

    python noise_reachability/calibration/step14_power.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
from analyze_step10 import mcnemar_greater  # noqa: E402


def power(hi: float, lo: float, n: int, sims: int = 4000, seed: int = 0) -> float:
    rng = random.Random(seed)
    passed = 0
    for _ in range(sims):
        wins = 0
        for _ in range(3):
            b = c = 0
            for _ in range(n):
                x, y = rng.random() < hi, rng.random() < lo
                b += x and not y
                c += y and not x
            wins += mcnemar_greater(b, c) < 0.05
        passed += wins >= 2
    return passed / sims


def main():
    n_rate, r_rate = 41 / 48, 32 / 48
    print(f"step 12 on 0/8: N {n_rate:.3f}, R {r_rate:.3f} (seeds pooled)")
    for n in (32, 48, 64, 96):
        cells = [f"gap {d:.2f}: {power(n_rate, n_rate - d, n):.3f}" for d in (0.0, 0.05, 0.10, n_rate - r_rate)]
        print(f"n = {n} episodes per seed on 0/8, higher policy at {n_rate:.3f}: P(pass) " + ", ".join(cells))


if __name__ == "__main__":
    main()
