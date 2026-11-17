"""Power of step 15's two tests (README.md, step 15); exact for test A, a Monte Carlo estimate for test B, not a result.

Test A: the base with n fresh `repeat` noises on one state (6/14 or 6/20) finds at least one success. With a true
per-noise rate p this happens with probability 1 - (1 - p)^n; if none is found, the exact one-sided 95% upper bound
on p is 1 - 0.05^(1/n).

Test B: on 7/38, per training seed s, the base against N7_s on the same n noise ids (paired), exact one-sided McNemar
(base better), passes with p < 0.05 in at least 2 of 3 seeds. The base's outcome on a noise id is drawn once per
simulation and shared by the three seeds' comparisons, as in the run (the same base episodes pair with N7_0, N7_1 and
N7_2). Given the base's outcome, N7_s's is drawn independently at its own rate; on the same noise the two are positively
correlated in practice, which gives fewer discordant pairs for the same gap and so more power, so this is the cautious
side. The base rate starts from step 9 on 7/38 (committed run: 7 of 50 fresh `repeat` ids).

    python noise_reachability/calibration/step15_power.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
from analyze_step10 import mcnemar_greater  # noqa: E402


def p_find(p: float, n: int) -> float:
    """Probability of at least one success in n independent episodes at rate p."""
    return 1 - (1 - p) ** n


def upper_bound_zero(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) upper bound on the rate after 0 successes in n episodes."""
    return 1 - alpha ** (1 / n)


def power_b(base: float, adapter: float, n: int, sims: int = 4000, seed: int = 0) -> float:
    rng = random.Random(seed)
    passed = 0
    for _ in range(sims):
        base_ok = [rng.random() < base for _ in range(n)]
        wins = 0
        for _ in range(3):
            b = c = 0
            for x in base_ok:
                y = rng.random() < adapter
                b += x and not y
                c += y and not x
            wins += mcnemar_greater(b, c) < 0.05
        passed += wins >= 2
    return passed / sims


def main():
    for n in (300, 400, 600):
        cells = [f"p {p:.3f}: {p_find(p, n):.3f}" for p in (0.002, 0.005, 0.01, 0.02)]
        print(f"test A, n = {n} per state: P(at least one success) " + ", ".join(cells)
              + f"; 0 of {n} gives an upper bound of {upper_bound_zero(n):.4f}")
    base = 7 / 50
    for n in (100, 200):
        cells = [f"N7 at {a:.3f}: {power_b(base, a, n):.3f}" for a in (base, 0.10, 0.07, 0.035, 0.0)]
        print(f"test B, n = {n} paired per seed on 7/38, base at {base:.3f}: P(pass) " + ", ".join(cells))


if __name__ == "__main__":
    main()
