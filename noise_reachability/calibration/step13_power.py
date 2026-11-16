"""Power of step 13's tests (README.md, step 13); a Monte Carlo estimate, not a result.

Per training seed, an adapter is compared with the base on the same n = 59 states x k seeds episodes, with an exact
two-sided McNemar test; the test finds a change if p < 0.05 in at least 2 of 3 seeds, all in the same direction (as
analyze_step13 reads it). The base succeeds at 0.40 on every episode (task 0 had 25 of 60 and task 6 22 of 60 base
successes on states 30-49 in earlier collections; task 7's range includes resistant states), the adapter at
0.40 + d. Episodes are drawn independently for the two policies; on the same scene they are positively correlated in
practice, which gives fewer discordant pairs for the same change and so more power: this is the cautious side.

    python noise_reachability/calibration/step13_power.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
from analyze_step12 import mcnemar_two_sided  # noqa: E402

STATES = 59
BASE = 0.40


def power(d: float, k: int, sims: int = 2000, seed: int = 0) -> float:
    rng = random.Random(seed)
    n, q = STATES * k, BASE + d
    found = 0
    for _ in range(sims):
        signs = []
        for _ in range(3):
            b = c = 0
            for _ in range(n):
                x, y = rng.random() < q, rng.random() < BASE
                b += x and not y
                c += y and not x
            if mcnemar_two_sided(b, c) < 0.05:
                signs.append(1 if b > c else -1)
        found += max(signs.count(1), signs.count(-1)) >= 2
    return found / sims


def main():
    print(f"base at {BASE} on {STATES} states; per seed k episodes per state; change found if two-sided p < 0.05 in "
          "at least 2 of 3 seeds in the same direction")
    for k in (4, 8, 12):
        cells = [f"d {d:+.2f}: {power(d, k):.3f}" for d in (0.0, 0.03, 0.05, 0.10, -0.05, -0.10)]
        print(f"k = {k} ({STATES * k} pairs per seed): " + ", ".join(cells))


if __name__ == "__main__":
    main()
