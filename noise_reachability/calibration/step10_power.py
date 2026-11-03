"""Power of step 10's test 1 (README.md, step 10), by simulation; an estimate, not a result.

Per training seed, N and the base each play the 192 trained episodes (all three tasks read); outcomes are independent
draws, N at rate p_n, the base at rate p_b. Test 1 passes when the one-sided exact McNemar p is under 0.05 for at least
2 of 3 seeds (analyze_step10.mcnemar_greater). The base is simulated once and shared by the three seeds, as in the run.
    python noise_reachability/calibration/step10_power.py [--sims 4000 --seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_step10 import ALPHA, MIN_SEEDS, TRAIN_SEEDS, mcnemar_greater  # noqa: E402

EPISODES = 6 * 16 + 2 * 32 + 32


def power(p_n, p_b, sims, rng, n=EPISODES):
    passes = 0
    for _ in range(sims):
        base = rng.random(n) < p_b
        ok = 0
        for _ in TRAIN_SEEDS:
            arm = rng.random(n) < p_n
            ok += mcnemar_greater(int((arm & ~base).sum()), int((base & ~arm).sum())) < ALPHA
        passes += ok >= MIN_SEEDS
    return passes / sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"{EPISODES} trained episodes per seed, test 1 = p < {ALPHA} in {MIN_SEEDS} of {len(TRAIN_SEEDS)} seeds, "
          f"sims {a.sims}, seed {a.seed}")
    for p_b in (0.0, 0.005):
        row = [f"N {p_n:.1%}: {power(p_n, p_b, a.sims, rng):.3f}" for p_n in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)]
        print(f"base {p_b:.1%}: " + ", ".join(row))


if __name__ == "__main__":
    main()
