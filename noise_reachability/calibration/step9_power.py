"""Power and false-pass rate of step 9's test (README.md, step 9), by simulation.

Per task, L library noise ids and R random noise ids, each run once on each of the task's two held-out states. The
test is analyze_step9.permutation_p: the exact one-sided p of the library's successes under relabelling noise ids within
each task. The simulation draws outcomes independently per (id, state): `random` ids succeed with rate p0, library ids with
rate m * p0. Correlation between a noise's outcomes on two states is not simulated.
    python noise_reachability/calibration/step9_power.py [--sims 1000 --seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_step9 import LIBRARY as LIBRARY_IDS, RANDOM_IDS, permutation_p  # noqa: E402

LIBRARY = {t: len(v) for t, v in LIBRARY_IDS.items()}
RANDOM_PER_TASK = len(RANDOM_IDS)
STATES_PER_TASK = 2
ALPHA = 0.05


def power(p0, m, sims, rng):
    passes = 0
    for _ in range(sims):
        lib = [rng.binomial(STATES_PER_TASK, min(1.0, m * p0), size=n) for n in LIBRARY.values()]
        ran = [rng.binomial(STATES_PER_TASK, p0, size=RANDOM_PER_TASK) for _ in LIBRARY]
        per_task = {t: (list(a), list(b)) for t, a, b in zip(LIBRARY, lib, ran)}
        passes += permutation_p(per_task) < ALPHA
    return passes / sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"library ids {LIBRARY}, random ids {RANDOM_PER_TASK} per task, {STATES_PER_TASK} states per task, "
          f"sims {a.sims}, seed {a.seed}, exact p")
    for p0 in (0.02, 0.05, 0.10):
        row = [f"m={m}: {power(p0, m, a.sims, rng):.3f}" for m in (1, 1.5, 2, 3)]
        print(f"random rate {p0:.2f}: " + ", ".join(row))


if __name__ == "__main__":
    main()
