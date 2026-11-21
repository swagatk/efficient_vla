"""Power of step 16's tests T1 and T2 (README.md, step 16); a Monte Carlo estimate on simulated data, not a result.

The tests are step16_g1.py's own functions, run on simulated crossed maps: 4 tasks x S states x K noise ids. Outcomes
are Bernoulli with logit = a_s + b_k + c_sk:

- a_s ~ N(0, SD_STATE^2): the state's difficulty (wide, since states 0-29 run from never to always solved);
- b_k ~ N(0, sd_id^2), per task: a noise's quality shared by the task's states (what T1 detects);
- c_sk = sd_inter * <z_s, v_k> / sqrt(2), z_s, v_k ~ N(0, I_2): a scene-dependent effect, smooth in a 2-d scene latent
  z_s (what T2 detects). The context the test sees is z_s plus d_noise irrelevant N(0, 1) coordinates; LIBERO's
  object contexts have 8-16 coordinates per task, of which an unknown number matter.
- the confound (review of 2026-11-20): difficulty set by the scene, a_s = conf * z_s1 + N(0, SD_STATE^2 - conf^2),
  with a shared id effect and no c_sk. T2 must not pass; with a free permutation of the contexts it does.

Power is the fraction of simulations whose permutation p is below 0.025, the stricter of Holm's two levels (the
cautious side); the rate at an effect of 0 is the test's false-positive rate at that level.

    python noise_reachability/calibration/step16_power.py [--sims 100] [--n_perm 100]
    python noise_reachability/calibration/step16_power.py --level_sims 300 --seed 2017   # only T2's level, confound
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import step16_g1 as g  # noqa: E402

SD_STATE = 2.0
STATES = 30
LEVEL = 0.025


def simulate(rng, S, K, sd_id, sd_inter, d_noise, tasks=4, conf=0.0):
    out = {}
    for t in range(tasks):
        z = rng.normal(size=(S, 2))
        v = rng.normal(size=(K, 2))
        a_s = conf * z[:, :1] + rng.normal(0, np.sqrt(SD_STATE ** 2 - conf ** 2), size=(S, 1))
        logit = (a_s + rng.normal(0, sd_id, size=(1, K))
                 + sd_inter * (z @ v.T) / np.sqrt(2))
        Y = (rng.random((S, K)) < 1 / (1 + np.exp(-logit))).astype(int)
        out[t] = (Y, np.hstack([z, rng.normal(size=(S, d_noise))]))
    return out


def power(test, sims, rng, **kw):
    hits = 0
    for _ in range(sims):
        hits += test(simulate(rng, STATES, **kw)) < LEVEL
    return hits / sims


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2016)
    ap.add_argument("--level_sims", type=int, default=0, help="only check T2's stratified null at this many simulations")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    import subprocess
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "noise_reachability"],
                                capture_output=True, text=True).stdout.strip())
    print(f"step 16 power: 4 tasks x {STATES} states, state sd {SD_STATE}, {a.sims} simulations, {a.n_perm} "
          f"permutations, level {LEVEL}, seed {a.seed}; an estimate, not a result")
    print(f"command: python noise_reachability/calibration/step16_power.py --sims {a.sims} --n_perm {a.n_perm} "
          f"--seed {a.seed}; commit {head}{' with uncommitted noise_reachability changes' if dirty else ''}")
    if a.level_sims:
        print(f"command: ... --level_sims {a.level_sims} --n_perm {a.n_perm} --seed {a.seed}")
        print(f"T2 stratified null, level: {a.level_sims} simulations, K 50, m 8, id sd 1.0, no scene-dependent effect; "
              f"rate of p < {LEVEL} (and < 0.05)")
        for d_noise in (0, 13):
            for conf in (0.0, 2.0):
                ps = np.array([g.t2_test(simulate(rng, STATES, K=50, sd_id=1.0, sd_inter=0.0, d_noise=d_noise, conf=conf),
                                         m=8, n_perm=a.n_perm, seed=int(rng.integers(1 << 30)))["p"]
                               for _ in range(a.level_sims)])
                print(f"  {d_noise} irrelevant, conf {conf}: {np.mean(ps < LEVEL):.3f} ({np.mean(ps < 0.05):.3f})", flush=True)
        print(f"({time.time() - t0:.0f} s)")
        return
    print("T1 (shared id effect), by ids per state K and id sd:")
    for K in (10, 20, 50):
        row = [power(lambda d: g.t1_test(d, n_perm=a.n_perm, seed=int(rng.integers(1 << 30)))["p"], a.sims, rng,
                     K=K, sd_id=s, sd_inter=0.0, d_noise=0) for s in (0.0, 0.3, 0.5, 1.0)]
        print(f"  K {K}: id sd 0, 0.3, 0.5, 1.0 -> " + ", ".join(f"{x:.2f}" for x in row), flush=True)
    print("T2 false positives under the confound (difficulty from the scene, id sd 1.0, no scene-dependent effect), "
          "K 50, m 8, by scene weight conf 0, 1, 2:")
    for strat, name in ((None, "free permutation (invalid)"), (g.STRATUM_SIZE, f"strata of {g.STRATUM_SIZE}")):
        for d_noise in (0, 13):
            row = [power(lambda d: g.t2_test(d, m=8, n_perm=a.n_perm, seed=int(rng.integers(1 << 30)),
                                             stratum_size=strat)["p"], a.sims, rng, K=50, sd_id=1.0, sd_inter=0.0,
                         d_noise=d_noise, conf=c) for c in (0.0, 1.0, 2.0)]
            print(f"  {name}, {d_noise} irrelevant: " + ", ".join(f"{x:.2f}" for x in row), flush=True)
    print(f"T2 (scene-dependent effect, strata of {g.STRATUM_SIZE}), by K, neighbours m, irrelevant context "
          "coordinates and interaction sd:")
    for K, m in ((50, 5), (50, 8), (20, 8)):
        for d_noise in (0, 13):
            row = [power(lambda d: g.t2_test(d, m=m, n_perm=a.n_perm, seed=int(rng.integers(1 << 30)))["p"], a.sims,
                         rng, K=K, sd_id=0.5, sd_inter=s, d_noise=d_noise) for s in (0.0, 0.5, 1.0, 1.5)]
            print(f"  K {K}, m {m}, {d_noise} irrelevant: inter sd 0, 0.5, 1.0, 1.5 -> "
                  + ", ".join(f"{x:.2f}" for x in row), flush=True)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
