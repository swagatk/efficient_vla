"""Power and level of step 17's test U1 (README.md, step 17); a Monte Carlo estimate on simulated data, not a result.

The test is step17_g1b.py's own u1_test, run on simulated crossed maps: 4 tasks x 30 states x 50 noise ids, outcomes
Bernoulli with step 16's model, logit = a_s + b_k + c_sk (calibration/step16_power.py):

- a_s ~ N(0, SD_STATE^2), or under the confound a_s = conf * z_s1 + N(0, SD_STATE^2 - conf^2) (difficulty set by the
  scene, the reason the null is stratified);
- b_k ~ N(0, sd_id^2) per task, the shared id effect (the per-task ticket);
- c_sk = sd_inter * <z_s, v_k> / sqrt(2), z_s, v_k ~ N(0, I_2): the scene-dependent effect U1 is meant to find.

Features stand in for SmolVLA's pooled prefix features: X_s = sqrt(share) * z_s A + sqrt(1 - share) * n_s B, with
z_s the 2-d scene latent, n_s ~ N(0, I_N_IRRELEVANT) directions that carry nothing about the outcome, and A, B random
maps into FEATURE_DIM dimensions with unit-variance rows, so ``share`` is the fraction of the feature variance that
follows the scene. Nothing is known about this share in the real features, so a range is simulated.

A simulation "passes" under the rule's own condition: p <= 0.05, U1 > 0 and D > 0. Power is the pass rate with an
interaction; level is the pass rate without one.

Stages (run in this order; the README records which choice they led to):

    select: every (d, lambda) of the grid at sd_inter 1.0 and each share; the pair with the highest mean power.
    level:  the chosen pair without interaction, under the confound (conf 0 and 2), shares 0.2 and 0.5.
    curve:  the chosen pair's power by sd_inter and share.

    python noise_reachability/calibration/step17_power.py --stage select [--sims 50 --n_perm 50]
    python noise_reachability/calibration/step17_power.py --stage level --d 4 --lam 1 [--sims 300 --n_perm 100]
    python noise_reachability/calibration/step17_power.py --stage curve --d 4 --lam 1 [--sims 100 --n_perm 100]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker process
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import step17_g1b as g  # noqa: E402

SD_STATE = 2.0
STATES, IDS, TASKS = 30, 50, 4
FEATURE_DIM = 64
N_IRRELEVANT = 20
GRID_D = (2, 4, 8)
GRID_LAM = (0.1, 1.0, 10.0)
SHARES = (0.05, 0.2, 0.5)


def simulate(rng, sd_id, sd_inter, share, conf=0.0):
    out = {}
    for t in range(TASKS):
        z = rng.normal(size=(STATES, 2))
        v = rng.normal(size=(IDS, 2))
        a_s = conf * z[:, :1] + rng.normal(0, np.sqrt(SD_STATE ** 2 - conf ** 2), size=(STATES, 1))
        logit = a_s + rng.normal(0, sd_id, size=(1, IDS)) + sd_inter * (z @ v.T) / np.sqrt(2)
        Y = (rng.random((STATES, IDS)) < 1 / (1 + np.exp(-logit))).astype(int)
        A = rng.normal(size=(2, FEATURE_DIM)) / np.sqrt(FEATURE_DIM)
        B = rng.normal(size=(N_IRRELEVANT, FEATURE_DIM)) / np.sqrt(FEATURE_DIM)
        A /= np.sqrt((A ** 2).sum(1, keepdims=True) * 2)               # rows of squared norm 1/2: z A has total variance 1
        B /= np.sqrt((B ** 2).sum(1, keepdims=True) * N_IRRELEVANT)    # rows of squared norm 1/N: n B has total variance 1
        X = np.sqrt(share) * z @ A + np.sqrt(1 - share) * rng.normal(size=(STATES, N_IRRELEVANT)) @ B
        out[t] = (Y, X)
    return out


def one(job):
    seed, d, lam, n_perm, kw = job
    rng = np.random.default_rng(seed)
    r = g.u1_test(simulate(rng, **kw), d, lam, n_perm=n_perm, seed=int(rng.integers(1 << 30)))
    return r["passes"], r["U1"], r["D"]


def rate(pool, rng, sims, d, lam, n_perm, **kw):
    jobs = [(int(rng.integers(1 << 62)), d, lam, n_perm, kw) for _ in range(sims)]
    res = pool.map(one, jobs)
    passes = np.array([r[0] for r in res])
    return passes.mean(), float(np.mean([r[1] for r in res])), float(np.mean([r[2] for r in res]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "level", "curve"])
    ap.add_argument("--d", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--sims", type=int)
    ap.add_argument("--n_perm", type=int)
    ap.add_argument("--seed", type=int, default=2017)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    a = ap.parse_args(argv)
    sims = a.sims or {"select": 50, "level": 300, "curve": 100}[a.stage]
    n_perm = a.n_perm or {"select": 50, "level": 100, "curve": 100}[a.stage]
    if a.stage != "select" and (a.d is None or a.lam is None):
        ap.error("--d and --lam are needed for this stage")
    rng = np.random.default_rng(a.seed)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "noise_reachability"],
                                capture_output=True, text=True).stdout.strip())
    t0 = time.time()
    print(f"step 17 power, stage {a.stage}: {TASKS} tasks x {STATES} states x {IDS} ids, state sd {SD_STATE}, features "
          f"{FEATURE_DIM}-d from a 2-d scene latent and {N_IRRELEVANT} irrelevant directions; {sims} simulations, "
          f"{n_perm} permutations, rule p <= {g.ALPHA}, U1 > 0, D > 0; seed {a.seed}; an estimate, not a result")
    print(f"command: python noise_reachability/calibration/step17_power.py {' '.join(argv or sys.argv[1:])}; "
          f"commit {head}{' with uncommitted noise_reachability changes' if dirty else ''}", flush=True)
    with Pool(a.workers) as pool:
        if a.stage == "select":
            print("pass rate at sd_inter 1.0, id sd 0.5, by (d, lambda) and share " + ", ".join(map(str, SHARES))
                  + " (mean U1, D at share 0.2):")
            best = None
            for d in GRID_D:
                for lam in GRID_LAM:
                    rows = [rate(pool, rng, sims, d, lam, n_perm, sd_id=0.5, sd_inter=1.0, share=s) for s in SHARES]
                    mean = float(np.mean([r[0] for r in rows]))
                    print(f"  d {d}, lambda {lam:g}: " + ", ".join(f"{r[0]:.2f}" for r in rows)
                          + f" (mean {mean:.2f}; U1 {rows[1][1]:+.3f}, D {rows[1][2]:+.3f})", flush=True)
                    if best is None or mean > best[0]:
                        best = (mean, d, lam)
            print(f"highest mean power: d {best[1]}, lambda {best[2]:g} ({best[0]:.2f}); check its level next")
        elif a.stage == "level":
            print(f"pass rate without interaction (level), d {a.d}, lambda {a.lam:g}, id sd 1.0, by share and conf:")
            for share in (0.2, 0.5):
                for conf in (0.0, 2.0):
                    r = rate(pool, rng, sims, a.d, a.lam, n_perm, sd_id=1.0, sd_inter=0.0, share=share, conf=conf)
                    se = np.sqrt(r[0] * (1 - r[0]) / sims)
                    print(f"  share {share}, conf {conf}: {r[0]:.3f} (se {se:.3f}; mean U1 {r[1]:+.4f}, D {r[2]:+.4f})",
                          flush=True)
        else:
            print(f"power, d {a.d}, lambda {a.lam:g}, id sd 0.5, by share and sd_inter 0, 0.5, 1.0, 1.5:")
            for share in SHARES:
                rows = [rate(pool, rng, sims, a.d, a.lam, n_perm, sd_id=0.5, sd_inter=s, share=share)
                        for s in (0.0, 0.5, 1.0, 1.5)]
                print(f"  share {share}: " + ", ".join(f"{r[0]:.2f}" for r in rows), flush=True)
    print(f"({time.time() - t0:.0f} s, {a.workers} workers)")


if __name__ == "__main__":
    main()
