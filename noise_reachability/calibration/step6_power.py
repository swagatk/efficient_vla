"""Model-based operating characteristics of step 6's rule (README.md). Estimates, not experiment results.

State rates are drawn from Beta(k + 0.5, 16 - k + 0.5) on the real seeds 0-15 counts, then binomial episodes for the
base and B. The D and p arithmetic is checked here against analyze_step6.uneven_gain, then vectorised.

    python noise_reachability/calibration/step6_power.py [results root]
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step6 as a  # noqa: E402
from reachability import pool  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
W = DATA / "week6"
labels = {k: sum(v) for k, v in pool([str(W / "noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv"),
                                      str(W / "noise_reachability_step1_kaggle/nas10_step1/per_episode.csv")],
                                     list(a.ALL_TASKS), list(a.LABEL_SEEDS)).items()}
grp = a.groups(labels, a.NEW_TASKS)
tasks = sorted(grp)
NSIM = 20000


def stat(g):
    """D and one-sided p per simulation; index 0 of each task's array is its systematic state."""
    gaps = [(x.sum(1, keepdims=True) - x) / (x.shape[1] - 1) - x for x in (g[t] for t in tasks)]
    A, B, C = gaps
    every = (A[:, :, None, None] + B[:, None, :, None] + C[:, None, None, :]) / 3
    obs = every[:, 0, 0, 0]
    return obs, (every >= obs[:, None, None, None] - a.TIE).reshape(len(obs), -1).mean(1)


def check_against_the_analysis():
    rng = np.random.default_rng(0)
    for _ in range(100):
        g = {(t, i): float(rng.integers(-16, 17)) / 16 for t in grp for i in grp[t][0] + grp[t][1]}
        arrays = {t: np.array([[g[(t, i)] for i in grp[t][0] + grp[t][1]]]) for t in tasks}
        D, p = stat(arrays)
        r = a.uneven_gain(g, grp)
        assert abs(D[0] - r["D"]) < 1e-9 and abs(p[0] - r["p"]) < 1e-9


def simulate(change, seed=1):
    r = np.random.default_rng(seed)
    base, b = {}, {}
    for t in tasks:
        k = np.array([labels[(t, i)] for i in grp[t][0] + grp[t][1]], float)
        p = r.beta(k + 0.5, 16 - k + 0.5, size=(NSIM, len(k)))
        base[t], b[t] = r.binomial(16, p), r.binomial(16, change(p))
    gains = {t: (b[t] - base[t]) / 16 for t in tasks}
    D, p = stat(gains)
    sys_hits = np.stack([base[t][:, 0] for t in tasks], 1)
    labels_ok = (sys_hits.sum(1) <= a.MAX_LABEL_HITS) & (sys_hits.max(1) <= a.MAX_STATE_HITS)
    low = np.concatenate([gains[t][:, 1:] for t in tasks], 1).mean(1)
    informative = low >= a.MIN_GAIN - a.TIE
    uneven = (D >= a.MIN_GAIN - a.TIE) & (p < a.ALPHA)
    both = labels_ok & informative
    return {"labels hold": labels_ok.mean(), "informative": informative.mean(), "mean low gain": low.mean(),
            "uneven given both": (both & uneven).sum() / max(1, both.sum())}


logit = lambda p: np.log(p / (1 - p))  # noqa: E731
expit = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def lift_low(d):
    def change(p):
        q = np.clip(p + d, 0, 1)
        q[:, 0] = p[:, 0]
        return q
    return change


check_against_the_analysis()
print("groups", grp, "simulations per row", NSIM)
for name, change in [("B equals base", lambda p: p)] + \
        [(f"uniform log-odds lift +{s}, no classes", lambda p, s=s: expit(logit(p) + s)) for s in (0.5, 1, 1.5, 2)] + \
        [(f"class effect: low +{d}, systematic +0", lift_low(d)) for d in (0.1, 0.15, 0.2, 0.3, 0.4)]:
    print(f"{name}:", {k: round(float(v), 3) for k, v in simulate(change).items()})
print("binomial, pooled labels rule only, P(hits <= 4 of 48):",
      {q: round(sum(math.comb(48, h) * q ** h * (1 - q) ** (48 - h) for h in range(5)), 3) for q in (0.05, 0.10, 0.15)})
