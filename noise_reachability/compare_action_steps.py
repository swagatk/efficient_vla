"""Step 0: compare evaluate_all.py runs at other n_action_steps with the n_action_steps=1 run (README.md).

Each run's per_episode.csv is compared with the baseline restricted to the same (task, seed, init state)
episodes: exact McNemar test, and an exact 95% interval for the paired difference in success rate
(Clopper-Pearson on the share of discordant episodes won by the run). Then the decision rule of README.md
picks the setting for steps 1-4.

    python -S noise_reachability/compare_action_steps.py \
        week6/smolvla_libero_lerobot_eval_kaggle/per_episode.csv 10=<run>/per_episode.csv 50=<run>/per_episode.csv

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline_performance"))
from compare_per_state import compare, format_report, load  # noqa: E402

BASELINE_STEPS = 1


def binom_cdf(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def clopper_pearson(k, n, level=0.95):
    """Exact two-sided interval for a binomial proportion with k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    a = (1 - level) / 2

    def solve(f):  # f is increasing in p on [0, 1]; find f(p) = 0 by bisection
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: (1 - binom_cdf(k - 1, n, p)) - a)  # P(X >= k | p) = a
    upper = 1.0 if k == n else solve(lambda p: a - binom_cdf(k, n, p))            # P(X <= k | p) = a
    return lower, upper


def paired_difference_ci(base_only, run_only, n_episodes, level=0.95):
    """Exact interval for (run successes - base successes) / n_episodes, conditional on the discordant count."""
    d = base_only + run_only
    if d == 0:
        return 0.0, 0.0
    lo, hi = clopper_pearson(run_only, d, level)
    return d / n_episodes * (2 * lo - 1), d / n_episodes * (2 * hi - 1)


def compare_setting(base_eps, run_eps, alpha):
    missing = set(run_eps) - set(base_eps)
    if missing:
        raise ValueError(f"{len(missing)} episodes of the run are not in the baseline, e.g. {sorted(missing)[:3]}")
    base = {k: base_eps[k] for k in run_eps}
    r = compare(base, run_eps)
    lo, hi = paired_difference_ci(r["episodes_a_only"], r["episodes_b_only"], r["n_episodes"])
    r["diff_ci95"] = [lo, hi]
    r["qualifies"] = r["b_successes"] > r["a_successes"] and r["mcnemar_p"] < alpha
    return r


def decide(results):
    """Setting for steps 1-4: the qualifying setting with most successes, ties to the smallest; else the baseline."""
    qualifying = [n for n, r in results.items() if r["qualifies"]]
    if not qualifying:
        return BASELINE_STEPS
    return max(qualifying, key=lambda n: (results[n]["b_successes"], -n))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="per_episode.csv of the n_action_steps=1 run")
    parser.add_argument("runs", nargs="+", help="<n_action_steps>=<per_episode.csv>")
    parser.add_argument("--alpha", type=float, default=None, help="Per-setting alpha (default 0.05 / number of runs)")
    parser.add_argument("--out", type=str, default=None, help="Write the report here, and <out>.json")
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        n, _, path = spec.partition("=")
        runs[int(n)] = path
    alpha = args.alpha if args.alpha is not None else 0.05 / len(runs)
    base_eps = load(args.baseline)
    results = {n: compare_setting(base_eps, load(path), alpha) for n, path in sorted(runs.items())}

    lines = []
    for n, r in results.items():
        lines.append(f"===== n_action_steps={n} against n_action_steps={BASELINE_STEPS} =====")
        lines.append(format_report(r, f"n_action_steps={BASELINE_STEPS} ({args.baseline})",
                                   f"n_action_steps={n} ({runs[n]})").rstrip())
        lo, hi = r["diff_ci95"]
        lines.append(f"paired difference B - A: {100 * (r['b_successes'] - r['a_successes']) / r['n_episodes']:+.1f} "
                     f"points, exact 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}")
        lines.append(f"qualifies (ahead with p < {alpha:g}): {'yes' if r['qualifies'] else 'no'}\n")
    chosen = decide(results)
    lines.append(f"DECISION: n_action_steps={chosen}"
                 + ("" if chosen == BASELINE_STEPS else " (run it on all 10 tasks before adopting it, README.md)"))
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.out:
        Path(args.out).write_text(report)
        Path(args.out + ".json").write_text(json.dumps(
            {"alpha": alpha, "baseline": args.baseline, "runs": runs, "decision": chosen,
             "results": {str(n): {k: v for k, v in r.items() if k != "per_task"} | {"per_task": {str(t): p for t, p in r["per_task"].items()}}
                         for n, r in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
