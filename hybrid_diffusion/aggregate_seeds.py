"""Aggregate base vs residual evaluation results over seeds written by run_task3_residual_3seed.sh.

Reads eval_results.json for each seed and reports per-seed success, mean +- std over seeds, and a
paired comparison (same init state and sampling-noise seed for both policies).
"""
import argparse
import json
import os
from math import comb
from statistics import mean, stdev


def mcnemar_exact_p(base_only: int, residual_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts."""
    n = base_only + residual_only
    if n == 0:
        return 1.0
    k = min(base_only, residual_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired_counts(base: dict, residual: dict) -> dict:
    """Outcome table for two {init_state_seed: success} maps over the same init states."""
    if set(base) != set(residual):
        raise ValueError(f"init states differ: base has {len(base)}, residual has {len(residual)}")
    return {
        "n": len(base),
        "base_success": sum(base.values()),
        "residual_success": sum(residual.values()),
        "base_only": sum(base[s] and not residual[s] for s in base),
        "residual_only": sum(residual[s] and not base[s] for s in base),
    }


def summarise(per_seed: dict) -> dict:
    """Pool per-seed paired_counts() tables into rates, mean/std over seeds and a pooled test."""
    seeds = sorted(per_seed)
    base_rates = [per_seed[s]["base_success"] / per_seed[s]["n"] for s in seeds]
    res_rates = [per_seed[s]["residual_success"] / per_seed[s]["n"] for s in seeds]
    base_only = sum(per_seed[s]["base_only"] for s in seeds)
    residual_only = sum(per_seed[s]["residual_only"] for s in seeds)
    return {
        "seeds": seeds,
        "episodes_per_seed": [per_seed[s]["n"] for s in seeds],
        "base_rate_mean": mean(base_rates),
        "base_rate_std": stdev(base_rates) if len(seeds) > 1 else 0.0,
        "residual_rate_mean": mean(res_rates),
        "residual_rate_std": stdev(res_rates) if len(seeds) > 1 else 0.0,
        "pooled_base_success": sum(per_seed[s]["base_success"] for s in seeds),
        "pooled_residual_success": sum(per_seed[s]["residual_success"] for s in seeds),
        "pooled_episodes": sum(per_seed[s]["n"] for s in seeds),
        "pooled_base_only": base_only,
        "pooled_residual_only": residual_only,
        "pooled_mcnemar_p": mcnemar_exact_p(base_only, residual_only),
    }


def load_successes(results_json: str, noise_seed: int, expected_episodes: int) -> dict:
    with open(results_json) as f:
        data = json.load(f)
    if int(data.get("eval_noise_seed", 0)) != noise_seed:
        raise ValueError(f"{results_json}: eval_noise_seed {data.get('eval_noise_seed', 0)} != {noise_seed}")
    out = {int(e["seed"]): bool(e["success"]) for e in data["episodes"]}
    if len(out) != expected_episodes:
        raise ValueError(f"{results_json}: {len(out)} episodes, expected {expected_episodes} (run incomplete?)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=str, default="1.0")
    parser.add_argument("--run_prefix", type=str, default="h512_l6_n0.5_e300")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    per_seed = {}
    for seed in args.seeds:
        tag = "" if seed == 0 else f"_s{seed}"
        base = load_successes(os.path.join(args.out_root, f"eval_base{tag}", "eval_results.json"), seed, args.episodes)
        res_dir = f"eval_residual_{args.run_prefix}{tag}_alpha{args.alpha}"
        res = load_successes(os.path.join(args.out_root, res_dir, "eval_results.json"), seed, args.episodes)
        per_seed[seed] = paired_counts(base, res)
        c = per_seed[seed]
        print(f"seed {seed}: base {c['base_success']}/{c['n']}  residual {c['residual_success']}/{c['n']}  "
              f"base-only {c['base_only']}  residual-only {c['residual_only']}")

    summary = summarise(per_seed)
    summary["per_seed"] = {str(s): per_seed[s] for s in sorted(per_seed)}
    print(f"base     {100 * summary['base_rate_mean']:.1f} +- {100 * summary['base_rate_std']:.1f} %  "
          f"({summary['pooled_base_success']}/{summary['pooled_episodes']})")
    print(f"residual {100 * summary['residual_rate_mean']:.1f} +- {100 * summary['residual_rate_std']:.1f} %  "
          f"({summary['pooled_residual_success']}/{summary['pooled_episodes']})")
    print(f"pooled discordant pairs: base-only {summary['pooled_base_only']}, residual-only "
          f"{summary['pooled_residual_only']}, exact McNemar p = {summary['pooled_mcnemar_p']:.4f}")

    name = "seed_summary_s" + "-".join(str(s) for s in sorted(per_seed)) + f"_alpha{args.alpha}.json"
    with open(os.path.join(args.out_root, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {os.path.join(args.out_root, name)}")


if __name__ == "__main__":
    main()
