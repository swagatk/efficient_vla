"""Do base-policy failures repeat on the same init states, or move with the sampling noise?

Reads eval_base[_s<seed>]/eval_results.json for several noise seeds of one task (written by
run_base_noise_seeds.sh) and compares how often an init state is solved by at least one seed with
what independent failures would give. See README.md for the hypothesis and decision rule.
"""
import argparse
import json
import os
import random
from statistics import mean

from aggregate_seeds import load_successes


def failure_counts(per_seed: dict) -> dict:
    """{init_state: number of seeds that failed on it} from {seed: {init_state: success}}."""
    seeds = sorted(per_seed)
    states = set(per_seed[seeds[0]])
    for s in seeds:
        if set(per_seed[s]) != states:
            raise ValueError(f"init states differ between seed {seeds[0]} and seed {s}")
    return {st: sum(not per_seed[s][st] for s in seeds) for st in sorted(states)}


def independent_histogram(fail_rates: list) -> list:
    """P(an init state fails in exactly k seeds), k = 0..len(fail_rates), if seeds fail independently."""
    hist = [1.0]
    for p in fail_rates:
        hist = [(hist[k] * (1 - p) if k < len(hist) else 0.0) + (hist[k - 1] * p if k > 0 else 0.0)
                for k in range(len(hist) + 1)]
    return hist


def permutation_p(per_seed: dict, n_perm: int, rng_seed: int) -> float:
    """One-sided p-value that failures are more concentrated on the same init states than chance.

    Shuffles which init states each seed failed on, keeping every seed's failure count, and compares
    the sum of squared per-state failure counts with the observed one.
    """
    seeds = sorted(per_seed)
    states = sorted(per_seed[seeds[0]])
    observed = sum(c * c for c in failure_counts(per_seed).values())
    n_fail = [sum(not per_seed[s][st] for st in states) for s in seeds]
    rng = random.Random(rng_seed)
    hits = 0
    for _ in range(n_perm):
        counts = dict.fromkeys(states, 0)
        for k in n_fail:
            for st in rng.sample(states, k):
                counts[st] += 1
        hits += sum(c * c for c in counts.values()) >= observed
    return (hits + 1) / (n_perm + 1)


def overlap_summary(per_seed: dict, random_min: float, systematic_max: float) -> dict:
    """Success of the best-of-seeds oracle against single seeds and against independent failures.

    randomness_index = (oracle - single) / (independent oracle - single): about 1 when failures move
    with the noise seed, about 0 when the same init states fail under every seed.
    """
    seeds = sorted(per_seed)
    counts = failure_counts(per_seed)
    n = len(counts)
    rates = [sum(per_seed[s].values()) / n for s in seeds]
    single = mean(rates)
    expected = independent_histogram([1 - r for r in rates])
    oracle = sum(c < len(seeds) for c in counts.values()) / n
    oracle_independent = 1 - expected[-1]
    possible = oracle_independent - single
    index = (oracle - single) / possible if possible > 1e-12 else None
    if index is None:
        verdict = "undetermined"
    elif index >= random_min:
        verdict = "mostly_random"
    elif index <= systematic_max:
        verdict = "mostly_systematic"
    else:
        verdict = "mixed"
    return {
        "seeds": seeds,
        "init_states": n,
        "success_per_seed": {str(s): sum(per_seed[s].values()) for s in seeds},
        "single_seed_rate_mean": single,
        "states_failing_in_k_seeds": [sum(c == k for c in counts.values()) for k in range(len(seeds) + 1)],
        "expected_if_independent": [n * p for p in expected],
        "always_failing_states": [st for st, c in counts.items() if c == len(seeds)],
        "oracle_rate": oracle,
        "oracle_rate_if_independent": oracle_independent,
        "randomness_index": index,
        "random_min": random_min,
        "systematic_max": systematic_max,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--random_min", type=float, default=0.6,
                        help="randomness index at or above this: failures mostly move with the noise seed")
    parser.add_argument("--systematic_max", type=float, default=0.3,
                        help="randomness index at or below this: failures mostly repeat on the same init states")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--perm_seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.seeds) < 2:
        parser.error("need at least two seeds to compare failures across seeds")

    per_seed = {}
    for seed in args.seeds:
        tag = "" if seed == 0 else f"_s{seed}"
        per_seed[seed] = load_successes(os.path.join(args.out_root, f"eval_base{tag}", "eval_results.json"), seed, args.episodes)

    summary = overlap_summary(per_seed, args.random_min, args.systematic_max)
    summary["permutation_p"] = permutation_p(per_seed, args.n_perm, args.perm_seed)
    n = summary["init_states"]
    for seed in summary["seeds"]:
        print(f"seed {seed}: base {summary['success_per_seed'][str(seed)]}/{n}")
    print(f"single seed  {100 * summary['single_seed_rate_mean']:.1f} %")
    print("init states failing in k seeds (observed | expected if independent):")
    for k, (obs, exp) in enumerate(zip(summary["states_failing_in_k_seeds"], summary["expected_if_independent"])):
        print(f"  k={k}: {obs:3d} | {exp:5.1f}")
    print(f"always failing init states: {summary['always_failing_states']}")
    print(f"best-of-{len(summary['seeds'])} oracle  {100 * summary['oracle_rate']:.1f} %  "
          f"(independent failures would give {100 * summary['oracle_rate_if_independent']:.1f} %)")
    index = summary["randomness_index"]
    print(f"randomness index {'n/a' if index is None else f'{index:.2f}'}  "
          f"(>= {args.random_min} mostly random, <= {args.systematic_max} mostly systematic)")
    print(f"permutation test for repeated failures: p = {summary['permutation_p']:.4f}")
    print(f"verdict: {summary['verdict']}")

    name = "failure_overlap_s" + "-".join(str(s) for s in summary["seeds"]) + ".json"
    with open(os.path.join(args.out_root, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {os.path.join(args.out_root, name)}")


if __name__ == "__main__":
    main()
