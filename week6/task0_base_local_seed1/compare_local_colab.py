"""Per-init-state comparison of the local task-0 base runs (noise seeds 0, 1) with the Colab runs (seeds 1-3).

    /home/swagat/miniconda3/envs/lerobot_env/bin/python week6/task0_base_local_seed1/compare_local_colab.py \
        > week6/task0_base_local_seed1/comparison.txt
"""
import json
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

WEEK6 = Path(__file__).resolve().parent.parent


def load(path):
    """eval_results.json -> {init state: success}."""
    return {e["seed"]: bool(e["success"]) for e in json.load(open(path))["episodes"]}


def mcnemar(a, b, name_a, name_b):
    only_a = sum(a[k] and not b[k] for k in a)
    only_b = sum(b[k] and not a[k] for k in a)
    both = sum(a[k] and b[k] for k in a)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    print(f"{name_a} vs {name_b}: only {name_a} {only_a}, only {name_b} {only_b}, both {both}, exact McNemar p = {p:.3f}")


colab = {s: load(WEEK6 / f"task0_base_noise_seeds_colab/eval_base_s{s}/eval_results.json") for s in (1, 2, 3)}
local = {0: load(WEEK6 / "task0_base_local_seed0/eval_results.json"),
         1: load(WEEK6 / "task0_base_local_seed1/eval_results.json")}
runs = {f"colab s{s}": r for s, r in colab.items()} | {f"local s{s}": r for s, r in local.items()}
assert all(sorted(r) == list(range(50)) for r in runs.values())

print("Successes per run (init states solved):")
for name, r in runs.items():
    print(f"  {name}: {sum(r.values())}/50  {sorted(k for k, v in r.items() if v)}")

print("\nPlanned check (same noise seed, paired by init state):")
mcnemar(local[1], colab[1], "local s1", "colab s1")
print("For scale:")
mcnemar(colab[2], colab[1], "colab s2", "colab s1")
mcnemar(local[1], local[0], "local s1", "local s0")
mcnemar(local[0], colab[1], "local s0", "colab s1")

never = [k for k in range(50) if not any(colab[s][k] for s in colab)]
print(f"\nInit states Colab never solved (0/96 by selection): {len(never)}")
for s, r in local.items():
    hit = [k for k in never if r[k]]
    p = fisher_exact([[len(hit), len(never) - len(hit)], [0, 96]])[1]
    print(f"  local s{s} solves {len(hit)}: {hit}  (Fisher vs 0/96 = {p:.2g}; biased, the states were selected on the Colab failures)")

n_local = sum(sum(r.values()) for r in local.values())
n_colab = sum(sum(r.values()) for r in colab.values())
print("\nSuccess rates:")
print(f"  local {n_local}/100 vs colab {n_colab}/150: Fisher p = {fisher_exact([[n_local, 100 - n_local], [n_colab, 150 - n_colab]])[1]:.3f}")
n1 = sum(local[1].values())
print(f"  local s1 {n1}/50 vs colab {n_colab}/150: Fisher p = {fisher_exact([[n1, 50 - n1], [n_colab, 150 - n_colab]])[1]:.3f}")
