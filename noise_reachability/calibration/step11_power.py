"""Power of step 11's tests (README.md, step 11), exact; an estimate, not a result.

The repeat arm is step 9's, already run and fixed: 23 of 400 on the eight states, 10 of 300 without task 7. Only the
sampled arm is random: X7 ~ Binomial(100, p) on task 7's two states and X ~ Binomial(300, p) on the other six, at one
rate p (as if sampling were equally hard everywhere). Test A uses X7 + X, test B uses X; Holm over the two at 0.05,
with analyze_step11's functions.
    python noise_reachability/calibration/step11_power.py
"""
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_step7 import fisher_greater  # noqa: E402
from analyze_step8 import holm  # noqa: E402

REPEAT_A, N_A = 23, 400
REPEAT_B, N_B = 10, 300
N_7 = N_A - N_B


def pmf(k, n, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def power(p, kmax=40):
    out = {"A": 0.0, "B": 0.0, "both": 0.0}
    for x in range(kmax + 1):
        px = pmf(x, N_B, p)
        for x7 in range(kmax + 1):
            w = px * pmf(x7, N_7, p)
            passed = holm({"test A": fisher_greater(REPEAT_A, x + x7, N_A), "test B": fisher_greater(REPEAT_B, x, N_B)})
            out["A"] += w * passed["test A"]
            out["B"] += w * passed["test B"]
            out["both"] += w * (passed["test A"] and passed["test B"])
    return out


def main():
    print(f"repeat fixed at step 9's {REPEAT_A}/{N_A} (A) and {REPEAT_B}/{N_B} (B); sampled at rate p on every state")
    for p in (0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.0575):
        r = power(p)
        print(f"sampled {p:.2%}: P(A passes) {r['A']:.3f}, P(B passes) {r['B']:.3f}, P(both) {r['both']:.3f}")


if __name__ == "__main__":
    main()
