"""Power of step 12's test 1 (README.md, step 12); a Monte Carlo estimate, not a result.

Per searched state, round 1's adapter succeeds at its step-10 rate p (N0-N2 pooled, from the committed report); round 2
at min(p + d, 1). Episodes are drawn independently for the two policies (on the same scene they are positively
correlated in practice, which gives fewer discordant pairs for the same gain and so more power: this is the cautious
side). Per seed: 16 episodes per task-0 state and 32 per state of tasks 6 and 7, exact one-sided McNemar per seed,
test 1 passes with p < 0.05 in at least 2 of 3 seeds, as analyze_step12 reads it.

    python noise_reachability/calibration/step12_power.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
from analyze_step10 import mcnemar_greater  # noqa: E402
from analyze_step12 import SEARCHED, STEP10, episodes_per_seed  # noqa: E402


def step10_rates(report: dict) -> dict:
    """{(task, state): N0-N2 pooled success rate} on the searched states."""
    out = {}
    for s in report["states"]:
        if s["state"] in SEARCHED.get(s["task"], ()):
            out[(s["task"], s["state"])] = sum(s[f"N{i}"][0] for i in range(3)) / (3 * s["episodes"])
    return out


def power(rates: dict, d: float, sims: int = 4000, seed: int = 0) -> float:
    rng = random.Random(seed)
    passed = 0
    for _ in range(sims):
        wins = 0
        for _ in range(3):
            b = c = 0
            for (t, s), p in rates.items():
                q = min(p + d, 1.0)
                for _ in range(episodes_per_seed(t)):
                    r, n = rng.random() < q, rng.random() < p
                    b += r and not n
                    c += n and not r
            wins += mcnemar_greater(b, c) < 0.05
        passed += wins >= 2
    return passed / sims


def main():
    rates = step10_rates(json.loads((STEP10 / "step10_report.txt.json").read_text()))
    n = sum(episodes_per_seed(t) for t, _ in rates)
    mean = sum(p * episodes_per_seed(t) for (t, _), p in rates.items()) / n
    print(f"searched states and round-1 rates (step 10, N0-N2 pooled): "
          + ", ".join(f"{t}/{s} {p:.3f}" for (t, s), p in sorted(rates.items())))
    print(f"{n} paired episodes per seed; round 1 at {mean:.3f} on them")
    for d in (0.0, 0.03, 0.05, 0.10, 0.15):
        print(f"round 2 at round 1 + {d:.2f} on every searched state: P(test 1 passes) = {power(rates, d):.3f}")


if __name__ == "__main__":
    main()
