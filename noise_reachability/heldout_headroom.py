"""Step 1's headroom restated with a held-out choice (paper open item; exploratory, no new episodes).

pass@16 per state is hindsight: it credits a state if any of 16 seeds solved it, which a selector can only realise by
trying seeds on that state. Here the seed for a state is chosen without looking at that state: leave one state out,
pick the seed with the most successes on the task's other states (ties: the average over the tied seeds), and score
it on the held-out state. The mean over states is what choosing a seed per task, from other states, realises. A
permutation null shuffles seed labels within each state (destroying any seed effect shared across states) and
recomputes it.

    python -S noise_reachability/heldout_headroom.py --tasks 0 4 6 7 8 --seeds 0-15 \
        week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv \
        week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv --out <file>
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reachability import parse_range, pass_at_k, pool  # noqa: E402


def leave_one_state_out(per_state):
    """{(task, state): expected success of the seed chosen on the task's other states}."""
    out = {}
    for t, s in per_state:
        others = [v for (tt, ss), v in per_state.items() if tt == t and ss != s]
        if not others:
            raise ValueError(f"task {t} has one state; nothing to choose from")
        totals = [sum(v[j] for v in others) for j in range(len(per_state[t, s]))]
        best = [j for j, x in enumerate(totals) if x == max(totals)]
        out[t, s] = sum(per_state[t, s][j] for j in best) / len(best)
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def summarize(per_state, perms=2000, seed=0):
    n = len(next(iter(per_state.values())))
    chosen = leave_one_state_out(per_state)
    p1 = mean(sum(v) / n for v in per_state.values())
    pn = mean(pass_at_k(n, sum(v), n) for v in per_state.values())
    sel = mean(chosen.values())
    rng = random.Random(seed)
    null = []
    for _ in range(perms):
        shuffled = {st: rng.sample(v, len(v)) for st, v in per_state.items()}
        null.append(mean(leave_one_state_out(shuffled).values()))
    tasks = {}
    for t in sorted({t for t, _ in per_state}):
        mine = [st for st in per_state if st[0] == t]
        tasks[t] = {"pass@1": mean(sum(per_state[st]) / n for st in mine),
                    f"pass@{n}": mean(pass_at_k(n, sum(per_state[st]), n) for st in mine),
                    "held_out_choice": mean(chosen[st] for st in mine)}
    return {"n_seeds": n, "n_states": len(per_state), "pass@1": p1, f"pass@{n}": pn, "held_out_choice": sel,
            "hindsight_headroom": (pn - p1) / (1 - p1), "held_out_headroom": (sel - p1) / (1 - p1),
            "null_mean": mean(null), "null_p": sum(x >= sel for x in null) / perms, "perms": perms,
            "perm_seed": seed, "per_task": tasks}


def format_report(s):
    n = s["n_seeds"]
    lines = [f"{s['n_states']} init states x {n} noise seeds; seed chosen per task on the other states (leave one out)",
             "", f"task | pass@1 | held-out choice | pass@{n} (hindsight)"]
    for t, v in s["per_task"].items():
        lines.append(f"{t:4d} | {v['pass@1']:.3f} | {v['held_out_choice']:.3f} | {v[f'pass@{n}']:.3f}")
    lines += ["", f"pooled: pass@1 {s['pass@1']:.3f}, held-out choice {s['held_out_choice']:.3f}, pass@{n} {s[f'pass@{n}']:.3f}",
              f"headroom: hindsight {s['hindsight_headroom']:.3f}, held-out {s['held_out_headroom']:.3f}",
              f"null (seed labels shuffled within state, {s['perms']} permutations, seed {s['perm_seed']}): "
              f"mean {s['null_mean']:.3f}, one-sided p {s['null_p']:.3f}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--perm_seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    s = summarize(pool(a.files, a.tasks, parse_range(a.seeds)), a.perms, a.perm_seed)
    text = format_report(s)
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out + ".json").write_text(json.dumps(s, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
