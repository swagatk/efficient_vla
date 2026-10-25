"""Step 1: per-state reachability of the base policy over noise seeds (README.md).

Pools evaluate_all.py per_episode.csv files that cover the same tasks and init states with different noise
seeds, and reports per (task, init state) the successes out of n seeds, pass@k (unbiased estimator), the
class reachable (some seed succeeded) or systematic (none did), and the pooled headroom
(pass@n - pass@1) / (1 - pass@1). With --repeat and --original (the same seed run twice), also the share
of episodes whose outcome changed against the 2p(1-p) expected under a fresh noise seed.

    python -S noise_reachability/reachability.py --tasks 0 4 6 7 8 --seeds 0-15 \
        week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv <step1 run>/per_episode.csv \
        --original week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv \
        --repeat <repeat run>/per_episode.csv --repeat_seed 0 --out <file>

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import json
from math import comb
from pathlib import Path

HEADROOM_MIN = 0.5
SYSTEMATIC_MIN = 5


def load(path):
    with open(path) as f:
        return {(int(r["task_id"]), int(r["seed"]), int(r["init_state"])): int(r["success"]) for r in csv.DictReader(f)}


def parse_range(spec):
    """'0-15' or '0,3,5-7' -> sorted list of ints."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def pool(files, tasks, seeds):
    """{(task, init_state): [success per seed in seed order]}; every (task, seed, init state) exactly once."""
    episodes = {}
    for path in files:
        for key, ok in load(path).items():
            if key[0] not in tasks or key[1] not in seeds:
                continue
            if key in episodes:
                raise ValueError(f"episode {key} appears in more than one file")
            episodes[key] = ok
    states = sorted({(t, i) for t, _, i in episodes})
    missing = [(t, s, i) for t, i in states for s in seeds if (t, s, i) not in episodes]
    if missing or sorted({t for t, _ in states}) != sorted(tasks):
        raise ValueError(f"incomplete: {len(missing)} episodes missing, e.g. {missing[:3]}; tasks {sorted({t for t, _ in states})}")
    return {st: [episodes[(st[0], s, st[1])] for s in seeds] for st in states}


def pass_at_k(n, c, k):
    """Unbiased estimate of P(at least one of k samples succeeds) from c successes in n."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(per_state):
    n = len(next(iter(per_state.values())))
    counts = {st: sum(v) for st, v in per_state.items()}
    curve = [sum(pass_at_k(n, c, k) for c in counts.values()) / len(counts) for k in range(1, n + 1)]
    p1, pn = curve[0], curve[-1]
    systematic = sorted(st for st, c in counts.items() if c == 0)
    per_task = {}
    for t in sorted({t for t, _ in per_state}):
        cs = [c for (tt, _), c in counts.items() if tt == t]
        per_task[t] = {"successes": sum(cs), "episodes": n * len(cs), "states": len(cs),
                       "systematic": [i for tt, i in systematic if tt == t],
                       "pass@1": sum(pass_at_k(n, c, 1) for c in cs) / len(cs),
                       f"pass@{n}": sum(pass_at_k(n, c, n) for c in cs) / len(cs)}
    headroom = (pn - p1) / (1 - p1) if p1 < 1 else 0.0
    return {"n_seeds": n, "n_states": len(counts), "pass_at_k": curve, "headroom": headroom,
            "systematic_states": [list(st) for st in systematic],
            "counts": {f"{t},{i}": c for (t, i), c in sorted(counts.items())}, "per_task": per_task,
            "hypothesis": {"headroom_ok": headroom >= HEADROOM_MIN, "systematic_ok": len(systematic) >= SYSTEMATIC_MIN}}


def repeat_stats(original, repeat, per_state, seed):
    """Outcome changes between two runs of one seed, against 2p(1-p) under a fresh noise seed."""
    keys = sorted((t, seed, i) for t, i in per_state)
    missing = [k for k in keys if k not in original or k not in repeat]
    if missing:
        raise ValueError(f"repeat incomplete: {len(missing)} episodes missing, e.g. {missing[:3]}")
    changed = sum(original[k] != repeat[k] for k in keys)
    expected = sum(2 * p * (1 - p) for p in (sum(v) / len(v) for v in per_state.values()))
    return {"episodes": len(keys), "changed": changed, "expected_changed_fresh_seed": expected,
            "ratio": changed / expected if expected else None}


def format_report(s, rep=None):
    n = s["n_seeds"]
    lines = [f"{s['n_states']} init states x {n} noise seeds", "",
             f"task | successes | pass@1 | pass@{n} | systematic init states (0 of {n})"]
    for t, p in s["per_task"].items():
        lines.append(f"{t:4d} | {p['successes']:3d}/{p['episodes']} | {p['pass@1']:.3f} | {p[f'pass@{n}']:.3f} | {p['systematic']}")
    curve = s["pass_at_k"]
    lines += ["", "pooled pass@k: " + ", ".join(f"k={k}: {curve[k - 1]:.3f}" for k in (1, 2, 4, 8, n) if k <= n),
              f"headroom (pass@{n} - pass@1) / (1 - pass@1) = {s['headroom']:.3f} "
              f"(hypothesis >= {HEADROOM_MIN}: {'yes' if s['hypothesis']['headroom_ok'] else 'no'})",
              f"systematic states: {len(s['systematic_states'])} of {s['n_states']} "
              f"(hypothesis >= {SYSTEMATIC_MIN}: {'yes' if s['hypothesis']['systematic_ok'] else 'no'})"]
    if rep:
        lines.append(f"repeat of one seed: {rep['changed']} of {rep['episodes']} episodes changed outcome; "
                     f"{rep['expected_changed_fresh_seed']:.1f} expected under a fresh noise seed (ratio {rep['ratio']:.2f})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="per_episode.csv files with the seeds to pool")
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=parse_range, required=True, help="e.g. 0-15")
    parser.add_argument("--original", help="per_episode.csv holding the original run of --repeat_seed")
    parser.add_argument("--repeat", help="per_episode.csv of the repeat run of --repeat_seed")
    parser.add_argument("--repeat_seed", type=int, default=0)
    parser.add_argument("--out", help="Write the report here, and <out>.json")
    args = parser.parse_args()

    per_state = pool(args.files, args.tasks, args.seeds)
    s = summarize(per_state)
    rep = repeat_stats(load(args.original), load(args.repeat), per_state, args.repeat_seed) if args.repeat else None
    report = format_report(s, rep)
    print(report, end="")
    if args.out:
        Path(args.out).write_text(report)
        Path(args.out + ".json").write_text(json.dumps(dict(s, repeat=rep, files=args.files), indent=2))


if __name__ == "__main__":
    main()
