"""Step 6 of README.md: do step 1's labels and arm B's uneven gain hold beyond task 0? Report and reading.

Groups come from the base on seeds 0-15 (step 1's files). The base and arm B on seeds 16-31 come from step 6's merged
per_episode files (tasks 4, 6, 7, 8) and from step 4's files (task 0). A state's gain is (B - base) / 16 on seeds 16-31.
The primary test uses tasks 4, 6, 8 only, because task 0 helped form the claim.

    python -S noise_reachability/analyze_step6.py --labels <step 0 per_episode.csv> <step 1 per_episode.csv> \
        --base <csv> --b <csv> --task0_base <csv> --task0_b <csv> --out <report.txt>

With --harness_only, needs only --labels and --base, prints the harness and exits 1 if a task fails it.

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step4 import paired  # noqa: E402
from compare_action_steps import clopper_pearson  # noqa: E402
from reachability import pool  # noqa: E402

NEW_TASKS = (4, 6, 7, 8)
ALL_TASKS = (0, 4, 6, 7, 8)
INIT_STATES = tuple(range(10))
LABEL_SEEDS = tuple(range(16))
EVAL_SEEDS = tuple(range(16, 32))
LOW_MAX = 8                    # low reachable: 1 to 8 of 16 on seeds 0-15
HARNESS_TOLERANCE = 18         # of 160, per task
MAX_LABEL_HITS = 4             # of the systematic states' episodes on seeds 16-31
MAX_STATE_HITS = 2             # of 16, on any one systematic state
MIN_GAIN = 0.10                # the low reachable states' mean gain, and D
ALPHA = 0.05
BOOTSTRAP_REPS, BOOTSTRAP_SEED = 10000, 0
TIE = 1e-12                    # gains are multiples of 1/16 averaged, so floats can differ in the last bits


def load(path, tasks):
    """{(task, seed, init_state): success}; refuses other tasks and repeated episodes."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (int(r["task_id"]), int(r["seed"]), int(r["init_state"]))
            if k[0] not in tasks:
                raise ValueError(f"{path}: episode of task {k[0]}, expected {tasks}")
            if k in out:
                raise ValueError(f"{path}: episode {k} appears twice")
            out[k] = int(r["success"])
    return out


def check_complete(eps, tasks, name):
    want = {(t, s, i) for t in tasks for s in EVAL_SEEDS for i in INIT_STATES}
    if set(eps) != want:
        raise ValueError(f"{name}: {len(set(eps) & want)} of {len(want)} episodes and {len(set(eps) - want)} others: "
                         "finish the run before reporting")


def state_counts(eps):
    """{(task, init_state): successes}."""
    out = {}
    for (t, _, i), ok in eps.items():
        out[(t, i)] = out.get((t, i), 0) + ok
    return out


def groups(labels, tasks):
    """{task: (systematic states, low reachable states)}, tasks without both left out."""
    out = {}
    for t in tasks:
        sys_states = [i for i in INIT_STATES if labels[(t, i)] == 0]
        low = [i for i in INIT_STATES if 1 <= labels[(t, i)] <= LOW_MAX]
        if sys_states and low:
            out[t] = (sys_states, low)
    return out


def gap(gain, chosen, states):
    """Mean gain of the other states minus mean gain of the chosen ones."""
    rest = [i for i in states if i not in chosen]
    return sum(gain[i] for i in rest) / len(rest) - sum(gain[i] for i in chosen) / len(chosen)


def uneven_gain(gains, grp):
    """D and its one-sided exact stratified permutation p. ``gains``: {(task, state): gain}."""
    tasks = sorted(grp)
    per_task = []
    for t in tasks:
        sys_states, low = grp[t]
        states = sorted(sys_states + low)
        gain = {i: gains[(t, i)] for i in states}
        per_task.append([gap(gain, c, states) for c in combinations(states, len(sys_states))])
    observed = sum(gap({i: gains[(t, i)] for i in grp[t][0] + grp[t][1]}, grp[t][0], grp[t][0] + grp[t][1])
                   for t in tasks) / len(tasks)
    total = at_least = 0
    for values in product(*per_task):
        total += 1
        at_least += sum(values) / len(tasks) >= observed - TIE
    return {"D": observed, "p": at_least / total, "assignments": total}


def task_gaps(gains, grp):
    """{task: mean gain of the low reachable states minus mean gain of the systematic ones}."""
    return {t: gap({i: gains[(t, i)] for i in grp[t][0] + grp[t][1]}, grp[t][0], grp[t][0] + grp[t][1]) for t in sorted(grp)}


def bootstrap_interval(base, b, grp, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% percentile interval of D, resampling seeds within each state, paired. States stay fixed, so it shows
    episode noise only."""
    rng = random.Random(seed)
    pairs = {(t, i): [b[(t, s, i)] - base[(t, s, i)] for s in EVAL_SEEDS] for t in grp for i in grp[t][0] + grp[t][1]}
    n = len(EVAL_SEEDS)
    values = []
    for _ in range(reps):
        gains = {k: sum(rng.choices(v, k=n)) / n for k, v in pairs.items()}
        values.append(sum(task_gaps(gains, grp).values()) / len(grp))
    values.sort()
    return [values[int(0.025 * reps)], values[int(0.975 * reps) - 1]]


def logit_gain(b_count, base_count, n=len(EVAL_SEEDS)):
    """Gain on the log-odds scale, with 0.5 added to each cell, so a floor does not shrink it."""
    logit = lambda c: math.log((c + 0.5) / (n - c + 0.5))  # noqa: E731
    return logit(b_count) - logit(base_count)


def harness(base_counts, labels, tasks):
    """Per task: base on seeds 16-31 against seeds 0-15, and how the states line up (not part of the gate)."""
    out = {}
    for t in tasks:
        now = sum(base_counts[(t, i)] for i in INIT_STATES)
        then = sum(labels[(t, i)] for i in INIT_STATES)
        out[t] = {"seeds_16_31": now, "seeds_0_15": then, "gap": now - then, "ok": abs(now - then) <= HARNESS_TOLERANCE,
                  "state_spearman": spearman([labels[(t, i)] for i in INIT_STATES], [base_counts[(t, i)] for i in INIT_STATES])}
    return out


def ranks(values):
    """Average ranks, from 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    """Spearman correlation, or None when either side is constant."""
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx)
    sy = sum((b - my) ** 2 for b in ry)
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) ** 0.5


def reading(labels_hold, informative, uneven):
    if not labels_hold:
        return "labels fail: the two classes do not carry beyond task 0; report per-state rates, not classes"
    if not informative:
        return "not informative: B's gain on the low reachable states was too small to compare states"
    if uneven:
        return ("labels hold, uneven gain holds: gain is smaller on the systematic states on tasks not used to form the "
                "claim, consistent with a class or a graded response, with one B checkpoint")
    return "labels hold, uneven gain not replicated at this size: task 0's pattern is not shown beyond task 0"


def analyse(labels, base, b, task0_base, task0_b):
    """``labels``: {(task, state): successes on seeds 0-15}. The others: episode dicts on seeds 16-31."""
    check_complete(base, NEW_TASKS, "base")
    check_complete(b, NEW_TASKS, "B")
    check_complete(task0_base, (0,), "task 0 base")
    check_complete(task0_b, (0,), "task 0 B")
    all_base, all_b = {**base, **task0_base}, {**b, **task0_b}
    base_counts, b_counts = state_counts(all_base), state_counts(all_b)
    gains = {k: (b_counts[k] - base_counts[k]) / len(EVAL_SEEDS) for k in base_counts}

    check = harness(base_counts, labels, NEW_TASKS)
    grp = groups(labels, NEW_TASKS)
    sys_states = [(t, i) for t in grp for i in grp[t][0]]
    low_states = [(t, i) for t in grp for i in grp[t][1]]
    label_hits = sum(base_counts[k] for k in sys_states)
    low_gain = sum(gains[k] for k in low_states) / len(low_states)
    primary = dict(uneven_gain(gains, grp), per_task={str(t): v for t, v in task_gaps(gains, grp).items()},
                   ci95=bootstrap_interval(all_base, all_b, grp))
    logit_gains = {k: logit_gain(b_counts[k], base_counts[k]) for k in base_counts}
    labels_hold = label_hits <= MAX_LABEL_HITS and max(base_counts[k] for k in sys_states) <= MAX_STATE_HITS
    informative = low_gain >= MIN_GAIN - TIE
    uneven = primary["D"] >= MIN_GAIN - TIE and primary["p"] < ALPHA
    states = [(t, i) for t in ALL_TASKS for i in INIT_STATES]
    return {
        "harness": {str(t): v for t, v in check.items()},
        "harness_ok": all(v["ok"] for v in check.values()),
        "systematic_states": sys_states, "low_reachable_states": low_states,
        "label_hits": label_hits, "label_episodes": len(sys_states) * len(EVAL_SEEDS), "labels_hold": labels_hold,
        "label_rate_ci95": list(clopper_pearson(label_hits, len(sys_states) * len(EVAL_SEEDS))),
        "label_hits_per_state": {f"{t}/{i}": base_counts[(t, i)] for t, i in sys_states},
        "low_gain": low_gain, "informative": informative,
        "primary": primary, "uneven": uneven,
        "reading": reading(labels_hold, informative, uneven),
        "secondary": {
            "with_task_0": uneven_gain(gains, groups(labels, ALL_TASKS)),
            "log_odds": uneven_gain(logit_gains, grp),
            "spearman_rate_gain": spearman([labels[k] / len(LABEL_SEEDS) for k in states], [gains[k] for k in states]),
            "per_task": {str(t): paired({k: v for k, v in all_base.items() if k[0] == t},
                                        {k: v for k, v in all_b.items() if k[0] == t}) for t in ALL_TASKS},
        },
        "per_state": [{"task": t, "init_state": i, "seeds_0_15": labels[(t, i)], "base": base_counts[(t, i)],
                       "b": b_counts[(t, i)], "gain": gains[(t, i)]} for t, i in states],
    }


def harness_line(t, v):
    return (f"harness task {t}: base {v['seeds_16_31']}/160 against {v['seeds_0_15']} on seeds 0-15 (gap {v['gap']:+d}, "
            f"allowed {HARNESS_TOLERANCE}): {'ok' if v['ok'] else 'FAILED'}; state Spearman {v['state_spearman']}")


def harness_gate(labels, base):
    """The harness on a finished base, before B runs: (lines, passed)."""
    check_complete(base, NEW_TASKS, "base")
    check = harness(state_counts(base), labels, NEW_TASKS)
    return ([harness_line(t, v) for t, v in check.items()],
            all(v["ok"] for v in check.values()))


def format_report(r):
    lines = ["step 6: labels and arm B's uneven gain on tasks 4, 6, 7, 8 (init states 0-9 x seeds 16-31)"]
    lines.extend(harness_line(t, v) for t, v in r["harness"].items())
    if not r["harness_ok"]:
        lines.append("HARNESS FAILED: investigate before reading anything below")
    ci = r["label_rate_ci95"]
    lines.append(f"labels: base {r['label_hits']}/{r['label_episodes']} on the systematic states (allowed "
                 f"{MAX_LABEL_HITS}, at most {MAX_STATE_HITS} on a state; per state {r['label_hits_per_state']}; 95% "
                 f"interval {100 * ci[0]:.1f}-{100 * ci[1]:.1f}%): {'hold' if r['labels_hold'] else 'fail'}")
    lines.append(f"B's mean gain on {len(r['low_reachable_states'])} low reachable states: {r['low_gain']:+.3f} "
                 f"(required {MIN_GAIN}): {'informative' if r['informative'] else 'not informative'}")
    p = r["primary"]
    lines.append(f"uneven gain: D {p['D']:+.3f} (required {MIN_GAIN}), exact stratified p {p['p']:.4f} over "
                 f"{p['assignments']} assignments, episode-noise 95% interval {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f}, "
                 f"per task {p['per_task']}: {'yes' if r['uneven'] else 'no'}")
    lines.append(f"reading: {r['reading']}")
    s = r["secondary"]
    q = s["with_task_0"]
    lines.append(f"secondary, with task 0: D {q['D']:+.3f}, p {q['p']:.4f} over {q['assignments']} assignments")
    q = s["log_odds"]
    lines.append(f"secondary, on the log-odds scale: D {q['D']:+.3f}, p {q['p']:.4f}")
    rho = s["spearman_rate_gain"]
    lines.append(f"secondary, Spearman of rate on seeds 0-15 and gain, 50 states: {'n/a' if rho is None else f'{rho:+.3f}'}")
    for t, v in s["per_task"].items():
        lines.append(f"secondary, task {t}: B {v['run']}/160 against base {v['ref']} (net {v['net']:+d}, p {v['p']:.3g})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True, help="per_episode.csv files of the base on seeds 0-15")
    parser.add_argument("--base", required=True)
    parser.add_argument("--b")
    parser.add_argument("--task0_base")
    parser.add_argument("--task0_b")
    parser.add_argument("--out", help="report file; the numbers also go to <out>.json")
    parser.add_argument("--harness_only", action="store_true", help="check the base against seeds 0-15 and stop")
    args = parser.parse_args()
    if not args.harness_only and not all((args.b, args.task0_base, args.task0_b, args.out)):
        parser.error("--b, --task0_base, --task0_b and --out are required without --harness_only")

    per_state = pool(args.labels, list(ALL_TASKS), list(LABEL_SEEDS))
    labels = {k: sum(v) for k, v in per_state.items()}
    if set(labels) != {(t, i) for t in ALL_TASKS for i in INIT_STATES}:
        raise ValueError("the label files do not cover every state")
    if args.harness_only:
        lines, passed = harness_gate(labels, load(args.base, NEW_TASKS))
        print("\n".join(lines))
        return 0 if passed else 1
    result = analyse(labels, load(args.base, NEW_TASKS), load(args.b, NEW_TASKS),
                     load(args.task0_base, (0,)), load(args.task0_b, (0,)))
    report = format_report(result)
    print(report, end="")
    Path(args.out).write_text(report)
    inputs = args.labels + [args.base, args.b, args.task0_base, args.task0_b]
    result["inputs"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    root = Path(__file__).resolve().parent.parent
    result["commit"] = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H"], capture_output=True,
                                      text=True).stdout.strip()
    Path(args.out + ".json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
