"""Step 2: choose the selector on init states 10-19, then test it on init states 0-9 (README.md).

Reads ``<root>/task_<t>/per_episode_<policy>.csv`` written by run_step2_noise_selection.sh.

- ``--stage select``: every selector against ``base`` on the same episodes, and the rule fixed in README.md
  picks one: the selector (score accel or medoid) with the most successes, ties to the smaller N and then to
  medoid; if it is not ahead of ``base``, step 2 stops here as a negative result. ``first_n<N>`` (no
  selection) is reported as a check of the batching, not a candidate.
- ``--stage report``: the chosen selector against ``base`` (exact McNemar, exact 95% interval of the paired
  difference), the hypothesis of README.md (at least +10 points with p < 0.05; on the step-1 systematic
  states at most 1 in 10 episodes solved), and base success here next to the step-0 lerobot-eval run as a
  harness check.

    python -S noise_reachability/analyze_step2.py --stage report --root <dir> --policies base,medoid_n8 \
        --step1 week6/noise_reachability_step1_kaggle/step1_reachability.txt.json \
        --lerobot_eval week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv --out <file>

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_action_steps import compare_setting  # noqa: E402
from noise_selection import parse_policy_label  # noqa: E402

ALPHA = 0.05
MIN_GAIN = 0.10
MAX_SYSTEMATIC_RATE = 0.10
HARNESS_MAX_GAP = 17   # episodes, about two standard errors of the difference on 150 (README.md)


def load_policy(root, tasks, policy):
    """{(task, seed, init_state): success} over the tasks' per_episode_<policy>.csv files."""
    eps = {}
    for t in tasks:
        with open(Path(root) / f"task_{t}" / f"per_episode_{policy}.csv") as f:
            for r in csv.DictReader(f):
                eps[(int(r["task_id"]), int(r["seed"]), int(r["init_state"]))] = int(r["success"])
    return eps


def choose(successes):
    """The selection rule: most successes among accel/medoid, ties to smaller N, then medoid."""
    candidates = [p for p in successes if p != "base" and parse_policy_label(p)[1] in ("accel", "medoid")]
    if not candidates:
        raise ValueError("no accel or medoid policy to choose from")
    order = {"medoid": 0, "accel": 1}
    best = min(candidates, key=lambda p: (-successes[p], parse_policy_label(p)[0], order[parse_policy_label(p)[1]]))
    return best, successes[best] > successes["base"]


def systematic_rate(eps, systematic_states):
    """(successes, episodes) of ``eps`` on the (task, init state) pairs in ``systematic_states``."""
    keys = [k for k in eps if (k[0], k[2]) in systematic_states]
    return sum(eps[k] for k in keys), len(keys)


def select_stage(episodes):
    if "base" not in episodes:
        raise ValueError("the select stage needs the base policy")
    keys = set(episodes["base"])
    for p, eps in episodes.items():
        if set(eps) != keys:
            raise ValueError(f"{p} does not cover the same episodes as base")
    successes = {p: sum(e.values()) for p, e in episodes.items()}
    chosen, ahead = choose(successes)
    comparisons = {p: compare_setting(episodes["base"], e, ALPHA) for p, e in episodes.items() if p != "base"}
    return {"n_episodes": len(keys), "successes": successes, "chosen": chosen, "chosen_ahead_of_base": ahead,
            "comparisons": comparisons}


def report_stage(base, chosen, systematic_states, lerobot_eval=None):
    if set(base) != set(chosen):
        raise ValueError(f"base has {len(base)} episodes and the chosen policy {len(chosen)}, not the same set: "
                         "finish the run before reporting")
    r = compare_setting(base, chosen, ALPHA)
    n = r["n_episodes"]
    gain = (r["b_successes"] - r["a_successes"]) / n
    sys_ok, sys_n = systematic_rate(chosen, systematic_states)
    base_sys_ok, _ = systematic_rate(base, systematic_states)
    out = {"comparison": r, "gain": gain,
           "systematic": {"chosen": sys_ok, "base": base_sys_ok, "episodes": sys_n},
           "hypothesis": {"gain_ok": gain >= MIN_GAIN and r["mcnemar_p"] < ALPHA,
                          "systematic_ok": sys_n == 0 or sys_ok / sys_n <= MAX_SYSTEMATIC_RATE}}
    if lerobot_eval is not None:
        same = {k: v for k, v in lerobot_eval.items() if k in base}
        if set(same) != set(base):
            raise ValueError(f"the lerobot-eval run covers {len(same)} of the {len(base)} episodes")
        gap = abs(sum(base.values()) - sum(same.values()))
        out["harness_check"] = {"base_here": sum(base.values()), "lerobot_eval": sum(same.values()),
                                "episodes": len(same), "gap_ok": gap <= HARNESS_MAX_GAP}
    return out


def format_select(s):
    lines = [f"select stage: {s['n_episodes']} episodes per policy", "",
             "policy | successes | only base | only policy | McNemar p"]
    lines.append(f"base | {s['successes']['base']} | | |")
    for p, r in s["comparisons"].items():
        lines.append(f"{p} | {r['b_successes']} | {r['episodes_a_only']} | {r['episodes_b_only']} | {r['mcnemar_p']:.4f}")
    lines += ["", f"chosen: {s['chosen']} ({'ahead of base: run the report stage' if s['chosen_ahead_of_base'] else 'not ahead of base: step 2 stops, negative result'})"]
    return "\n".join(lines) + "\n"


def format_report_stage(r, name):
    c, sy = r["comparison"], r["systematic"]
    lo, hi = c["diff_ci95"]
    lines = [f"report stage: base vs {name}, {c['n_episodes']} paired episodes", "",
             f"base {c['a_successes']}, {name} {c['b_successes']}; only base {c['episodes_a_only']}, only {name} "
             f"{c['episodes_b_only']}; exact McNemar p = {c['mcnemar_p']:.4f}",
             f"paired difference {100 * r['gain']:+.1f} points, exact 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}",
             "", "task | base | " + name]
    lines += [f"{t} | {p['a']}/{p['n']} | {p['b']}/{p['n']}" for t, p in c["per_task"].items()]
    lines += ["", f"step-1 systematic states: {name} {sy['chosen']}/{sy['episodes']}, base {sy['base']}/{sy['episodes']}",
              f"hypothesis: gain >= {100 * MIN_GAIN:.0f} points with p < {ALPHA}: {'yes' if r['hypothesis']['gain_ok'] else 'no'}; "
              f"systematic episodes solved <= {100 * MAX_SYSTEMATIC_RATE:.0f}%: {'yes' if r['hypothesis']['systematic_ok'] else 'no'}"]
    if "harness_check" in r:
        h = r["harness_check"]
        lines.append(f"harness check: base {h['base_here']} here, {h['lerobot_eval']} through lerobot-eval on the same "
                     f"{h['episodes']} (task, seed, init state) episodes (different noise draws, not paired); "
                     f"gap <= {HARNESS_MAX_GAP}: {'yes' if h['gap_ok'] else 'NO: investigate the harness before reading the result'}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("select", "report"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[0, 4, 6, 7, 8])
    p.add_argument("--policies", required=True, help="comma-separated; report: base,<chosen>")
    p.add_argument("--step1", help="step1_reachability.txt.json (systematic states), report stage")
    p.add_argument("--lerobot_eval", help="per_episode.csv of the step-0 lerobot-eval run, report stage")
    p.add_argument("--out")
    args = p.parse_args()

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    episodes = {pol: load_policy(args.root, args.tasks, pol) for pol in policies}
    if args.stage == "select":
        result = select_stage(episodes)
        text = format_select(result)
    else:
        if policies[0] != "base" or len(policies) != 2:
            raise SystemExit("--policies must be base,<chosen> for the report stage")
        systematic = {tuple(x) for x in json.load(open(args.step1))["systematic_states"]}
        lerobot = None
        if args.lerobot_eval:
            with open(args.lerobot_eval) as f:
                lerobot = {(int(r["task_id"]), int(r["seed"]), int(r["init_state"])): int(r["success"])
                           for r in csv.DictReader(f)}
        result = report_stage(episodes["base"], episodes[policies[1]], systematic, lerobot)
        text = format_report_stage(result, policies[1])
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out + ".json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
