"""Step 3: the trained noise policy against the base, per task (README.md, step 3).

Reads ``<root>/task_<t>/eval/per_episode_{base,actor}.csv`` and ``<root>/task_<t>/train_log.csv`` written by
run_step3_noise_sac.sh. Per task: exact McNemar and the exact 95% interval of the paired difference, and the
hypothesis of README.md (at least +10 points with p < 0.05). On tasks with step-1 systematic states: the
actor's success on systematic and on reachable states, and whether the first stays below a third of the
second. The training log is summarised (success during warm-up, the first and the last 100 episodes after
it); it is a description of training, not a result.

    python -S noise_reachability/analyze_step3.py --root <dir> --tasks 7 0 \
        --step1 week6/noise_reachability_step1_kaggle/step1_reachability.txt.json --out <file>

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step2 import ALPHA, MIN_GAIN  # noqa: E402
from compare_action_steps import compare_setting  # noqa: E402
from reachability import parse_range  # noqa: E402

SYSTEMATIC_RATIO = 1 / 3   # success on systematic states below this fraction of success on reachable ones
TRAIN_WINDOW = 100


def load_eval(root, task, policy):
    """{(task, seed, init_state): success} of one task's per_episode_<policy>.csv."""
    with open(Path(root) / f"task_{task}" / "eval" / f"per_episode_{policy}.csv") as f:
        return {(int(r["task_id"]), int(r["seed"]), int(r["init_state"])): int(r["success"]) for r in csv.DictReader(f)}


def expected_keys(task, seeds, init_states):
    return {(task, s, i) for s in seeds for i in init_states}


def split_rate(eps, task, systematic_states):
    """(successes, episodes) on systematic and on reachable init states of ``task``."""
    sys_keys = [k for k in eps if (task, k[2]) in systematic_states]
    reach_keys = [k for k in eps if (task, k[2]) not in systematic_states]
    return ((sum(eps[k] for k in sys_keys), len(sys_keys)), (sum(eps[k] for k in reach_keys), len(reach_keys)))


def systematic_ok(sys_ok, sys_n, reach_ok, reach_n):
    """Systematic success rate below SYSTEMATIC_RATIO times the reachable success rate (strict)."""
    return sys_ok / sys_n < SYSTEMATIC_RATIO * (reach_ok / reach_n)


def task_result(base, actor, task, seeds, init_states, systematic_states):
    want = expected_keys(task, seeds, init_states)
    for name, eps in (("base", base), ("actor", actor)):
        if set(eps) != want:
            raise ValueError(f"task {task}: {name} has {len(set(eps) & want)} of {len(want)} episodes "
                             f"and {len(set(eps) - want)} others: finish the run before reporting")
    r = compare_setting(base, actor, ALPHA)
    gain = (r["b_successes"] - r["a_successes"]) / r["n_episodes"]
    out = {"comparison": r, "gain": gain, "gain_ok": gain >= MIN_GAIN and r["mcnemar_p"] < ALPHA}
    (s_ok, s_n), (r_ok, r_n) = split_rate(actor, task, systematic_states)
    if s_n:
        (bs_ok, _), (br_ok, _) = split_rate(base, task, systematic_states)
        out["systematic"] = {"actor": [s_ok, s_n], "actor_reachable": [r_ok, r_n], "base": [bs_ok, s_n],
                             "base_reachable": [br_ok, r_n], "ok": systematic_ok(s_ok, s_n, r_ok, r_n)}
    return out


def train_summary(rows, warmup, window=TRAIN_WINDOW):
    """Success counts during warm-up, the first ``window`` and the last ``window`` episodes after it."""
    eps = sorted(rows, key=lambda r: int(r["episode"]))
    if [int(r["episode"]) for r in eps] != list(range(len(eps))):
        raise ValueError("train_log.csv does not hold episodes 0..n-1 exactly once")
    after = [int(r["success"]) for r in eps[warmup:]]
    rate = lambda xs: [sum(xs), len(xs)]
    return {"episodes": len(eps), "warmup": rate([int(r["success"]) for r in eps[:warmup]]),
            "first_after_warmup": rate(after[:window]), "last": rate(after[-window:])}


def format_report(results):
    lines = ["step 3: noise policy (SAC) vs base, paired episodes per task", "",
             "task | base | actor | only base | only actor | McNemar p | difference (95% interval) | +10 points, p < 0.05"]
    for t, r in results.items():
        c = r["comparison"]
        lo, hi = c["diff_ci95"]
        lines.append(f"{t} | {c['a_successes']}/{c['n_episodes']} | {c['b_successes']}/{c['n_episodes']} | "
                     f"{c['episodes_a_only']} | {c['episodes_b_only']} | {c['mcnemar_p']:.4f} | "
                     f"{100 * r['gain']:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}) | {'yes' if r['gain_ok'] else 'no'}")
    for t, r in results.items():
        if "systematic" in r:
            s = r["systematic"]
            lines += ["", f"task {t} systematic states: actor {s['actor'][0]}/{s['actor'][1]}, reachable "
                          f"{s['actor_reachable'][0]}/{s['actor_reachable'][1]} (base {s['base'][0]}/{s['base'][1]} and "
                          f"{s['base_reachable'][0]}/{s['base_reachable'][1]}); systematic below a third of reachable: "
                          f"{'yes' if s['ok'] else 'no'}"]
    for t, r in results.items():
        if "train" in r:
            tr = r["train"]
            f = lambda x: f"{x[0]}/{x[1]}"
            lines += ["", f"task {t} training ({tr['episodes']} episodes, stochastic actor, init states 10-49): warm-up "
                          f"{f(tr['warmup'])}, first {TRAIN_WINDOW} after warm-up {f(tr['first_after_warmup'])}, "
                          f"last {TRAIN_WINDOW} {f(tr['last'])}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[7, 0])
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--init_states", default="0-9")
    p.add_argument("--warmup_episodes", type=int, default=20)
    p.add_argument("--step1", required=True, help="step1_reachability.txt.json (systematic states)")
    p.add_argument("--out")
    args = p.parse_args(argv)

    systematic = {tuple(x) for x in json.load(open(args.step1))["systematic_states"]}
    seeds, states = parse_range(args.seeds), parse_range(args.init_states)
    results = {}
    for t in args.tasks:
        results[t] = task_result(load_eval(args.root, t, "base"), load_eval(args.root, t, "actor"), t, seeds, states,
                                 systematic)
        log = Path(args.root) / f"task_{t}" / "train_log.csv"
        if log.exists():
            with open(log) as f:
                results[t]["train"] = train_summary(list(csv.DictReader(f)), args.warmup_episodes)
    text = format_report(results)
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out + ".json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
