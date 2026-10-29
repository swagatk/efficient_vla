"""Step 4 of README.md: weight updates on task 0's systematic init states, report and decision.

Reads ``per_episode_<policy>.csv`` files of ``self_filtered_bc.py --stage eval`` (task_id, seed, init_state,
success): the base, arm A (self-filtered LoRA) and arm C (demonstration LoRA) per training seed, and arm B
(continued-training checkpoint). Every arm is compared with the base on the same episodes, separately on the
systematic and the reachable init states (exact McNemar, exact interval of the paired difference), A against C
per training seed on the systematic episodes, and the outcome is read by the rule fixed in README.md.

    python -S noise_reachability/analyze_step4.py --base <csv> --arm A0=<csv> A1=<csv> A2=<csv> \
        C0=<csv> C1=<csv> C2=<csv> B=<csv> --harness <csv of the base on seeds 0-2> --out <report.txt>

Reads CSV only: run with ``python -S`` so sitecustomize does not import MuJoCo.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_action_steps import paired_difference_ci  # noqa: E402  (puts baseline_performance on sys.path)
from compare_per_state import mcnemar_exact  # noqa: E402

TASK = 0
SYSTEMATIC_STATES = (2, 5, 6, 7, 8)      # step 1: 0 of 16 noise seeds
INIT_STATES = tuple(range(10))
EVAL_SEEDS = tuple(range(16, 32))        # fresh seeds: 0-15 chose the systematic states
ALPHA = 0.05
MIN_NET_SYSTEMATIC = 8                   # 10 points of 80 systematic episodes
MAX_NET_LOSS_REACHABLE = 4               # 5 points of 80 reachable episodes
MIN_SEEDS = 2                            # of an arm's 3 training seeds
HARNESS_SEEDS = (0, 1, 2)
HARNESS_REFERENCE = 7                    # step 0, task 0 at n_action_steps=10, init states 0-9 x seeds 0-2
HARNESS_TOLERANCE = 6                    # a gap of 7 or more episodes means the harness is investigated
ARMS = ("A", "C")                        # LoRA arms with training seeds; B has one checkpoint


def load(path, task=TASK):
    """{(task, seed, init_state): success} of one per_episode csv; refuses other tasks and repeated episodes."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (int(r["task_id"]), int(r["seed"]), int(r["init_state"]))
            if k[0] != task:
                raise ValueError(f"{path}: episode of task {k[0]}, expected {task}")
            if k in out:
                raise ValueError(f"{path}: episode {k} appears twice")
            out[k] = int(r["success"])
    return out


def expected_keys(seeds=EVAL_SEEDS, init_states=INIT_STATES, task=TASK):
    return {(task, s, i) for s in seeds for i in init_states}


def check_complete(eps, name, seeds=EVAL_SEEDS):
    want = expected_keys(seeds)
    if set(eps) != want:
        raise ValueError(f"{name}: {len(set(eps) & want)} of {len(want)} episodes and {len(set(eps) - want)} "
                         "others: finish the run before reporting")


def split(eps):
    """(systematic, reachable) sub-dicts of ``eps``."""
    sys_eps = {k: v for k, v in eps.items() if k[2] in SYSTEMATIC_STATES}
    return sys_eps, {k: v for k, v in eps.items() if k[2] not in SYSTEMATIC_STATES}


def paired(ref, run):
    """Exact McNemar and interval of ``run`` against ``ref`` on the same episodes; net = run-only minus ref-only."""
    if set(ref) != set(run):
        raise ValueError("the two runs do not cover the same episodes")
    ref_only = sum(1 for k in run if ref[k] and not run[k])
    run_only = sum(1 for k in run if run[k] and not ref[k])
    return {"ref": sum(ref.values()), "run": sum(run.values()), "n": len(run), "ref_only": ref_only,
            "run_only": run_only, "net": run_only - ref_only, "p": mcnemar_exact(ref_only, run_only),
            "diff_ci95": list(paired_difference_ci(ref_only, run_only, len(run)))}


def arm_result(base, arm):
    """The README hypothesis for one arm (one training seed) against the base."""
    (b_sys, b_reach), (a_sys, a_reach) = split(base), split(arm)
    s, r = paired(b_sys, a_sys), paired(b_reach, a_reach)
    s["ok"] = s["net"] >= MIN_NET_SYSTEMATIC and s["p"] < ALPHA
    r["guard_ok"] = r["net"] >= -MAX_NET_LOSS_REACHABLE and not (r["net"] < 0 and r["p"] < ALPHA)
    return {"systematic": s, "reachable": r, "meets": s["ok"] and r["guard_ok"]}


def c_ahead_of_a(a_eps, c_eps):
    """C against A on the systematic episodes of one training seed: C ahead with paired p < ALPHA."""
    r = paired(split(a_eps)[0], split(c_eps)[0])
    r["ahead"] = r["net"] > 0 and r["p"] < ALPHA
    return r


def pooled(base, per_seed):
    """Arm pooled over training seeds, the base repeated once per seed (overstates the evidence; README)."""
    pb, pa = {}, {}
    for seed, eps in per_seed.items():
        for (t, s, i), v in eps.items():
            pb[(t, 100 * seed + s, i)] = base[(t, s, i)]
            pa[(t, 100 * seed + s, i)] = v
    return arm_result(pb, pa)


def harness_check(eps):
    check_complete(eps, "harness check", HARNESS_SEEDS)
    n = sum(eps.values())
    return {"base": n, "n": len(eps), "reference": HARNESS_REFERENCE, "gap": n - HARNESS_REFERENCE,
            "ok": abs(n - HARNESS_REFERENCE) <= HARNESS_TOLERANCE}


def decide(results):
    """The reading fixed in README.md, from per-seed results of A and C, B's result and A-vs-C per seed."""
    count = {arm: sum(r["meets"] for r in results[arm].values()) for arm in ARMS}
    a_meets, c_meets = count["A"] >= MIN_SEEDS, count["C"] >= MIN_SEEDS
    c_ahead = sum(r["ahead"] for r in results["C_vs_A"].values()) >= MIN_SEEDS
    if a_meets:
        return "A meets the hypothesis: contradicts the task-6 result; re-check the harness and the data first"
    if c_meets and c_ahead:
        return "main claim supported: C meets the hypothesis, A does not, and C is ahead of A on systematic states"
    if c_meets:
        return "consistent with the claim, not a demonstration: C meets the hypothesis, A does not, C not ahead of A"
    if results["B"]["meets"]:
        return "no LoRA arm meets the hypothesis; B (the larger demonstration update) does"
    return "negative: no arm meets the hypothesis"


def analyse(base, arms, harness=None):
    """``arms``: {"A": {seed: eps}, "C": {seed: eps}, "B": eps}."""
    check_complete(base, "base")
    for arm in ARMS:
        if set(arms[arm]) != set(arms["A"]) or len(arms[arm]) != 3:
            raise ValueError(f"arm {arm} needs the same 3 training seeds as A, got {sorted(arms[arm])}")
        for seed, eps in arms[arm].items():
            check_complete(eps, f"{arm}{seed}")
    check_complete(arms["B"], "B")
    out = {arm: {seed: arm_result(base, eps) for seed, eps in sorted(arms[arm].items())} for arm in ARMS}
    out["B"] = arm_result(base, arms["B"])
    out["C_vs_A"] = {seed: c_ahead_of_a(arms["A"][seed], arms["C"][seed]) for seed in sorted(arms["A"])}
    out["pooled"] = {arm: pooled(base, arms[arm]) for arm in ARMS}
    out["base"] = {"systematic": sum(split(base)[0].values()), "reachable": sum(split(base)[1].values())}
    out["harness"] = harness_check(harness) if harness is not None else None
    out["decision"] = decide(out)
    if out["harness"] is not None and not out["harness"]["ok"]:
        out["decision"] = "not read: the harness check failed (" + out["decision"] + ")"
    return out


def _line(name, x):
    s, r = x["systematic"], x["reachable"]
    return (f"{name:<10} systematic {s['run']}/{s['n']} vs base {s['ref']} (net {s['net']:+d}, p {s['p']:.3g}, "
            f"diff {100 * s['diff_ci95'][0]:+.1f} to {100 * s['diff_ci95'][1]:+.1f}) {'ok' if s['ok'] else '-'} | "
            f"reachable {r['run']}/{r['n']} vs {r['ref']} (net {r['net']:+d}, p {r['p']:.3g}) "
            f"guard {'ok' if r['guard_ok'] else 'FAILED'} | meets: {x['meets']}")


def format_report(out):
    lines = ["step 4: weight updates on task 0's systematic init states (init states 0-9 x seeds 16-31)",
             f"systematic states {list(SYSTEMATIC_STATES)}; base {out['base']['systematic']}/80 systematic, "
             f"{out['base']['reachable']}/80 reachable"]
    h = out["harness"]
    lines.append("harness check: not given" if h is None else
                 f"harness check: base {h['base']}/{h['n']} against step 0's {h['reference']} "
                 f"(gap {h['gap']:+d}, allowed {HARNESS_TOLERANCE}): {'ok' if h['ok'] else 'FAILED'}")
    for arm in ARMS:
        for seed, x in out[arm].items():
            lines.append(_line(f"{arm}{seed}", x))
        lines.append(_line(f"{arm} pooled", out["pooled"][arm]))
    lines.append(_line("B", out["B"]))
    for seed, r in out["C_vs_A"].items():
        lines.append(f"C{seed} vs A{seed} systematic: {r['run']} vs {r['ref']} (net {r['net']:+d}, p {r['p']:.3g}) "
                     f"{'C ahead' if r['ahead'] else '-'}")
    lines.append(f"decision: {out['decision']}")
    return "\n".join(lines)


def parse_arms(specs):
    """'A0=path' ... 'B=path' -> {"A": {0: eps}, "C": {...}, "B": eps}."""
    arms = {"A": {}, "C": {}}
    for spec in specs:
        name, path = spec.split("=", 1)
        if name == "B":
            arms["B"] = load(path)
        elif name[:1] in ARMS and name[1:].isdigit():
            arms[name[0]][int(name[1:])] = load(path)
        else:
            raise ValueError(f"unknown arm {name!r}: use A<seed>, C<seed> or B")
    if "B" not in arms:
        raise ValueError("arm B is missing")
    return arms


def input_record(args) -> dict:
    """Commit of the analysis code and the SHA-256 of every input file."""
    files = [args.base] + [s.split("=", 1)[1] for s in args.arm] + ([args.harness] if args.harness else [])
    repo = Path(__file__).resolve().parent.parent
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"git_commit": head.stdout.strip() or None,
            "inputs": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="per_episode csv of the base on the evaluation episodes")
    p.add_argument("--arm", nargs="+", required=True, help="A<seed>=csv, C<seed>=csv, B=csv")
    p.add_argument("--harness", help="per_episode csv of the base on init states 0-9 x seeds 0-2")
    p.add_argument("--out", required=True, help="report file; the numbers also go to <out>.json")
    args = p.parse_args(argv)
    out = analyse(load(args.base), parse_arms(args.arm), load(args.harness) if args.harness else None)
    report = format_report(out)
    print(report)
    Path(args.out).write_text(report + "\n")
    Path(args.out + ".json").write_text(json.dumps({"args": vars(args), "provenance": input_record(args), **out},
                                                   indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
