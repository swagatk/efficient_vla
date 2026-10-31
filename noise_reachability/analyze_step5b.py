"""Step 5b analysis: the gates, the per-seed rule and the reading fixed in README.md. Standard library only.

    python3 -S noise_reachability/analyze_step5b.py --root <run folder> --epochs 37 --out step5b_report.txt
    python3 -S noise_reachability/analyze_step5b.py --root <run folder> --gate harness      # exit 1 stops the run
    python3 -S noise_reachability/analyze_step5b.py --root <run folder> --gate training --epochs 37

Layout: ``eval/shared/per_episode_{base,prior}.csv`` as parts, ``eval/seedT/...`` per training seed,
``train/seedT/train_log.csv``. Evaluation parts are ``partK`` folders, one per seed group.
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step4 import ALPHA, MAX_NET_LOSS_REACHABLE, MIN_NET_SYSTEMATIC, MIN_SEEDS, TASK  # noqa: E402
from analyze_step4 import check_complete, load, paired, split  # noqa: E402

TRAIN_SEEDS = (0, 1, 2)
POLICY_REVISION = "a24f1f3804"
# Both references: week6/noise_reachability_step5b_colab/references/ (tests/test_step5b_references.py).
HARNESS_REFERENCE = 37                # step 4's base on these 160 episodes
HARNESS_TOLERANCE = 16                # 20% of the episodes, as the other steps
BASE_TRAINING_SUCCESS = 50 / 120      # step 4's collection on init states 10-49; 3 standard errors above the gate
TRAIN_STATES_PER_EPOCH = 40
GATE_EPOCHS = 5
GATE_MIN_SUCCESS = 0.30
MIN_TRAINING_GAIN = 0.08


def load_arm(folder, arm):
    """One arm's episodes from its ``part*/`` csv files; a repeated episode raises."""
    out = {}
    for path in sorted(glob.glob(f"{folder}/part*/per_episode_{arm}.csv")):
        eps = load(path)
        if set(eps) & set(out):
            raise ValueError(f"{path}: episodes repeated across parts")
        out.update(eps)
    return out


def check_provenance(root, epochs=None):
    """One commit, clean trees, the protocol's settings and one policy path in every record of the run (not the pilot)."""
    records = [json.loads(line) for f in sorted(Path(root).rglob("provenance.jsonl")) if "pilot" not in f.parts
               for line in open(f)]
    if not records:
        raise ValueError("provenance: no records")
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1 or None in commits:
        raise ValueError(f"provenance: commits {commits}")
    for r in records:
        st = r["settings"]
        if r.get("git_dirty") is not False or r.get("git_dirty_noise_reachability") is not False:
            raise ValueError("provenance: a run on a dirty tree")
        if st["task_id"] != TASK or st["n_action_steps"] != 10 or st["max_steps"] != 520:
            raise ValueError(f"provenance: settings {st}")
        if st["train_init_states"] != "10-49" or st["updates_per_epoch"] != 100:
            raise ValueError(f"provenance: settings {st}")
    paths = {r["settings"]["policy_path"] for r in records}
    if len(paths) != 1 or POLICY_REVISION not in next(iter(paths)):
        raise ValueError(f"provenance: policy paths {paths}")
    if epochs is not None:
        check_final_epoch(records, epochs)
    return {"records": len(records), "commit": commits.pop(), "policy_path": paths.pop()}


def check_final_epoch(records, epochs):
    """Every learned arm was evaluated on the last epoch's actor."""
    for r in records:
        st = r["settings"]
        if st.get("arm") in ("actor", "actor_mean") and st["epoch"] != epochs - 1:
            raise ValueError(f"provenance: {st['arm']} evaluated at epoch {st['epoch']}, expected {epochs - 1}")


def harness(root):
    eps = load_arm(f"{root}/eval/shared", "base")
    check_complete(eps, "base")
    n = sum(eps.values())
    return {"base": n, "reference": HARNESS_REFERENCE, "gap": n - HARNESS_REFERENCE,
            "ok": abs(n - HARNESS_REFERENCE) <= HARNESS_TOLERANCE}


def training_rows(root, seed, epochs):
    path = Path(root, "train", f"seed{seed}", "train_log.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    want = epochs * TRAIN_STATES_PER_EPOCH
    if len(rows) != want or {int(r["epoch"]) for r in rows} != set(range(epochs)):
        raise ValueError(f"{path}: {len(rows)} rows, expected {want} over {epochs} epochs")
    if len({r["seed"] for r in rows}) != want:
        raise ValueError(f"{path}: repeated episode seeds")
    return rows


def last_epochs_success(rows, epochs):
    last = [int(r["success"]) for r in rows if int(r["epoch"]) >= epochs - GATE_EPOCHS]
    return sum(last) / len(last)


def first_epoch_success(rows_per_seed):
    """Success of the untrained actor on the training states: epoch 0 of every training seed, pooled."""
    first = [int(r["success"]) for rows in rows_per_seed for r in rows if int(r["epoch"]) == 0]
    return sum(first) / len(first)


def seed_result(base, prior, actor):
    """The README rule for one training seed."""
    (b_sys, b_reach), (a_sys, a_reach), (p_sys, p_reach) = split(base), split(actor), split(prior)
    s, r, l, lr = paired(b_sys, a_sys), paired(b_reach, a_reach), paired(p_sys, a_sys), paired(p_reach, a_reach)
    s["ok"] = s["net"] >= MIN_NET_SYSTEMATIC and s["p"] < ALPHA
    r["guard_ok"] = r["net"] >= -MAX_NET_LOSS_REACHABLE and not (r["net"] < 0 and r["p"] < ALPHA)
    r["gain"] = r["net"] >= MIN_NET_SYSTEMATIC and r["p"] < ALPHA
    r["learned"] = lr["net"] >= MIN_NET_SYSTEMATIC and lr["p"] < ALPHA
    l["ok"] = l["net"] >= MIN_NET_SYSTEMATIC and l["p"] < ALPHA
    return {"systematic": s, "reachable": r, "against_prior": l, "reachable_against_prior": lr, "meets": s["ok"] and r["guard_ok"], "learned": l["ok"]}


def decide(seeds, training, training_prior):
    """The reading fixed in README.md, from {seed: seed_result}, {seed: last epochs' training success} and the
    untrained actor's training success (epoch 0 of the three seeds)."""
    r = list(seeds.values())
    meets = sum(x["meets"] for x in r) >= MIN_SEEDS
    both = sum(x["meets"] and x["learned"] for x in r) >= MIN_SEEDS
    if both:
        return "hypothesis rejected: the actor transfers to held-out systematic states; re-check for bugs first"
    if meets:
        return "the repeated form explains the gain, not learning"
    if sum(x["reachable"]["gain"] and x["reachable"]["learned"] for x in r) >= MIN_SEEDS:
        return "hypothesis stands: the actor transfers to held-out reachable states, not to systematic ones"
    if sum(v >= training_prior + MIN_TRAINING_GAIN for v in training.values()) >= MIN_SEEDS:
        return "hypothesis stands: the actor learned its training states and did not transfer"
    return "no reading on transfer: the learner did not learn at this budget"


def analyse(root, epochs):
    h = harness(root)
    base = load_arm(f"{root}/eval/shared", "base")
    prior = load_arm(f"{root}/eval/shared", "prior")
    check_complete(prior, "prior")
    seeds, training, secondary, rows = {}, {}, {}, []
    for t in TRAIN_SEEDS:
        actor = load_arm(f"{root}/eval/seed{t}", "actor")
        check_complete(actor, f"actor{t}")
        seeds[t] = seed_result(base, prior, actor)
        rows.append(training_rows(root, t, epochs))
        training[t] = last_epochs_success(rows[-1], epochs)
        mean = load_arm(f"{root}/eval/seed{t}", "actor_mean")
        check_complete(mean, f"actor_mean{t}", (16,))
        secondary[f"actor_mean{t}"] = sum(mean.values())
    zero = load_arm(f"{root}/eval/shared", "zero")
    check_complete(zero, "zero", (16,))
    secondary["zero"] = sum(zero.values())
    training_prior = first_epoch_success(rows)
    out = {"harness": h, "base": {"systematic": sum(split(base)[0].values()), "reachable": sum(split(base)[1].values())},
           "prior": {"systematic": sum(split(prior)[0].values()), "reachable": sum(split(prior)[1].values())},
           "seeds": seeds, "training_last_epochs": training, "training_prior": training_prior,
           "secondary_of_10": secondary, "decision": decide(seeds, training, training_prior),
           "provenance": check_provenance(root, epochs)}
    if not h["ok"]:
        out["decision"] = "not read: the harness failed (" + out["decision"] + ")"
    return out


def format_report(o):
    h = o["harness"]
    lines = [f"harness: base {h['base']}/160 against {h['reference']} (gap {h['gap']:+d}, {'ok' if h['ok'] else 'FAILED'})",
             f"base: systematic {o['base']['systematic']}/80, reachable {o['base']['reachable']}/80",
             f"prior: systematic {o['prior']['systematic']}/80, reachable {o['prior']['reachable']}/80"]
    for t, x in o["seeds"].items():
        s, r, l = x["systematic"], x["reachable"], x["against_prior"]
        lines.append(f"actor{t}: systematic {s['run']}/80 (net {s['net']:+d}, p {s['p']:.3g}) | reachable {r['run']}/80 "
                     f"(net {r['net']:+d}, p {r['p']:.3g}, guard {'ok' if r['guard_ok'] else 'FAILED'}) | against prior "
                     f"net {l['net']:+d}, p {l['p']:.3g} | meets {x['meets']} learned {x['learned']}, reachable {r['learned']} | training, last "
                     f"{GATE_EPOCHS} epochs: {100 * o['training_last_epochs'][t]:.1f}%")
    lines.append(f"training, untrained actor (epoch 0, 3 seeds): {100 * o['training_prior']:.1f}%")
    lines.append(f"secondary, of 10 (one outcome per state, not tested): {o['secondary_of_10']}")
    lines.append(f"decision: {o['decision']}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--gate", choices=["harness", "training"])
    p.add_argument("--out")
    args = p.parse_args(argv)
    if args.gate == "harness":
        h = harness(args.root)
        print(f"gate harness: base {h['base']}/160 against {h['reference']}: {'passed' if h['ok'] else 'FAILED'}")
        return 0 if h["ok"] else 1
    if args.gate == "training":
        v = last_epochs_success(training_rows(args.root, 0, args.epochs), args.epochs)
        ok = v >= GATE_MIN_SUCCESS
        print(f"gate training: seed 0 at {100 * v:.1f}% over the last {GATE_EPOCHS} epochs: {'passed' if ok else 'FAILED'}")
        return 0 if ok else 1
    out = analyse(args.root, args.epochs)
    report = format_report(out)
    print(report)
    Path(args.out).write_text(report + "\n")
    Path(args.out + ".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
