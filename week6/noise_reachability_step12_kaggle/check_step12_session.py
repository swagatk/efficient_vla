"""Checks of one step-12 session's results zip (unzipped), before the run is finished. Reads no evaluation success.

What it checks: the harness gate per task, provenance (analyze_step12.check_provenance), the explorer's search per
state, that every finished round-2 run trained on round 1's episodes plus every search success of its task
(analyze_step12.check_training_data), the trained adapters, whether a round-2 run whose training set equals round 1's
also reproduced round 1's training (the loss at every logged step), and how many evaluation episodes each policy has
(counts only, no successes). With --previous, the previous session's CSV files must be unchanged at the start of the
same files in this session (a resume appends, never rewrites).

    python -I -S week6/noise_reachability_step12_kaggle/check_step12_session.py <run> [--previous <run>] --out <txt>
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step12 as a  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "check_step10_session", ROOT / "week6/noise_reachability_step10_kaggle/check_step10_session.py")
s10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s10)        # adapters() and kept_from() are shared with step 10's session check


def eval_counts(root):
    """{(task, policy): (episodes, expected)} for every policy with any evaluation file; refuses repeats and strays."""
    out = {}
    for t in a.TASKS:
        want = set(a.keys_of(t, a.eval_states(t)))
        for p in a.POLICIES:
            keys = []
            for f in sorted(glob.glob(str(a.task_root(root, t) / "eval" / p / "part*" / "per_episode_lora.csv"))):
                keys += [(int(r["seed"]), int(r["init_state"])) for r in csv.DictReader(open(f))]
            if keys:
                if len(keys) != len(set(keys)) or set(keys) - want:
                    raise ValueError(f"task {t} {p}: repeated or unplanned episodes")
                out[(t, p)] = (len(keys), len(want))
    return out


def same_as_round_1(root, step10=a.STEP10):
    """For each finished round-2 run trained on no search success: (logged steps, steps whose loss equals round 1's
    N<t>_<s> exactly). Equal everywhere means round 2 reproduced round 1's training: R<t>_<s> is in effect N<t>_<s>."""
    out = {}
    for d in sorted(Path(root, "runs").glob("R*")):
        name = d.name
        r1 = Path(step10, "runs", "N" + name[1:])
        if set(a.train_episodes(d / "provenance.jsonl")) != set(a.train_episodes(r1 / "provenance.jsonl")):
            continue
        new = json.loads((d / "train_history.json").read_text())
        old = json.loads((r1 / "train_history.json").read_text())
        if [e["step"] for e in new] != [e["step"] for e in old]:
            raise ValueError(f"{name}: logged steps differ from round 1's")
        out[name] = (len(new), sum(x["loss"] == y["loss"] for x, y in zip(new, old)))
    return out


def report(root, prev=None, zip_path=None, step10=a.STEP10):
    lines = []
    if zip_path:
        lines.append(f"zip sha256 {hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()}")
    commits = Path(root, "repo_commit.txt").read_text().split()
    lines.append(f"repo_commit.txt: {len(commits)} session(s), commits {sorted(set(commits))}")
    h = a.harness(root, step10)
    lines.append(f"harness gate: {'passed' if h['passes'] else 'FAILED'}, {h['same']}/{h['episodes']} episodes the "
                 f"same as step 10 in success and steps (at least {a.HARNESS_MIN_SAME})")
    for t, v in h["per_task"].items():
        lines.append(f"  task {t} state {a.HARNESS_STATE[t]}: {v['same']}/{v['episodes']} the same, successes "
                     f"{v['successes_now']} (step 10 {v['successes_step10']})")
    prov = a.check_provenance(root)
    lines.append(f"provenance: {prov['records']} records, commit {prov['commit']}, clean")
    search = a.search_counts(root)
    for (t, s), v in search.items():
        lines.append(f"search {t}/{s}: {v['hits']}/{v['episodes']} successes, noise ids {v['ids']}")
    lines.append(f"search total: {sum(v['hits'] for v in search.values())}/{sum(v['episodes'] for v in search.values())}")
    checked = a.check_training_data(root, search, step10, skip_missing=True)
    per_state = a.training_data(root)
    for name, (old, new) in checked.items():
        lines.append(f"training set {name}: round 1's {old} episodes plus {new} search successes, as the protocol "
                     f"says; per init state {json.dumps(per_state[name], sort_keys=True)}")
    for name, v in s10.adapters(root).items():
        lines.append(f"adapter {name}: step {v['step']}, loss {v['loss']}, {v['sec_per_step']} s/step")
    for name, (n, same) in same_as_round_1(root, step10).items():
        lines.append(f"{name} trained on round 1's set only: loss equal to N{name[1:]}'s at {same}/{n} logged steps")
    done = 0
    for (t, p), (n, want) in eval_counts(root).items():
        done += n
        lines.append(f"eval task {t} {p}: {n}/{want} episodes" + ("" if n == want else " (partial)"))
    total = sum(len(a.keys_of(t, a.eval_states(t))) * len(a.POLICIES) for t in a.TASKS)
    lines.append(f"eval episodes run: {done}/{total} (counts only; no success read)")
    if prev:
        kept, changed = s10.kept_from(prev, root)
        lines.append(f"previous session's CSV files kept unchanged at the start: {kept}, changed or missing: {changed}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--previous")
    ap.add_argument("--zip")
    ap.add_argument("--out")
    x = ap.parse_args(argv)
    text = report(Path(x.root), x.previous and Path(x.previous), x.zip)
    print(text, end="")
    if x.out:
        Path(x.out).write_text(text)


if __name__ == "__main__":
    main()
