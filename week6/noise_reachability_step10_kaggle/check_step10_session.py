"""Checks of one step-10 session's results zip (unzipped), before the run is finished. Reads no evaluation success.

What it checks: the harness gate, provenance (analyze_step10.check_provenance), the replays per task and per source
state (succeeded again; same number of steps as the original), arm A's collection, the trained adapters, and how many
evaluation episodes each policy has (counts only, no successes). With --previous, the previous session's CSV files must
be unchanged at the start of the same files in this session (a resume appends, never rewrites).

    python -I -S week6/noise_reachability_step10_kaggle/check_step10_session.py <run> [--previous <run>] --out <txt>
"""
import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step10 as a  # noqa: E402


def replay_by_state(root, plan, steps):
    """{(task, family, state): [planned, succeeded, same steps]} over the replays."""
    out = {}
    for t, eps in plan.items():
        folder = a.task_root(root, t) / "replay"
        for f, s, k, j in eps:
            name = f"per_episode_{f}_state{s}.csv" if j is None else f"confirm_{f}_n{k}_state{s}.csv"
            seed = a.EPISODE_SEED_BASE + k if j is None else j
            row = next(r for r in csv.DictReader(open(folder / name)) if int(r["seed"]) == seed)
            c = out.setdefault((t, f, s), [0, 0, 0])
            c[0] += 1
            c[1] += int(row["success"])
            c[2] += int(row["success"]) == 1 and int(row["steps"]) == steps[(t, f, s, k, j)]
    return out


def source_of(t, s):
    folders = sorted({folder for folder, st in a.REPLAY_SOURCES[t] if st == s})
    return ", ".join(folders)


def eval_counts(root):
    """{(task, policy): (episodes, expected)} for every policy with any evaluation file; refuses repeats and strays."""
    out = {}
    for t in a.TASKS:
        want = {(s, i) for s in a.EVAL_SEEDS[t] for i in a.EVAL_STATES[t]}
        for p in a.policies(t):
            name = "base" if p == "base" else "lora"
            keys = []
            for f in sorted(glob.glob(str(a.task_root(root, t) / "eval" / p / "part*" / f"per_episode_{name}.csv"))):
                keys += [(int(r["seed"]), int(r["init_state"])) for r in csv.DictReader(open(f))]
            if keys:
                if len(keys) != len(set(keys)) or set(keys) - want:
                    raise ValueError(f"task {t} {p}: repeated or unplanned episodes")
                out[(t, p)] = (len(keys), len(want))
    return out


def adapters(root):
    """Per training run: final step, final training loss, final validation loss if any, training episodes (from the
    log when present)."""
    out = {}
    for d in sorted(Path(root, "runs").iterdir()):
        h = json.loads((d / "train_history.json").read_text())
        last = h[-1]
        val = [e["val_loss"] for e in h if "val_loss" in e]
        log = Path(root, "logs", f"train_{d.name}.log")
        m = re.search(r"(\d+) training episodes \(success\), (\d+) validation", log.read_text()) if log.exists() else None
        out[d.name] = {"step": last["step"], "loss": round(last["loss"], 4), "val_loss": round(val[-1], 4) if val else None,
                       "sec_per_step": round(last["sec_per_step"], 3),
                       "episodes": (int(m.group(1)), int(m.group(2))) if m else None}
    return out


def kept_from(prev, root):
    """Previous session's CSV files that are a prefix of the same file now, and those that are not."""
    kept, changed = 0, []
    for f in sorted(Path(prev).rglob("*.csv")):
        g = Path(root) / f.relative_to(prev)
        old = f.read_text().splitlines()
        if g.exists() and g.read_text().splitlines()[:len(old)] == old:
            kept += 1
        else:
            changed.append(str(f.relative_to(prev)))
    return kept, changed


def report(root, prev=None, zip_path=None):
    lines = []
    if zip_path:
        lines.append(f"zip sha256 {hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()}")
    commits = Path(root, "repo_commit.txt").read_text().split()
    lines.append(f"repo_commit.txt: {len(commits)} session(s), commits {sorted(set(commits))}")
    gate = a.gate(root)
    lines.append("harness gate: " + ("passed" if gate is None else f"FAILED {gate}"))
    prov = a.check_provenance(root)
    lines.append(f"provenance: {prov['records']} records, commit {prov['commit']}, clean")
    plan, steps = a.replay_plan(), a.source_steps()
    for t, c in a.replay_counts(root, plan, steps).items():
        lines.append(f"replays task {t}: {c['succeeded']}/{c['planned']} succeeded, {c['same_steps']} same steps")
    for (t, f, s), (n, ok, same) in sorted(replay_by_state(root, plan, steps).items()):
        lines.append(f"  {t}/{s} {f}: {ok}/{n} succeeded, {same} same steps (found in {source_of(t, s)})")
    rows = list(csv.DictReader(open(a.task_root(root, 0) / "collect_data" / "collection.csv")))
    lines.append(f"collection task 0: {len(rows)} episodes, {sum(int(r['success']) for r in rows)} successes")
    for name, v in adapters(root).items():
        lines.append(f"adapter {name}: step {v['step']}, loss {v['loss']}, val_loss {v['val_loss']}, "
                     f"{v['sec_per_step']} s/step, episodes (train, val) {v['episodes']}")
    done = 0
    for (t, p), (n, want) in eval_counts(root).items():
        done += n
        lines.append(f"eval task {t} {p}: {n}/{want} episodes" + ("" if n == want else " (partial)"))
    total = sum(len(a.EVAL_SEEDS[t]) * len(a.EVAL_STATES[t]) * len(a.policies(t)) for t in a.TASKS)
    lines.append(f"eval episodes run: {done}/{total} (counts only; no success read)")
    if prev:
        kept, changed = kept_from(prev, root)
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
