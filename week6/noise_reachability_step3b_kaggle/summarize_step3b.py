"""Checks for this folder's README, from the files in it (no number typed by hand).

1. Each collection stream (run/seed_<s>/task_7/): episodes, successes, commit, settings, episode time, errors.
2. Kaggle's report (run/step3b_report.txt.json) against the local re-run (local_rerun/step3b_report.txt.json).
3. With --replay (the folder of step3b_replay.zip plus each stream's checkpoint.pt, not in git): the saved
   episodes and replay parts, replay size against the logged decisions, and the stored noise's distribution.

    python -I week6/noise_reachability_step3b_kaggle/summarize_step3b.py --replay <dir> > week6/noise_reachability_step3b_kaggle/step3b_checks.txt
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2, 3]
TASK = 7
PROTOCOL = {"task_id": 7, "n_action_steps": 10, "max_steps": 520, "train_init_states": "10-49", "num_episodes": 250,
            "warmup_episodes": 250, "updates_per_decision": 0, "noise_scale": 5.0, "train_hard_reset": 0}


def stream_table():
    out = ["## Collection streams (run/seed_<s>/task_7/)", "",
           "| Seed | Episodes (0-249 once, in order) | Successes | Decisions | Records | Commit | Dirty | "
           "Settings as protocol | Episode time (h) | First record (UTC) | Last record | Tracebacks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        d = HERE / "run" / f"seed_{s}" / f"task_{TASK}"
        rows = list(csv.DictReader(open(d / "train_log.csv")))
        prov = [json.loads(line) for line in open(d / "provenance.jsonl")]
        eps = [int(r["episode"]) for r in rows]
        commits = sorted({p["git_commit"][:7] for p in prov})
        dirty = any(v for p in prov for k, v in p.items() if k.startswith("git_dirty"))
        same = all(p["settings"][k] == v and p["settings"]["seed"] == s for p in prov for k, v in PROTOCOL.items())
        log = (d / "train.log").read_text() + (HERE / "run" / f"seed_{s}" / "runner.log").read_text()
        out.append(f"| {s} | {len(rows)} ({'yes' if eps == list(range(250)) else 'NO'}) | "
                   f"{sum(int(r['success']) for r in rows)} | {sum(int(r['decisions']) for r in rows)} | {len(prov)} | "
                   f"{', '.join(commits)} | {dirty} | {same} | {sum(float(r['time_s']) for r in rows) / 3600:.2f} | "
                   f"{prov[0]['date'][:16]} | {prov[-1]['date'][:16]} | {log.count('Traceback')} |")
    return out


def report_table():
    a = json.load(open(HERE / "run" / "step3b_report.txt.json"))
    b = json.load(open(HERE / "local_rerun" / "step3b_report.txt.json"))
    ga, gb = a["analysis_provenance"], b["analysis_provenance"]
    out = ["## Analysis: Kaggle against the local re-run", "",
           f"| Measure | Kaggle ({', '.join(ga['gpus'])}; {ga['git_commit'][:7]}, dirty {ga['git_dirty']}) | "
           f"Local ({', '.join(gb['gpus'])}; {gb['git_commit'][:7]}, dirty {gb['git_dirty']}) | Difference |",
           "|---|---|---|---|"]
    for k in ("critic", "control"):
        x, y = a[k], b[k]
        out.append(f"| {k} action AUC | {x['action_auc']:.4f} ({x['action_ci'][0]:.4f} to {x['action_ci'][1]:.4f}) | "
                   f"{y['action_auc']:.4f} ({y['action_ci'][0]:.4f} to {y['action_ci'][1]:.4f}) | "
                   f"{y['action_auc'] - x['action_auc']:+.4f} |")
        out.append(f"| {k} success AUC | {x['success_auc']:.4f} | {y['success_auc']:.4f} | "
                   f"{y['success_auc'] - x['success_auc']:+.4f} |")
    out.append(f"| critic minus control, action AUC | {a['critic']['action_auc'] - a['control']['action_auc']:+.4f} | "
               f"{b['critic']['action_auc'] - b['control']['action_auc']:+.4f} | |")
    out.append(f"| decision | {a['decision']} | {b['decision']} | |")
    out += ["", f"Data: {a['n_held_episodes']} held-out episodes, {a['n_held_transitions']} held-out transitions, "
            f"{a['n_train_transitions']} training transitions (local: {b['n_held_episodes']}, {b['n_held_transitions']}, "
            f"{b['n_train_transitions']}); same settings: {a['settings'] == b['settings']}; "
            f"same stream counts: {a['streams'] == b['streams']}."]
    return out


def replay_checks(root: Path):
    import numpy as np
    import torch
    from scipy import stats

    out = ["## Replay data and checkpoints (--replay, not in git)", "",
           "| Seed | Checkpoint episodes | Parts listed / on disk | Transitions | Logged decisions | Terminal rewards | "
           "Logged successes | Logs equal to run/ |", "|---|---|---|---|---|---|---|---|"]
    us = []
    for s in SEEDS:
        d = root / f"seed_{s}" / f"task_{TASK}"
        ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=True)
        parts = [np.load(d / p, allow_pickle=False) for p in ck["replay_parts"]]
        us += [p["u"] for p in parts]
        rows = list(csv.DictReader(open(d / "train_log.csv")))
        same = all((d / f).read_bytes() == (HERE / "run" / f"seed_{s}" / f"task_{TASK}" / f).read_bytes()
                   for f in ("train_log.csv", "provenance.jsonl"))
        out.append(f"| {s} | {ck['episodes_done']} | {len(ck['replay_parts'])} / {len(list(d.glob('*.npz')))} | "
                   f"{sum(len(p['u']) for p in parts)} | {sum(int(r['decisions']) for r in rows)} | "
                   f"{int(sum(p['reward'].sum() for p in parts))} | {sum(int(r['success']) for r in rows)} | {same} |")
    u = np.concatenate(us)
    x = (u * 5).ravel()
    ks = stats.kstest(x[:200000], "norm")
    out += ["", f"Stored noise x 5 ({u.shape[0]} decisions x {u.shape[1]} entries): mean {x.mean():.4f}, "
            f"std {x.std():.4f}, skew {stats.skew(x):.4f}, excess kurtosis {stats.kurtosis(x):.4f}, "
            f"max |u| {np.abs(u).max():.4f}; KS against N(0, 1) on the first 200,000 entries: "
            f"D {ks.statistic:.4f}, p {ks.pvalue:.2f}; repeated rows: {u.shape[0] - len(np.unique(u, axis=0))}."]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replay", type=Path, help="folder with seed_<s>/task_7/ replay parts and checkpoint.pt")
    args = p.parse_args()
    lines = stream_table() + [""] + report_table()
    if args.replay:
        lines += [""] + replay_checks(args.replay)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
