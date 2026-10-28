"""Tables for this folder's README, from the files in it (no number typed by hand).

1. The Colab run's critic and temperature over training (run/task_<t>/train_log.csv).
2. The offline critic variants on the recorded task-7 replay (offline/results/<label>.json).

    python -S week6/noise_reachability_step3_colab/summarize_step3.py > week6/noise_reachability_step3_colab/step3_diagnosis.txt
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODES = [20, 60, 100, 300, 800, 1499]
VARIANTS = [  # label, description
    ("td_utd8_ref", "TD, 8 updates/decision (the run's settings)"),
    ("td_utd1", "TD, 1 update/decision"),
    ("td_utd1_noent", "TD, 1 update/decision, no entropy term in the target"),
    ("clip_utd8", "TD target clipped to [0, 1], 8 updates/decision"),
    ("clip_utd1", "TD target clipped to [0, 1], 1 update/decision"),
    ("clip_utd1_alpha1e3", "clipped, 1 update/decision, fixed alpha 0.001"),
    ("mc_utd1", "Monte Carlo return, 1 update/decision"),
    ("mc_utd8", "Monte Carlo return, 8 updates/decision"),
    ("mc_full_critic", "Monte Carlo return, critic only, all 1500 episodes"),
]


def g(x, fmt="{:.3g}"):
    return "" if x in (None, "") else fmt.format(float(x))


def run_table():
    out = ["## Colab run: critic and temperature during training (train_log.csv, mean over the episode's updates)", "",
           "| Task | Episode | Q mean | Alpha | Entropy | Critic loss |", "|---|---|---|---|---|---|"]
    for t in (7, 0):
        rows = list(csv.DictReader(open(HERE / "run" / f"task_{t}" / "train_log.csv")))
        for e in EPISODES:
            r = rows[e]
            out.append(f"| {t} | {e} | {g(r['q_mean'])} | {g(r['alpha'])} | {g(r['entropy'], '{:.1f}')} | "
                       f"{g(r['critic_loss'])} |")
    return out


def offline_table():
    out = ["## Offline critics on the task-7 replay (held-out: every 5th episode)", "",
           "| Variant | Episodes | Q mean at end | Success AUC: best, end | Action AUC: min-max, end | "
           "Held-out Q, success / failure |", "|---|---|---|---|---|---|"]
    for label, desc in VARIANTS:
        log = json.load(open(HERE / "offline" / "results" / f"{label}.json"))["log"]
        ev = [r for r in log if "auc_q" in r]
        last = ev[-1]
        stopped = " (stopped: Q > 1e4)" if len(log) < json.load(open(HERE / "offline" / "results" / f"{label}.json"))["args"]["episodes"] else ""
        out.append(f"| {desc} | {len(log)}{stopped} | {g(log[-1]['q_mean'])} | "
                   f"{max(r['auc_q'] for r in ev):.3f}, {last['auc_q']:.3f} | "
                   f"{min(r['auc_action'] for r in ev):.3f}-{max(r['auc_action'] for r in ev):.3f}, "
                   f"{last['auc_action']:.3f} | {g(last['q_held_succ'])} / {g(last['q_held_fail'])} |")
    return out


if __name__ == "__main__":
    print("\n".join(run_table() + [""] + offline_table()))
