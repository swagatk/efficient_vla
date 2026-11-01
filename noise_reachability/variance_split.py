"""Exploratory: share of success variance between tasks, between init states within a task, and between executions.

One-way variance components, as a comparison with arXiv 2610.29730; not part of any step's rule. Base policy,
n_action_steps=10, seeds 0-15 (steps 0, 1) and seeds 16-31 (step 6). Run from the repo root.
"""
import csv
from collections import defaultdict

FILES = {"seeds 0-15": [("week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv", range(3)),
                        ("week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv", range(3, 16))],
         "seeds 16-31": [("week6/noise_reachability_step6_kaggle/run/eval/per_episode_base.csv", range(16, 32))]}
TASKS = (4, 6, 7, 8)        # 8 included: its stove is not fixed by the init state (README.md, step 7)


def load(parts):
    cells = defaultdict(list)
    for f, seeds in parts:
        for r in csv.DictReader(open(f)):
            t, s = int(r["task_id"]), int(r["seed"])
            if t in TASKS and s in seeds:
                cells[(t, int(r["init_state"]))].append(int(r["success"]))
    return cells


def split(cells):
    """Sums of squares: tasks, states within tasks, executions within states (balanced cells)."""
    k = {len(v) for v in cells.values()}
    assert len(k) == 1 and len(cells) == len(TASKS) * 10, (k, len(cells))
    k = k.pop()
    mean = lambda x: sum(x) / len(x)
    cell_mean = {c: mean(v) for c, v in cells.items()}
    task_mean = {t: mean([m for c, m in cell_mean.items() if c[0] == t]) for t in TASKS}
    grand = mean(list(task_mean.values()))
    n_t, n_s = len(TASKS), 10
    ms_exec = sum((x - cell_mean[c]) ** 2 for c, v in cells.items() for x in v) / (n_t * n_s * (k - 1))
    ms_state = k * sum((cell_mean[c] - task_mean[c[0]]) ** 2 for c in cells) / (n_t * (n_s - 1))
    ms_task = k * n_s * sum((m - grand) ** 2 for m in task_mean.values()) / (n_t - 1)
    v_state = max(0.0, (ms_state - ms_exec) / k)
    v_task = max(0.0, (ms_task - ms_state) / (k * n_s))
    total = v_task + v_state + ms_exec
    return k, v_task / total, v_state / total, ms_exec / total


for name, parts in FILES.items():
    k, t, s, e = split(load(parts))
    print(f"{name}, {k} executions per state: task {t:.3f}, state within task {s:.3f}, execution {e:.3f}")
