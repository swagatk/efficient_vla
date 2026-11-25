"""Step 18's states (README.md, step 18, "States"), chosen by a rule from committed base-policy episodes before any
step-18 episode.

Pool: tasks 0, 4, 6 and 7; a state is a candidate if the committed base-policy runs give it at least MIN_EPISODES
`sampled` episodes with a pooled rate under MAX_SAMPLED, and at least MIN_EPISODES `repeat` episodes on fresh ids
(ids searched on that state, not carried from another state or replayed). Per task, the PER_TASK candidates with the
highest pooled `repeat` rate are taken (ties: more episodes, then the lower state).

Sources (base policy only; adapters, replays, harness reruns, pilots, step 9's library arm and step 17's repeats of
step 16 are left out):
    python3 -S noise_reachability/calibration/step18_states.py [--week6 week6]
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = (0, 4, 6, 7)
MIN_EPISODES = 50
MAX_SAMPLED = 0.01
PER_TASK = 2
S = "noise_reachability_step"
REPEAT = (f"{S}5a_colab/run/per_episode_repeat_state*.csv",
          f"{S}7_kaggle/run/task_*/per_episode_repeat_state*.csv",
          f"{S}8_kaggle/run/task_*/control/per_episode_repeat_state*.csv",
          f"{S}8_kaggle/run/task_*/test/per_episode_repeat_state*.csv",
          f"{S}9_kaggle/run/task_*/test/random/per_episode_repeat_state*.csv",
          f"{S}15_kaggle/run/task_6/search/part*/per_episode_repeat_state*.csv",
          f"{S}15_kaggle/run/task_7/search/base/per_episode_repeat_state*.csv",
          f"{S}16_kaggle/run/task_*/map/part*/per_episode_repeat_state*.csv")
SAMPLED = (f"{S}5a_colab/run/per_episode_sampled_state*.csv",
           f"{S}7_kaggle/run/task_*/per_episode_sampled_state*.csv",
           f"{S}8_kaggle/run/task_*/control/per_episode_sampled_state*.csv",
           f"{S}8_kaggle/run/task_*/screen/per_episode_sampled_state*.csv",
           f"{S}8_kaggle/run/task_*/test/per_episode_sampled_state*.csv",
           f"{S}11_kaggle/run/task_*/test/sampled/per_episode_sampled_state*.csv")


def tally(week6: Path, patterns) -> dict:
    """{(task, state): [successes, episodes]} over the files; every row is one episode."""
    out = defaultdict(lambda: [0, 0])
    for pat in patterns:
        files = sorted(glob.glob(str(week6 / pat)))
        assert files, pat
        for f in files:
            for row in csv.DictReader(open(f)):
                k = (int(row["task_id"]), int(row["init_state"]))
                out[k][0] += int(row["success"])
                out[k][1] += 1
    return dict(out)


def candidates(repeat: dict, sampled: dict) -> list:
    rows = []
    for k in sorted(set(repeat) & set(sampled)):
        (rs, rn), (ss, sn) = repeat[k], sampled[k]
        if k[0] in TASKS and rn >= MIN_EPISODES and sn >= MIN_EPISODES and ss / sn < MAX_SAMPLED:
            rows.append((k, rs, rn, ss, sn))
    return rows


def select(rows) -> list:
    chosen = []
    for t in TASKS:
        mine = sorted((r for r in rows if r[0][0] == t), key=lambda r: (-r[1] / r[2], -r[2], r[0][1]))
        chosen += mine[:PER_TASK]
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week6", default=str(ROOT / "week6"))
    args = ap.parse_args(argv)
    rows = candidates(tally(Path(args.week6), REPEAT), tally(Path(args.week6), SAMPLED))
    chosen = select(rows)
    print(f"candidates (>= {MIN_EPISODES} episodes of each, sampled under {MAX_SAMPLED:.0%}): {len(rows)}")
    for k, rs, rn, ss, sn in rows:
        mark = "  chosen" if any(c[0] == k for c in chosen) else ""
        print(f"  {k[0]}/{k[1]}: repeat {rs}/{rn} ({rs / rn:.3f}), sampled {ss}/{sn}{mark}")
    print("chosen: " + ", ".join(f"{k[0]}/{k[1]}" for k, *_ in chosen))


if __name__ == "__main__":
    main()
