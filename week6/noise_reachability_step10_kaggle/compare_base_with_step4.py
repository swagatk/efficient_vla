"""Harness check: step 10's base on task 0 against step 4's base on the same episodes (both Kaggle T4, sampled noise,
hard resets, states 0-9 x seeds 16-31). Per state: successes in each run and episodes identical in success and steps.

    python -I -S week6/noise_reachability_step10_kaggle/compare_base_with_step4.py --out <txt>
"""
import argparse
import csv
import glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP4 = HERE.parent / "noise_reachability_step4_kaggle/run/eval/per_episode_base.csv"
STEP10 = HERE / "run"


def episodes(paths, task=0):
    """{(seed, init state): (success, steps)} of one task over the files; refuses an episode twice."""
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            if int(r["task_id"]) != task:
                continue
            key = (int(r["seed"]), int(r["init_state"]))
            if key in out:
                raise ValueError(f"{p}: episode {key} twice")
            out[key] = (int(r["success"]), int(r["steps"]))
    return out


def compare(old, new):
    """Per init state over the shared episodes: [old successes, new successes, identical episodes, shared]."""
    per = {}
    for s, i in sorted(set(old) & set(new)):
        c = per.setdefault(i, [0, 0, 0, 0])
        c[0] += old[(s, i)][0]
        c[1] += new[(s, i)][0]
        c[2] += old[(s, i)] == new[(s, i)]
        c[3] += 1
    return per


def report(step4=STEP4, step10=STEP10):
    old = episodes([step4])
    new = episodes(sorted(glob.glob(str(Path(step10) / "task_0/eval/base/part*/per_episode_base.csv"))))
    per = compare(old, new)
    lines = [f"step 4 base: {len(old)} task-0 episodes; step 10 base: {len(new)}; shared {sum(c[3] for c in per.values())}"]
    lines += [f"state {i}: step 4 {c[0]}/{c[3]}, step 10 {c[1]}/{c[3]}, identical (success and steps) {c[2]}/{c[3]}"
              for i, c in sorted(per.items())]
    lines.append(f"total: step 4 {sum(c[0] for c in per.values())}, step 10 {sum(c[1] for c in per.values())}, "
                 f"identical {sum(c[2] for c in per.values())} of {sum(c[3] for c in per.values())}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text = report()
    print(text, end="")
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
