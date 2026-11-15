"""Step 14, before training: does the linked data folder give exactly R0_s's training episodes? (README.md, step 14)

    python noise_reachability/check_step14_data.py <data_dir>      # exit 1 if any seed differs

self_filtered_bc's own selection (``split_episodes`` with ``--selection success`` and no validation states) on the folder
the notebook links for L0_s must give the episode names step 12's committed R0_s provenance lists, for every training
seed. Catches a wrong or older input version before hours of training. Needs numpy (reads the recordings' metadata).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "self_filtered_bc"))
sys.path.insert(0, str(REPO / "noise_reachability"))

import analyze_step14 as a  # noqa: E402
from analyze_step12 import train_episodes  # noqa: E402


def check(data_dir, step12=a.STEP12) -> dict[int, bool]:
    import self_filtered_bc as sfbc

    out = {}
    for s in a.TRAIN_SEEDS:
        got = [p.name for p in sfbc.split_episodes(data_dir, "success", set(), s)[0]]
        want = train_episodes(Path(step12, "runs", f"R{a.TASK}_{s}", "provenance.jsonl"))
        out[s] = got == want
        print(f"seed {s}: {len(got)} episodes selected, R{a.TASK}_{s} trained on {len(want)}: "
              f"{'the same' if out[s] else 'DIFFERENT'}", flush=True)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    return 0 if all(check(argv[0]).values()) else 1


if __name__ == "__main__":
    sys.exit(main())
