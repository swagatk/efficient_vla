"""Step 3b check before running: how often do the critic's and the control's 95% intervals exclude 0.5 when the
noise has no effect? 10 synthetic null datasets (tests/test_analyze_noise_signal.py::synthetic, 500 episodes x 8
decisions, success from the state only), the analysis of analyze_noise_signal.py with smaller critics.

    python -I noise_reachability/calibration/step3b_null_calibration.py   # writes step3b_null_calibration.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "noise_reachability"))
import torch  # noqa: E402

import analyze_noise_signal as ans  # noqa: E402
from test_analyze_noise_signal import synthetic  # noqa: E402

torch.set_num_threads(4)
rows = []
for seed in range(10):
    d = synthetic(0.0, n_episodes=500, n_dec=8, seed=100 + seed)
    r = ans.analyse(d, n_critics=3, n_boot=500, hidden=64, n_updates=3000)
    rows.append({"seed": seed, "critic": r["critic"], "control": r["control"], "decision": r["decision"]})
    print(seed, {k: (round(r[k]["action_auc"], 3), [round(x, 3) for x in r[k]["action_ci"]]) for k in ("critic", "control")},
          r["decision"], flush=True)
ex = lambda k: sum(ans.excludes_chance(x[k]["action_ci"]) for x in rows)  # noqa: E731
print("intervals excluding 0.5 out of 10: critic", ex("critic"), "control", ex("control"))
json.dump(rows, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
