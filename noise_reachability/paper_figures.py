"""The paper's figures from committed result files.

fig_step1_map: step 1's successes of 16 seeds per initial state (tasks 0, 4, 6, 7, 8 x states 0-9).
fig_step8_states: per resistant state of step 8, sampled and repeat success with the exact 95% intervals its report
computed; n per point is in the label.

    python noise_reachability/paper_figures.py --out <dir>      # writes both as .pdf and .png
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEP8 = ROOT / "week6/noise_reachability_step8_kaggle/run/step8_report.txt.json"
STEP1 = ROOT / "week6/noise_reachability_step1_kaggle/step1_reachability.txt.json"
SERIES = (("sampled", "#2a78d6", "o"), ("repeat", "#eb6834", "s"))   # categorical slots 1, 2; shape as second cue


def points(report: dict) -> list[dict]:
    """One row per state in the report's order: label, n, and per family (rate, low, high) as fractions."""
    out = []
    for s in report["states"]:
        row = {"label": f"{s['task']}/{s['state']}\nn={s['episodes']}", "n": s["episodes"]}
        for fam, _, _ in SERIES:
            row[fam] = (s[fam] / s["episodes"], *s[f"{fam}_interval"])
        out.append(row)
    return out


def state_grid(step1: dict) -> tuple[list[int], list[list[int]]]:
    """Tasks in order and, per task, successes of n seeds for states 0-9."""
    tasks = sorted({int(k.split(",")[0]) for k in step1["counts"]})
    return tasks, [[step1["counts"][f"{t},{s}"] for s in range(10)] for t in tasks]


def draw_map(step1: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    n = step1["n_seeds"]
    tasks, grid = state_grid(step1)
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=n, aspect="auto")
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            zero = c == 0
            ax.text(j, i, "0" if zero else str(c), ha="center", va="center", fontsize=7,
                    color="#b3261e" if zero else ("white" if c > 0.6 * n else "#222222"),
                    fontweight="bold" if zero else "normal")
    ax.set_xticks(range(10), [str(s) for s in range(10)])
    ax.set_yticks(range(len(tasks)), [f"task {t}" for t in tasks])
    ax.set_xlabel("initial state")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.3)
    return save(fig, out_dir, "fig_step1_map")


def save(fig, out_dir: Path, name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)
    return paths


def draw(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for j, (fam, color, marker) in enumerate(SERIES):
        xs = [i + (j - 0.5) * 0.28 for i in range(len(rows))]
        rate = [100 * r[fam][0] for r in rows]
        lo = [100 * (r[fam][0] - r[fam][1]) for r in rows]
        hi = [100 * (r[fam][2] - r[fam][0]) for r in rows]
        ax.errorbar(xs, rate, yerr=[lo, hi], fmt=marker, color=color, ms=4, lw=1.2, capsize=2, label=fam)
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_ylabel("success (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    return save(fig, out_dir, "fig_step8_states")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = points(json.loads(STEP8.read_text()))
    for p in draw(rows, Path(a.out)) + draw_map(json.loads(STEP1.read_text()), Path(a.out)):
        print("wrote", p)


if __name__ == "__main__":
    main()
