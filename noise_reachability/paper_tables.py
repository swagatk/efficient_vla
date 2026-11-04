"""The paper's result tables, generated from committed result files (results-to-paper rule: no number typed by hand).

Writes one Markdown table per results section into --out_dir, plus tables.json with every input file's sha256, the
command and the git commit. Standard library only.

    python -S noise_reachability/paper_tables.py --out_dir noise_reachability/paper_tables
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_step7 import clopper_pearson  # noqa: E402

W6 = "week6"
INPUTS = {
    "step1": f"{W6}/noise_reachability_step1_kaggle/step1_reachability.txt.json",
    "heldout": f"{W6}/noise_reachability_step1_kaggle/heldout_headroom_no_task8.txt.json",
    "step5a": f"{W6}/noise_reachability_step5a_colab/run/step5a_report.txt.json",
    "step8": f"{W6}/noise_reachability_step8_kaggle/run/step8_report.txt.json",
    "step9": f"{W6}/noise_reachability_step9_kaggle/run/step9_report.txt.json",
}
FAMILIES = ("sampled", "full", "repeat")


def pct(lo_hi):
    return f"{100 * lo_hi[0]:.1f}-{100 * lo_hi[1]:.1f}%"


def rate(k, n, interval=None):
    return f"{k}/{n} ({pct(interval or clopper_pearson(k, n))})"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def headroom_table(step1, held):
    n = step1["n_seeds"]
    pooled = ["pooled, tasks 0, 4, 6, 7, 8", f"{step1['pass_at_k'][0]:.3f}", f"{step1['pass_at_k'][-1]:.3f}",
              f"{step1['headroom']:.3f}", "", len(step1["systematic_states"])]
    rows = [[t, f"{v['pass@1']:.3f}", f"{v[f'pass@{n}']:.3f}", "", "", len(v["systematic"])]
            for t, v in step1["per_task"].items()]
    rows.append(pooled)
    rows.append(["pooled, tasks 0, 4, 6, 7", f"{held['pass@1']:.3f}", f"{held[f'pass@{n}']:.3f}",
                 f"{held['hindsight_headroom']:.3f}", f"{held['held_out_headroom']:.3f} (p {held['null_p']:.2f})", ""])
    return table(["Task", "pass@1", f"pass@{n}", "Headroom, hindsight", "Headroom, held out", f"States at 0 of {n}"], rows)


def step5a_table(r):
    per = {f: sum(s["per_family"][f] for s in r["states"].values()) for f in FAMILIES}
    n = sum(s["episodes"] for s in r["states"].values()) // len(FAMILIES)
    rows = [[f"`{f}`", rate(per[f], n)] for f in FAMILIES]
    return table(["Family", f"Successes of {n}, states {', '.join(r['states'])}"], rows)


def step8_table(r):
    t1 = next(s for s in r["states"] if (s["task"], s["state"]) == (6, 8))
    p = r["pooled"]
    rows = [["1: 6/8", rate(t1["sampled"], t1["episodes"], t1["sampled_interval"]),
             rate(t1["repeat"], t1["episodes"], t1["repeat_interval"]), f"{r['tests']['test 1']['p']:.2g}",
             "passes" if r["tests"]["test 1"]["passes"] else "does not pass"],
            ["2: four states pooled", rate(p["sampled"], p["episodes"], p["sampled_interval"]),
             rate(p["repeat"], p["episodes"], p["repeat_interval"]), f"{r['tests']['test 2']['p']:.2g}",
             "passes" if r["tests"]["test 2"]["passes"] else "does not pass"]]
    return table(["Test", "`sampled`", "`repeat`", "p", "Holm"], rows)


def step9_table(r):
    rows = []
    for t, v in r["tasks"].items():
        rows.append([t, ", ".join(f"{t}/{s}" for s in v["states"]), f"{v['library']}/{v['library_episodes']}",
                     f"{v['random']}/{v['random_episodes']}"])
    q = r["pooled"]
    rows.append(["All", "", rate(q["library"], q["library_episodes"], q["library_interval"]),
                 rate(q["random"], q["random_episodes"], q["random_interval"])])
    return table(["Task", "Held-out states", "Earlier hits", "Fresh ids"], rows) + f"\nPermutation p {r['p']:.3g}.\n"


def build(root=ROOT):
    data = {k: json.loads((root / v).read_text()) for k, v in INPUTS.items()}
    return {
        "4.1_headroom.md": headroom_table(data["step1"], data["heldout"]),
        "4.2_step5a_families.md": step5a_table(data["step5a"]),
        "4.3_step8_tests.md": step8_table(data["step8"]),
        "4.4_step9_carry_over.md": step9_table(data["step9"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = build()
    for name, text in tables.items():
        (out / name).write_text(text)
        print(f"== {name}\n{text}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *INPUTS.values()], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    prov = {"command": "python -S noise_reachability/paper_tables.py " + " ".join(argv or sys.argv[1:]),
            "git_commit": commit, "inputs_dirty": dirty, "tables": sorted(tables),
            "inputs": {v: hashlib.sha256((ROOT / v).read_bytes()).hexdigest() for v in INPUTS.values()}}
    (out / "tables.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
