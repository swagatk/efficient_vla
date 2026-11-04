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
from analyze_step5a import CONTROL_STATE as CONTROL_5A, HARNESS_REFERENCE as REFERENCE_5A  # noqa: E402
from analyze_step7 import CONTROL as CONTROL_7, clopper_pearson  # noqa: E402
from analyze_step8 import CONTROL as CONTROL_8  # noqa: E402

W6 = "week6"
INPUTS = {
    "step1": f"{W6}/noise_reachability_step1_kaggle/step1_reachability.txt.json",
    "heldout_all": f"{W6}/noise_reachability_step1_kaggle/heldout_headroom.txt.json",
    "heldout": f"{W6}/noise_reachability_step1_kaggle/heldout_headroom_no_task8.txt.json",
    "step5a": f"{W6}/noise_reachability_step5a_colab/run/step5a_report.txt.json",
    "step7": f"{W6}/noise_reachability_step7_kaggle/run/step7_report.txt.json",
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


def tex_cell(c) -> str:
    s = str(c)
    parts = s.split("`")
    s = "".join(f"\\texttt{{{p}}}" if i % 2 else p for i, p in enumerate(parts))
    return s.replace("%", "\\%")


def to_tex(md: str) -> str:
    """A Markdown table (and any trailing note line) from ``table`` as a booktabs tabular."""
    lines = md.strip("\n").split("\n")
    rows = [l for l in lines if l.startswith("|")]
    note = [l for l in lines if l and not l.startswith("|")]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows if not r.startswith("| ---") and "--- |" not in r]
    header, body = cells[0], cells[1:]
    out = ["\\begin{tabular}{" + "l" * len(header) + "}", "\\toprule",
           " & ".join(tex_cell(c) for c in header) + " \\\\", "\\midrule"]
    out += [" & ".join(tex_cell(c) for c in r) + " \\\\" for r in body]
    out += ["\\bottomrule", "\\end{tabular}"]
    out += [f"\\par\\smallskip{{\\footnotesize {tex_cell(n)}}}" for n in note]
    return "\n".join(out) + "\n"


def headroom_table(step1, held_all, held):
    """Per task and pooled: pass@1, the seed chosen on the task's other states, pass@n, states at 0 of n."""
    n = step1["n_seeds"]
    rows = [[t, f"{v['pass@1']:.3f}", f"{held_all['per_task'][t]['held_out_choice']:.3f}", f"{v[f'pass@{n}']:.3f}",
             len(v["systematic"])] for t, v in step1["per_task"].items()]
    rows.append(["0, 4, 6, 7, 8", f"{step1['pass_at_k'][0]:.3f}", f"{held_all['held_out_choice']:.3f}",
                 f"{step1['pass_at_k'][-1]:.3f}", len(step1["systematic_states"])])
    rows.append(["0, 4, 6, 7", f"{held['pass@1']:.3f}", f"{held['held_out_choice']:.3f}", f"{held[f'pass@{n}']:.3f}",
                 sum(len(step1["per_task"][t]["systematic"]) for t in held["per_task"])])
    note = (f"Headroom (pass@{n} - pass@1) / (1 - pass@1): tasks 0, 4, 6, 7, 8 {step1['headroom']:.3f} in hindsight, "
            f"{held_all['held_out_headroom']:.3f} held out (p {held_all['null_p']:.2f}); tasks 0, 4, 6, 7 "
            f"{held['hindsight_headroom']:.3f} and {held['held_out_headroom']:.3f} (p {held['null_p']:.2f}).")
    return table(["Task", "pass@1", "Held out", f"pass@{n}", f"At 0 of {n}"], rows) + "\n" + note + "\n"


def step5a_table(r):
    per = {f: sum(s["per_family"][f] for s in r["states"].values()) for f in FAMILIES}
    n = sum(s["episodes"] for s in r["states"].values()) // len(FAMILIES)
    rows = [[f"`{f}`", rate(per[f], n)] for f in FAMILIES]
    return table(["Family", f"Successes of {n}, states {', '.join(r['states'])}"], rows)


def search_states_table(r5a, r7):
    """Per labelled state of the deep searches (steps 5a and 7): successes of each family's ids, and the jitter class."""
    rows = []
    for step, r, task_of in (("5a", r5a, lambda k, s: 0), ("7", r7, lambda k, s: int(k))):
        for k, s in r["states"].items():
            state = s.get("state", k)
            n = s["episodes"] // len(FAMILIES)
            rows.append([f"{task_of(k, s)}/{state}", step] + [f"{s['per_family'][f]}/{n}" for f in FAMILIES] + [s["class"]])
    return table(["State", "Step", "`sampled`", "`full`", "`repeat`", "Jitter"], rows)


def harness_table(r5a, r7, r8, r9):
    """The base's successes of 30 (states 0-9 x seeds 0-2) at the start of each run, against step 0's reference."""
    runs = {"5a": {"0": {"base": r5a["harness"]["base"], "reference": REFERENCE_5A}}, "7": r7["harness"],
            "8": r8["harness"], "9": r9["harness"]}
    tasks = sorted({t for h in runs.values() for t in h}, key=int)
    rows = []
    for t in tasks:
        ref = {h[t]["reference"] for h in runs.values() if t in h}
        assert len(ref) == 1, t
        rows.append([t, ref.pop()] + [runs[s][t]["base"] if t in runs[s] else "" for s in runs])
    return table(["Task", "Step 0"] + [f"Step {s}" for s in runs], rows)


def control_table(r5a, r7, r8):
    """Easy control states, successes of 35 noise ids per family, as each run's gate read them."""
    rows = [["5a", f"0/{CONTROL_5A}"] + [r5a["control"][f] for f in FAMILIES]]
    for step, r, states in (("7", r7, CONTROL_7), ("8", r8, CONTROL_8)):
        for t, s in states.items():
            c = r["control"][str(t)]
            rows.append([step, f"{t}/{s}"] + [c.get(f, "") for f in FAMILIES])
    return table(["Step", "State", "`sampled`", "`full`", "`repeat`"], rows)


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
        "4.1_headroom.md": headroom_table(data["step1"], data["heldout_all"], data["heldout"]),
        "4.2_step5a_families.md": step5a_table(data["step5a"]),
        "4.2b_search_states.md": search_states_table(data["step5a"], data["step7"]),
        "4.3_step8_tests.md": step8_table(data["step8"]),
        "3_harness.md": harness_table(data["step5a"], data["step7"], data["step8"], data["step9"]),
        "3_controls.md": control_table(data["step5a"], data["step7"], data["step8"]),
        "4.4_step9_carry_over.md": step9_table(data["step9"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tex_dir", help="also write each table as a booktabs tabular (<name>.tex) here")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = build()
    for name, text in tables.items():
        (out / name).write_text(text)
        print(f"== {name}\n{text}")
        if a.tex_dir:
            Path(a.tex_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.tex_dir) / name.replace(".md", ".tex")).write_text(to_tex(text))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *INPUTS.values()], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    prov = {"command": "python -S noise_reachability/paper_tables.py " + " ".join(argv or sys.argv[1:]),
            "git_commit": commit, "inputs_dirty": dirty, "tables": sorted(tables),
            "inputs": {v: hashlib.sha256((ROOT / v).read_bytes()).hexdigest() for v in INPUTS.values()}}
    (out / "tables.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
