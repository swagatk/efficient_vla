"""The paper's result tables, generated from committed result files (results-to-paper rule: no number typed by hand).

Writes one Markdown table per results section into --out_dir, plus tables.json with every input file's sha256, the
command and the git commit. Standard library only.

    python -S noise_reachability/paper_tables.py --out_dir noise_reachability/paper_tables
"""
import argparse
import hashlib
import json
import re
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
    "step10": f"{W6}/noise_reachability_step10_kaggle/run/step10_report.txt.json",
    "step11": f"{W6}/noise_reachability_step11_kaggle/run/step11_report.txt.json",
    "step10_train_0": f"{W6}/noise_reachability_step10_kaggle/run/runs/N0_0/provenance.jsonl",
    "step10_train_6": f"{W6}/noise_reachability_step10_kaggle/run/runs/N6_0/provenance.jsonl",
    "step10_train_7": f"{W6}/noise_reachability_step10_kaggle/run/runs/N7_0/provenance.jsonl",
}
# Step 10's table rows: label, role every state must have in the report, and (task, state) pairs.
STEP10_ROWS = (
    ("Trained", "trained", ((0, 2), (0, 5), (0, 6), (0, 7), (0, 8))),
    ("Trained", "trained", ((0, 11),)),
    ("Trained", "trained", ((6, 8), (6, 12))),
    ("Trained", "trained", ((7, 33),)),
    ("Held out", "held out", ((0, 14), (0, 27))),
    ("Held out", "held out", ((6, 14), (6, 20))),
    ("Held out", "held out", ((7, 38), (7, 47))),
    ("Easy", "easy", ((0, 0),)),
    ("Easy", "easy", ((0, 1), (0, 3), (0, 4), (0, 9))),
    ("Easy", "easy", ((6, 0), (7, 4))),
)
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


def step11_table(r):
    """Tests A and B of step 11: step 9's repeat arm against step 11's sampled arm, with exact intervals and Holm."""
    rows = []
    for name, label in (("test A", "A: all eight"), ("test B", "B: tasks 0, 4, 6")):
        v = r[name]
        rows.append([label, rate(v["repeat"], v["episodes"], v["repeat_interval"]),
                     rate(v["sampled"], v["episodes"], v["sampled_interval"]), f"{v['p']:.2g}",
                     "passes" if v["passes"] else "does not pass"])
    per = ", ".join(f"{s['task']}/{s['state']} {s['repeat']}-{s['sampled']}" for s in r["states"])
    return table(["Test", "`repeat` (step 9)", "`sampled`", "p", "Holm"], rows) + f"\nPer state, repeat-sampled of 50: {per}.\n"


def train_counts(provenance_line: dict) -> dict[int, int]:
    """{init state: training episodes} of one training run, from the recording names its provenance lists."""
    out = {}
    for name in provenance_line["settings"]["train_episodes"]:
        i = int(re.search(r"_init(\d+)[_.]", name).group(1))
        out[i] = out.get(i, 0) + 1
    return out


def step10_table(r, train):
    """Per group of states: N's training episodes, then successes of base, N0-N2 and A0-A2 (A on task 0's states 0-9
    only); n per policy. ``train`` is {task: {state: episodes}} (the same for N's three seeds)."""
    states = {(s["task"], s["state"]): s for s in r["states"]}
    rows = []
    for label, role, keys in STEP10_ROWS:
        assert all(states[k]["role"] == role for k in keys), (label, keys)
        n = sum(states[k]["episodes"] for k in keys)
        tot = lambda p: sum(states[k][p][0] for k in keys)
        has_a = all("A0" in states[k] for k in keys)
        rows.append([label, ", ".join(f"{t}/{s}" for t, s in keys), sum(train[t].get(s, 0) for t, s in keys), n,
                     tot("base"), ", ".join(str(tot(f"N{i}")) for i in range(3)),
                     ", ".join(str(tot(f"A{i}")) for i in range(3)) if has_a else "--"])
    t1, t2 = r["tests"]["test 1"], r["tests"]["test 2"]
    note = ("Test 1 (trained states, " + f"{t1['per_seed']['0']['n']} paired episodes): N0, N1, N2 "
            + ", ".join(str(v["first"]) for v in t1["per_seed"].values()) + f" against base {t1['per_seed']['0']['second']}, "
            + "p " + ", ".join(f"{v['p']:.1g}" for v in t1["per_seed"].values())
            + f". Test 2 (0/2, 0/5-0/8, seeds pooled, {t2['n']}): N {t2['first']} against A {t2['second']}, p {t2['p']:.1g}.")
    header = ["Role", "States", "N train", "n", "Base", "N0, N1, N2", "A0, A1, A2"]
    return table(header, rows) + "\n" + note + "\n"


def build(root=ROOT):
    data = {k: (json.loads((root / v).read_text()) if v.endswith(".json") else
                json.loads((root / v).read_text().splitlines()[0])) for k, v in INPUTS.items()}
    return {
        "4.1_headroom.md": headroom_table(data["step1"], data["heldout_all"], data["heldout"]),
        "4.2_step5a_families.md": step5a_table(data["step5a"]),
        "4.2b_search_states.md": search_states_table(data["step5a"], data["step7"]),
        "4.3_step8_tests.md": step8_table(data["step8"]),
        "3_harness.md": harness_table(data["step5a"], data["step7"], data["step8"], data["step9"]),
        "3_controls.md": control_table(data["step5a"], data["step7"], data["step8"]),
        "4.4_step9_carry_over.md": step9_table(data["step9"]),
        "4.5_step11_replication.md": step11_table(data["step11"]),
        "4.6_step10_learned.md": step10_table(data["step10"], {t: train_counts(data[f"step10_train_{t}"]) for t in (0, 6, 7)}),
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
