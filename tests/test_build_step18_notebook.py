"""Step 18's notebook: built from analyze_step18's constants, every code cell parses, the commands carry each
condition's flags, and the run's order and gates are the README's."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step18 as a  # noqa: E402
import deep_noise_search as d  # noqa: E402
from noise_processes import family_label  # noqa: E402


def build(tmp_path, ref="SET_AFTER_COMMIT"):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "noise_reachability/build_step18_notebook.py", ref, str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    nb = json.loads(out.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def settings(cells):
    ns = {}
    exec(cells[0].split("def sh(cmd")[0].split("print(f\"step 18")[0].replace(
        'assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"', ""), ns)
    return ns


def test_cells_parse_and_settings_are_the_constants(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        ast.parse(c)
    ns = settings(cells)
    assert ns["STATES"] == [list(x) for x in a.STATES] and ns["N_ACTION_STEPS"] == 10
    assert ns["CHECKPOINT_REVISION"] == "a24f1f380461b665158d52acd140152c79e8b36d"
    assert ns["OUT_NAME"] == "step18" and ns["STOP_AT_H"] == 11.0 and ns["TASKS"] == [0, 4, 6, 7]
    for c, (fam, rho, sigma, n) in a.CONDITIONS.items():
        assert ns["CONDITIONS"][c] == [fam, rho, sigma]
        for t, s in a.STATES:
            from analyze_step15 import parse_ids
            ids = [k for spec in ns["BLOCKS"][c][f"{t}/{s}"] for k in parse_ids(spec)]
            assert ids == a.condition_ids(t, s, c)
            assert all(len(parse_ids(spec)) <= a.BLOCK for spec in ns["BLOCKS"][c][f"{t}/{s}"])
    assert ns["HARNESS_IDS"] == {f"{t}/{s}": a.ids_spec(a.harness_ids(t, s)) for t, s in a.STATES}


def test_flags_parse_to_the_condition_and_its_file_name(tmp_path):
    ns = settings(build(tmp_path))
    for c, flags in list(ns["FLAGS"].items()) + [(ns["HARNESS_LABEL"], ns["HARNESS_FLAGS"])]:
        args = d.parse_args(flags.split() + ["--init_state", "0", "--noise_ids", "1", "--output_dir", "x"])
        assert family_label(args.family, args.rho, args.sigma) == c
        if c in a.CONDITIONS:
            assert [args.family, args.rho, args.sigma] == list(a.CONDITIONS[c][:3])
    assert ns["HARNESS_LABEL"] == a.HARNESS_LABEL


def test_unreviewed_notebook_refuses_to_run_and_commit_must_be_full(tmp_path):
    assert 'assert REPO_REF != "SET_AFTER_COMMIT"' in build(tmp_path)[0]
    r = subprocess.run([sys.executable, "noise_reachability/build_step18_notebook.py", "abc123", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True)
    assert r.returncode != 0


def test_order_and_gates(tmp_path):
    src = "\n".join(build(tmp_path))
    order = ["CHECK_DONE = True", "check_noise_processes.py", "HARNESS_DONE, KEPT = False, []", "EPISODES_DONE = False",
             "--table"]
    pos = [src.index(m) for m in order]
    assert pos == sorted(pos)
    assert "--lora_dir" not in src and "--record_dir" not in src and "--jitter_seeds" not in src
    assert "if CHECK_DONE and not FAILURE" in src and "if HARNESS_DONE and not FAILURE" in src
    assert "if EPISODES_DONE and not FAILURE" in src
    assert "for t, s in KEPT:" in src                         # dropped states run nothing
    assert "--gate harness" in src
    assert 'f"{OUT_ROOT}/harness/session{SESSION}/task{t}_state{s}"' in src            # every session reruns it
    assert "SESSION = len(Path(OUT_ROOT, \"repo_commit.txt\").read_text().split()) - 1" in src
    # the resume cell appends this session's commit before the session index is counted
    assert src.index('f.write(head + "\\n")') < src.index("SESSION = len(")


def test_first_line_counts(tmp_path):
    first = build(tmp_path)[0]
    n = sum(len(a.condition_ids(t, s, c)) for t, s in a.STATES for c in a.CONDITIONS)
    assert n == 8 * (300 + 60 + 2) and f"{n} episodes" in first and "296 jobs" in first and "16 harness" in first
