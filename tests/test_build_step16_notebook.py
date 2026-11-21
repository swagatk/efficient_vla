"""Step 16's notebook: built from analyze_step16's constants, every code cell parses, and the run's order and gates
are the README's."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step16 as a  # noqa: E402


def build(tmp_path, ref="SET_AFTER_COMMIT"):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "noise_reachability/build_step16_notebook.py", ref, str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    nb = json.loads(out.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_parse_and_settings_are_the_constants(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        ast.parse(c)
    settings = cells[0]
    ns = {}
    exec(settings.split("def sh(cmd")[0].split("print(f\"step 16")[0].replace(
        'assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"', ""), ns)
    assert ns["TASKS"] == list(a.TASKS) and ns["MAP_STATES"] == list(a.MAP_STATES) and ns["PARTS"] == list(a.PARTS)
    assert ns["N_ACTION_STEPS"] == 10 and ns["HARNESS_IDS"] == a.ids_spec(a.HARNESS_IDS)
    assert (ns["HARNESS_TASK"], ns["HARNESS_STATE"]) == (a.HARNESS_TASK, a.HARNESS_STATE)
    assert ns["CHECKPOINT_REVISION"] == "a24f1f380461b665158d52acd140152c79e8b36d"
    assert ns["OUT_NAME"] == "step16" and ns["STOP_AT_H"] == 11.0


def test_unreviewed_notebook_refuses_to_run(tmp_path):
    assert 'assert REPO_REF != "SET_AFTER_COMMIT"' in build(tmp_path)[0]


def test_commit_must_be_full(tmp_path):
    r = subprocess.run([sys.executable, "noise_reachability/build_step16_notebook.py", "abc123", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True)
    assert r.returncode != 0


def test_order_and_gates(tmp_path):
    src = "\n".join(build(tmp_path))
    order = ["CHECK_DONE = True", "SCENES_DONE = False", "HARNESS_DONE = False", "PARTS_DONE = 0", "--table"]
    pos = [src.index(m) for m in order]
    assert pos == sorted(pos)
    assert "--family repeat" in src and "--lora_dir" not in src and "--record_dir" not in src
    assert "if j == 1 and not part1_gate_passes()" in src           # parts 2-5 only after part 1's gate
    assert 'HARNESS_DONE = run_gate("harness")' in src
    assert "if SCENES_DONE and not FAILURE" in src and "if HARNESS_DONE and not FAILURE" in src
