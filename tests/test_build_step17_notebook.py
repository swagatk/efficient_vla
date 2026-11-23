"""Step 17's notebook: built from analyze_step17's constants, every code cell parses, and the run's order and gate are
the README's."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step17 as a  # noqa: E402


def build(tmp_path, ref="SET_AFTER_COMMIT"):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "noise_reachability/build_step17_notebook.py", ref, str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    nb = json.loads(out.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_parse_and_settings_are_the_constants(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        ast.parse(c)
    ns = {}
    exec(cells[0].split("def sh(cmd")[0].split("print(f\"step 17")[0].replace(
        'assert REPO_REF != "SET_AFTER_COMMIT", "this notebook has no reviewed commit yet"', ""), ns)
    assert ns["TASKS"] == list(a.TASKS) and ns["MAP_STATES"] == list(a.MAP_STATES) and ns["N_ACTION_STEPS"] == 10
    assert ns["CHECKPOINT_REVISION"] == "a24f1f380461b665158d52acd140152c79e8b36d"
    assert ns["OUT_NAME"] == "step17" and ns["STOP_AT_H"] == 4.0
    ids = ns["IDS"]
    assert len(ids) == 120 and sum(v == "10000-10001" for v in ids.values()) == 4
    assert {k for k, v in ids.items() if v == "10000-10001"} == {f"{t}/{s}" for t, s in a.DUPLICATE_STATES}
    assert all(v in ("10000", "10000-10001") for v in ids.values())


def test_unreviewed_notebook_refuses_to_run(tmp_path):
    assert 'assert REPO_REF != "SET_AFTER_COMMIT"' in build(tmp_path)[0]


def test_commit_must_be_full(tmp_path):
    r = subprocess.run([sys.executable, "noise_reachability/build_step17_notebook.py", "abc123", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True)
    assert r.returncode != 0


def test_order_gate_and_command(tmp_path):
    src = "\n".join(build(tmp_path))
    order = ["CHECK_DONE = True", "FEATURES_DONE = False", "--gate features", "--features {OUT_ROOT}/step17_features.json",
             "step17_results.zip"]
    pos = [src.index(m) for m in order]
    assert pos == sorted(pos)
    assert "--family repeat" in src and "--lora_dir" not in src and "--record_dir" not in src
    assert '--output_dir {out} --feature_dir {out}/json' in src          # analyze_step17 checks this folder name
    assert "if CHECK_DONE and not FAILURE" in src and "if FEATURES_DONE and not FAILURE" in src
    assert "-x '*.npz'" in src                                            # features are JSON, so they stay in the zip
