import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step6_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def test_cells_compile_and_settings_match_the_protocol(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    text = "\n".join(cells)
    assert f'REPO_REF = "{SHA}"' in text
    assert "TASKS = [4, 6, 7, 8]" in text and "N_ACTION_STEPS = 10" in text
    assert 'EVAL_SEED_PARTS = ["16-19", "20-23", "24-27", "28-31"]' in text and 'EVAL_INIT_STATES = "0-9"' in text
    step4 = json.loads((ROOT / "week6/noise_reachability_step4_kaggle/run/arm_b_checkpoint.json").read_text())
    assert f'B_SHA256 = "{step4["model_sha256"]}"' in text


def test_the_base_and_the_harness_come_before_b(tmp_path):
    text = "\n".join(build(tmp_path))
    base = text.index('BASE_DONE = evaluate("base", POLICY_PATH)')
    gate = text.index("--harness_only", base)
    b = text.index('evaluate("B", B_PATH)')
    assert base < gate < b
    assert 'evaluate("B", B_PATH) if BASE_DONE else False' in text


def test_a_short_commit_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step6_notebook.py", "abc1234",
                        str(tmp_path / "nb.ipynb")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "full commit" in r.stderr


def test_results_are_zipped_before_a_failure_is_raised(tmp_path):
    text = "\n".join(build(tmp_path))
    assert text.index("step6_results.zip") < text.index("raise RuntimeError(FAILURE)")
    assert 'fail("the harness failed' in text and "evaluation failed" in text
