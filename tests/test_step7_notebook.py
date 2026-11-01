import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step7 as a7  # noqa: E402

SHA = "a" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step7_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def test_cells_compile_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    text = "\n".join(cells)
    assert f'REPO_REF = "{SHA}"' in text and "step 6" not in cells[0].split("def sh")[0]
    assert f"SYSTEMATIC = {a7.SYSTEMATIC}" in text and f"CONTROL = {a7.CONTROL}" in text
    assert 'NOISE_IDS = "0-69"' in text and 'CONTROL_IDS = "0-34"' in text and 'JITTER_SEEDS = "5000-5003"' in text
    assert "N_ACTION_STEPS = 10" in text and 'OUT_NAME = "step7"' in text
    assert 'CHECKPOINT_REVISION = "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"' in text.replace(
        "CHECKPOINT_REPO, CHECKPOINT_REVISION", "CHECKPOINT_REVISION")


def test_no_settings_cell_of_step_6_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("B_SHA256", "EVAL_SEED_PARTS", "self_filtered_bc.py", "LABEL_FILES", "step6"):
        assert gone not in text


def test_gates_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ('CHECK_DONE = True', 'HARNESS_DONE = CHECK_DONE and', 'CONTROL_DONE = HARNESS_DONE and',
                                     'SEARCH_DONE = CONTROL_DONE and', 'if SEARCH_DONE:', 'if CONFIRM_DONE:')]
    assert order == sorted(order)
    assert 'run_gate("harness")' in text and 'run_gate("control")' in text


def test_every_process_gets_n_action_steps_from_one_variable(tmp_path):
    text = "\n".join(build(tmp_path))
    for entry in ("deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--init_states 0-9 --n_action_steps {N_ACTION_STEPS}"):
        assert entry in text


def test_only_task_8_skips_the_environment_seed_check(tmp_path):
    assert '(" --scene_varies" if t == 8 else "")' in "\n".join(build(tmp_path))


def test_a_short_commit_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step7_notebook.py", "abc1234",
                        str(tmp_path / "nb.ipynb")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "full commit" in r.stderr


def test_results_are_zipped_before_a_failure_is_raised(tmp_path):
    text = "\n".join(build(tmp_path))
    assert text.index("step7_results.zip") < text.index("raise RuntimeError(FAILURE)")
    assert text.index("if CONFIRM_DONE:") < text.index("step7_results.zip")
