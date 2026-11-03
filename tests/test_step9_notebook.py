import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step9 as a9  # noqa: E402

SHA = "b" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step9_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def test_cells_compile_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    text = "\n".join(cells)
    assert f'REPO_REF = "{SHA}"' in text
    assert f"TASKS = {list(a9.TASKS)}" in text
    assert f"HELD_OUT = { {t: list(v) for t, v in a9.HELD_OUT.items()} }" in text
    assert "N_ACTION_STEPS = 10" in text and 'OUT_NAME = "step9"' in text
    assert '"a24f1f380461b665158d52acd140152c79e8b36d"' in text
    assert "analyze_step9.py --root {OUT_ROOT}" in text


def test_nothing_of_step_8_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("step8", "analyze_step8", "CONTROL", "SCREEN", "TEST1", "JITTER", '"sampled"', "--hits", "--order"):
        assert gone not in text, gone


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("CHECK_DONE = True", "HARNESS_DONE = CHECK_DONE and", "SCENES_DONE = HARNESS_DONE",
                                     'listed("--jobs") if SCENES_DONE', "if TESTS_DONE:", "step9_results.zip",
                                     "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)


def test_jobs_run_the_repeat_family_into_one_folder_per_arm(tmp_path):
    text = "\n".join(build(tmp_path))
    assert 'out = f"{task_dir(t)}/test/{arm}"' in text
    assert "--family repeat --init_state {s} --noise_ids {ids} --output_dir {out}" in text


def test_every_process_gets_n_action_steps_from_one_variable(tmp_path):
    text = "\n".join(build(tmp_path))
    for entry in ("deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "check_scene.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--init_states 0-9 --n_action_steps {N_ACTION_STEPS}"):
        assert entry in text


def test_resume_refuses_two_inputs(tmp_path):
    text = "\n".join(build(tmp_path))
    assert "assert len(previous) <= 1" in text and "FRESH START" in text and "RESUMING from" in text


def test_a_short_commit_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step9_notebook.py", "abc1234",
                        str(tmp_path / "nb.ipynb")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "full commit" in r.stderr
