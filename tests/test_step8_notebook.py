import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step8 as a8  # noqa: E402
import check_scene as c  # noqa: E402

SHA = "a" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step8_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def test_cells_compile_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    text = "\n".join(cells)
    assert f'REPO_REF = "{SHA}"' in text
    assert f"TASKS = {list(a8.TASKS)}" in text and f"CONTROL = {a8.CONTROL}" in text
    assert f"SCREEN_TASKS = {list(a8.SCREEN_TASKS)}" in text and "SCREEN_STATES = range(10, 30)" in text
    assert f"TEST1 = {a8.TEST1}" in text and "DIRECT = {7: [33, 38, 47]}" in text
    for name, ids in (("CONTROL_IDS", a8.CONTROL_IDS), ("FIRST_IDS", a8.FIRST_IDS), ("SECOND_IDS", a8.SECOND_IDS),
                      ("ALL_SCREEN_IDS", a8.ALL_SCREEN_IDS)):
        assert f'{name} = "{ids[0]}-{ids[-1]}"' in text
    assert 'JITTER_SEEDS = "5000-5003"' in text and "N_ACTION_STEPS = 10" in text and 'OUT_NAME = "step8"' in text
    assert f'"{a8.POLICY_REVISION}"' in text


def test_nothing_of_step_7_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("step7", "analyze_step7", "SYSTEMATIC", "--scene_varies", '"full"'):
        assert gone not in text


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("CHECK_DONE = True", "HARNESS_DONE = CHECK_DONE and", "CONTROL_DONE = HARNESS_DONE and",
                                     "if CONTROL_DONE:", "if SCREEN_DONE:", "if SELECT_DONE else None", "if TESTS_DONE:",
                                     "if CONFIRM_DONE:")]
    assert order == sorted(order)


def test_every_process_gets_n_action_steps_from_one_variable(tmp_path):
    text = "\n".join(build(tmp_path))
    for entry in ("deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "check_scene.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--init_states 0-9 --n_action_steps {N_ACTION_STEPS}"):
        assert entry in text


def test_results_are_zipped_before_a_failure_is_raised(tmp_path):
    text = "\n".join(build(tmp_path))
    assert text.index("if CONFIRM_DONE:") < text.index("step8_results.zip") < text.index("raise RuntimeError(FAILURE)")


def test_a_short_commit_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step8_notebook.py", "abc1234",
                        str(tmp_path / "nb.ipynb")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "full commit" in r.stderr


def test_the_scene_noise_is_in_no_test():
    assert (c.SCENE_NOISE_ID not in a8.TEST1_IDS and c.SCENE_NOISE_ID not in a8.ALL_SCREEN_IDS
            and all(c.SCENE_NOISE_ID not in a8.test2_ids(i) for i in range(a8.MAX_SELECTED)))


def test_moved_bodies_use_the_tolerance_on_any_coordinate():
    a = {"stove": [0.0, 0.0, 1.0], "table": [1.0, 2.0, 3.0]}
    assert c.moved_bodies(a, {k: list(v) for k, v in a.items()}) == {}
    assert c.moved_bodies(a, dict(a, stove=[0.0, 0.0, 1.0 + 9e-7])) == {}
    assert c.moved_bodies(a, dict(a, stove=[0.0, -0.0124, 1.0])) == {"stove": 0.0124}
    assert list(c.moved_bodies(a, dict(a, stove=[0.0, 0.0, 1.0 + 2e-6]))) == ["stove"]


def test_a_body_in_one_scene_only_counts_as_moved():
    assert c.moved_bodies({"a": [0.0]}, {"b": [0.0]}) == {"a": float("inf"), "b": float("inf")}


def test_action_difference_finds_size_and_first_step():
    assert c.action_difference([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.5]]) == {"max": 0.5, "first_step": 1, "lengths": [2, 2]}
    assert c.action_difference([[1.0]], [[1.0]]) == {"max": 0.0, "first_step": None, "lengths": [1, 1]}


def test_resume_refuses_two_inputs_and_says_fresh_start(tmp_path):
    text = "\n".join(build(tmp_path))
    assert "assert len(previous) <= 1" in text and "FRESH START" in text and "RESUMING from" in text
