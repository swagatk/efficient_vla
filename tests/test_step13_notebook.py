"""Step 13 notebook (Colab T4): settings agree with analyze_step13.py, and every job command, parsed by
self_filtered_bc's own argument parser, passes the analysis's provenance check for its folder."""
import ast
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))

import analyze_step13 as a  # noqa: E402

SHA = "f" * 40
OUT_ROOT = "/content/drive/MyDrive/efficient_vla_runs/step13/run"


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step13_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    nb = json.loads(out.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"], nb


def cell(cells, marker):
    found = [c for c in cells if marker in c]
    assert len(found) == 1, marker
    return found[0]


def namespace(cells):
    ns = {"POLICY_PATH": "/ckpt", "RUN_ENV": {}, "JOBS": 1, "SESSION": "session2"}
    exec(cell(cells, "T_SESSION = time.time()"), ns)
    for c in cells:
        tree = ast.parse(c)
        keep = (ast.FunctionDef, ast.Assign) if "def sfbc(" in c else (ast.FunctionDef,)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, keep)], type_ignores=[]), "c", "exec"), ns)
    return ns


def parse(cmd, ns):
    import self_filtered_bc as sfbc

    argv = shlex.split(cmd.split(" && ", 1)[1])
    assert argv[:3] == [ns["PY"], "-u", f'{ns["REPO"]}/self_filtered_bc/self_filtered_bc.py']
    return vars(sfbc.parse_args(argv[3:]))


def test_cells_parse_settings_match_and_first_line(tmp_path, capsys):
    cells, nb = build(tmp_path)
    for c in cells:
        ast.parse(c)
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    ns = namespace(cells)
    assert ns["REPO_REF"] == SHA and ns["OUT_ROOT"] == OUT_ROOT and ns["TASKS"] == list(a.TASKS)
    assert {t: tuple(map(int, v.split(","))) for t, v in ns["STATES"].items()} == a.STATES
    assert ns["PARTS"] == a.PARTS and ns["POLICIES"] == a.POLICIES and ns["ADAPTER_SHA256"] == a.ADAPTER_SHA256
    assert ns["RUN_POLICIES"] == {t: a.run_policies(t) for t in a.TASKS} and "R0" not in ns["RUN_POLICIES"][7]
    assert ns["HARNESS_STATE"] == a.HARNESS_STATE and ns["HARNESS_SEEDS"] == "16-19"
    assert ns["CHECKPOINT_REVISION"].startswith(a.POLICY_REVISION)
    assert capsys.readouterr().out.strip() == (
        f"step 13 at {SHA}: tasks [0, 6, 7], 59 states, seeds 200-207, policies {a.POLICIES}, 2848 evaluation episodes "
        f"to run plus 12 harness episodes per session, stop after 20.0 h, output {OUT_ROOT}")


def test_eval_commands_pass_the_provenance_check(tmp_path):
    ns = namespace(build(tmp_path)[0])
    n = 0
    for t in a.TASKS:
        for p in a.run_policies(t):
            for j in range(len(a.PARTS)):
                st = parse(ns["eval_cmd"](p, t, j), ns)
                assert st["eval_output_dir"] == f"{OUT_ROOT}/eval/{p}/task_{t}/part{j}"
                assert (st["n_action_steps"], st["max_steps"], st["policy_path"]) == (10, 520, "/ckpt")
                a.check_settings(OUT_ROOT, Path(st["eval_output_dir"]) / "provenance.jsonl", {"settings": st})
                n += 1
    assert n == 2 * (7 + 7 + 4)
    st = parse(ns["eval_cmd"]("R1", 6, 0), ns)
    assert st["lora_dir"] == ("/content/drive/MyDrive/Alienware_Research_archives/efficient_vla/expert-iteration-step-12/"
                              "output/step12/runs/R6_1/adapter")
    assert parse(ns["eval_cmd"]("base", 0, 1), ns)["eval_policies"] == "base"
    assert sum(ns["part_size"](t, j) for t in a.TASKS for j in range(2)) == 472


def test_harness_command_and_its_folder(tmp_path):
    ns = namespace(build(tmp_path)[0])
    for t in a.TASKS:
        st = parse(ns["harness_cmd"](t), ns)
        assert st["eval_output_dir"] == f"{OUT_ROOT}/harness_pending/session2/task_{t}"
        a.check_settings(OUT_ROOT, Path(f"{OUT_ROOT}/harness/session2/task_{t}/provenance.jsonl"), {"settings": st})


def test_session_harness_moves_only_on_a_pass_and_evaluation_waits_for_it(tmp_path):
    text = "\n".join(build(tmp_path)[0])
    order = [text.index(s) for s in ('SESSION = f"session{k}"', "hashlib.sha256(p.read_bytes())",
                                     'PENDING = f"{OUT_ROOT}/harness_pending/{SESSION}"',
                                     '--gate harness_pending/{SESSION}', 'os.rename(PENDING, f"{OUT_ROOT}/harness_failed/',
                                     'os.rename(PENDING, f"{OUT_ROOT}/harness/{SESSION}")', "HARNESS_OK = True",
                                     "if HARNESS_OK and not FAILURE:", "ALL_DONE = all(", "step13_report.txt",
                                     "step13_results.zip", "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert 'raise RuntimeError(f"step 13 started at commit' in text


def test_jobs_skip_finished_parts_and_never_run_r_on_task_7(tmp_path):
    cells, _ = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    finished = f"{OUT_ROOT}/eval/N1/task_6/part1/per_episode_lora.csv"
    rows = [{"seed": str(s), "init_state": str(i)} for s in range(204, 208) for i in a.STATES[6]]
    dup = [{"seed": "200", "init_state": "30"}] * 80                      # 80 rows but one distinct episode
    ns.update(HARNESS_OK=True, FAILURE=None, read_csv=lambda p: rows if p == finished else (dup if "N2/task_0/part0" in p else []),
              run_jobs=lambda jobs, max_parallel: captured.update(jobs) or {}, failed=lambda c: {}, fail=lambda m: None,
              hours_left=lambda: 10.0, sh=lambda *x, **k: None)
    exec(cell(cells, "if HARNESS_OK and not FAILURE:\n    stop_leftovers()\n    jobs = {}"), ns)
    assert len(captured) == 36 - 1 and "eval_N1_task6_part1" not in captured and "eval_N2_task0_part0" in captured
    assert not [k for k in captured if k.startswith("eval_R") and "task7" in k]
    names = list(captured)
    assert names.index("eval_R2_task6_part0") < names.index("eval_base_task0_part1")   # part 0 of all before part 1


def test_leftover_processes_are_stopped_before_harness_and_evaluation(tmp_path):
    cells, _ = build(tmp_path)
    harness = cell(cells, 'PENDING = f"{OUT_ROOT}/harness_pending/{SESSION}"')
    evaluation = cell(cells, "if HARNESS_OK and not FAILURE:\n    stop_leftovers()")
    assert harness.index("stop_leftovers()") < harness.index("run_jobs(") and "stop_leftovers()" in evaluation
    helpers = cell(cells, "def stop_leftovers(")
    assert "pkill -f 'self_filtered_bc/self_filtered_bc[.]py'" in helpers      # [.] keeps pkill off its own shell
    assert "gate_rc := sh(" in harness and ") == 1:" in harness and "elif gate_rc != 0:" in harness


def test_last_cell_syncs_drive_then_releases_the_runtime_even_after_a_failure(tmp_path):
    cells = build(tmp_path)[0]
    download, release = cells[-2], cells[-1]
    assert 'files.download("/content/step13_results.zip")' in download and "raise" not in download
    assert "RELEASE_RUNTIME = True" in release
    order = [release.index(s) for s in ("drive.flush_and_unmount()", "time.sleep(90)", "runtime.unassign()",
                                         "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert release.index("runtime.unassign()") > release.index("if RELEASE_RUNTIME:")


def test_session_length_is_a_form_field_with_a_checked_range(tmp_path):
    import pytest

    settings = cell(build(tmp_path)[0], "T_SESSION = time.time()")
    assert 'STOP_AT_H = 20.0  #@param {type:"number"}' in settings
    for bad in ("30", "0.1"):
        with pytest.raises(AssertionError, match="choose 0.5-24 hours"):
            exec(settings.replace("STOP_AT_H = 20.0", f"STOP_AT_H = {bad}"), {})
    ns = namespace(build(tmp_path)[0])
    exec(settings.replace("STOP_AT_H = 20.0", "STOP_AT_H = 8"), ns)   # the session clock starts in this cell
    assert ns["STOP_AT_H"] == 8 and 7.99 < ns["hours_left"]() <= 8
