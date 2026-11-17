"""Step 15 notebook: settings agree with analyze_step15.py, and every job command, parsed by its script's own argument
parser, passes the analysis's provenance check for its folder (the generated cells' helpers are executed)."""
import ast
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))

import analyze_step15 as a  # noqa: E402

SHA = "e" * 40
STEP10_IN = "/kaggle/input/noise-reachability-step-10/step10"
OUT_ROOT = "/kaggle/working/step15"


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step15_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def cell(cells, marker):
    found = [c for c in cells if marker in c]
    assert len(found) == 1, marker
    return found[0]


def namespace(cells):
    ns = {"POLICY_PATH": "/ckpt", "RUN_ENV": {}}
    exec(cell(cells, "T_SESSION = time.time()"), ns)
    for c in cells:
        tree = ast.parse(c)
        keep = (ast.FunctionDef, ast.Assign) if "def search_cmd(" in c else (ast.FunctionDef,)
        body = [n for n in tree.body if isinstance(n, keep)]
        exec(compile(ast.Module(body=body, type_ignores=[]), "cell", "exec"), ns)
    ns.update(STEP10_IN=STEP10_IN)
    return ns


def parse(cmd, ns):
    argv = shlex.split(cmd.split(" && ", 1)[1])
    assert argv[:2] == [ns["PY"], "-u"]
    if argv[2] == "noise_reachability/deep_noise_search.py":
        import deep_noise_search

        return vars(deep_noise_search.parse_args(argv[3:]))
    import self_filtered_bc as sfbc

    assert argv[2] == f'{ns["REPO"]}/self_filtered_bc/self_filtered_bc.py'
    return vars(sfbc.parse_args(argv[3:]))


def check(ns, cmd, folder):
    st = parse(cmd, ns)
    a.check_settings(OUT_ROOT, Path(folder) / "provenance.jsonl", {"settings": st})
    assert st["n_action_steps"] == 10 and st["max_steps"] == 520 and st["policy_path"] == "/ckpt"
    out = st.get("eval_output_dir") or st["output_dir"]
    assert out == folder, (out, folder)
    return st


def test_settings_match_the_analysis(tmp_path):
    ns = namespace(build(tmp_path))
    assert ns["REPO_REF"] == SHA and ns["OUT_NAME"] == "step15" and ns["OUT_ROOT"] == OUT_ROOT
    assert (ns["TASK_A"], tuple(ns["STATES_A"]), tuple(ns["A_PARTS"])) == (a.TASK_A, a.STATES_A, a.A_PARTS)
    assert (ns["TASK_B"], ns["STATE_B"], ns["B_IDS"], ns["B_POLICIES"]) == (a.TASK_B, a.STATE_B, "9000-9199",
                                                                             a.B_POLICIES)
    assert ns["SCENE_STATES"] == dict(a.SCENE_STATES) and ns["N7_SHA256"] == a.N7_SHA256
    assert ns["HARNESS_BASE_IDS"] == a.ids_spec(a.HARNESS_BASE_IDS) and ns["TASKS"] == [6, 7]
    assert ns["CHECKPOINT_REVISION"].startswith(a.POLICY_REVISION) and ns["STOP_AT_H"] == 11.0


def test_first_line_counts_the_episodes(tmp_path, capsys):
    namespace(build(tmp_path))
    line = capsys.readouterr().out.strip()
    assert line == (f"step 15 at {SHA}: test A 6/[14, 20] ids ['9000-9199', '9200-9399'], test B 7/38 ids 9000-9199 for "
                    "['base', 'N0', 'N1', 'N2'], at most 1614 episodes (14 harness, 800 test A, 800 test B) plus replays, "
                    "stop after 11.0 h, output /kaggle/working/step15")


def test_n_ids_counts_specs():
    ns = namespace(build(Path(__import__("tempfile").mkdtemp())))
    assert ns["n_ids"]("9000-9199") == 200 and ns["n_ids"](a.ids_spec(a.HARNESS_BASE_IDS)) == 10
    assert ns["n_ids"]("16-19") == 4 and ns["n_ids"]("9123") == 1 and ns["n_ids"]("9001-9002,9250") == 3


def test_harness_commands_pass_the_provenance_check(tmp_path):
    cells = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    ns.update(read_csv=lambda p: [], SCENES_DONE=True, FAILURE=None,
              run_stage=lambda name, jobs, total: captured.update(jobs) or False)
    exec(cell(cells, "HARNESS_DONE = False"), ns)
    assert sorted(captured) == ["harness_N0", "harness_base"]
    st = check(ns, captured["harness_base"][0], f"{OUT_ROOT}/harness/base")
    assert st["lora_dir"] is None and st["noise_ids"] == a.ids_spec(a.HARNESS_BASE_IDS)
    st = check(ns, captured["harness_N0"][0], f"{OUT_ROOT}/harness/N0")
    assert st["lora_dir"] == f"{STEP10_IN}/runs/N7_0/adapter" and st["eval_policies"] == "lora"
    assert {captured["harness_base"][1], captured["harness_N0"][1]} == {0, 1}


def test_test_jobs_pass_the_provenance_check_and_skip_failed_scenes(tmp_path):
    ns = namespace(build(tmp_path))
    ns.update(read_csv=lambda p: [], scene_passed=lambda t, s: True)
    jobs = ns["test_jobs"]()
    assert len(jobs) == 8 and sorted(g for _, g in jobs.values()) == [0, 0, 0, 0, 1, 1, 1, 1]
    for name, (cmd, _) in jobs.items():
        if name.startswith("testA"):
            s, j = int(name.split("_")[1].removeprefix("state")), int(name[-1])
            st = check(ns, cmd, f"{OUT_ROOT}/task_6/search/part{j}")
            assert st["init_state"] == s and st["noise_ids"] == a.A_PARTS[j] and st["lora_dir"] is None
        else:
            policy = name.split("_")[1]
            st = check(ns, cmd, f"{OUT_ROOT}/task_7/search/{policy}")
            assert st["noise_ids"] == "9000-9199" and st["init_state"] == 38
            assert st["lora_dir"] == (None if policy == "base" else f"{STEP10_IN}/runs/N7_{policy[1]}/adapter")
    ns.update(scene_passed=lambda t, s: (t, s) != (6, 20) and t != 7)
    assert sorted(ns["test_jobs"]()) == ["testA_state14_part0", "testA_state14_part1"]


def test_test_jobs_skip_finished_files(tmp_path):
    ns = namespace(build(tmp_path))
    done = f"{OUT_ROOT}/task_7/search/N1/per_episode_repeat_state38.csv"
    ns.update(read_csv=lambda p: [{}] * 200 if p == done else [{}] * 199, scene_passed=lambda t, s: True)
    assert sorted(ns["test_jobs"]()) == sorted(
        ["testA_state14_part0", "testA_state14_part1", "testA_state20_part0", "testA_state20_part1",
         "testB_base", "testB_N0", "testB_N2"])


def test_replay_commands_pass_the_provenance_check(tmp_path):
    cells = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    ns.update(read_csv=lambda p: [], TESTS_DONE=True, FAILURE=None,
              listed=lambda flag: [["6", "14", "9001-9002,9250"], ["6", "20", "9399"]],
              run_stage=lambda name, jobs, total: captured.update(jobs) or True)
    src = cell(cells, "REPLAYS_DONE = False")
    exec(src[src.index("REPLAYS_DONE = False"):], ns)
    assert sorted(captured) == ["replay_state14", "replay_state20"] and ns["REPLAYS_DONE"]
    st = check(ns, captured["replay_state14"][0], f"{OUT_ROOT}/task_6/replay")
    assert st["noise_ids"] == "9001-9002,9250" and st["init_state"] == 14


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("RESUMING from", "STEP10_IN = found[0]", "sha256 equal to step 10's", "CHECK_DONE = True",
                                     "SCENES_DONE = CHECK_DONE", 'HARNESS_DONE = run_gate("harness")', "def test_jobs(",
                                     'run_stage("tests A and B"', 'listed("--replay_jobs")', "step15_report.txt",
                                     "WORK REMAINS", "step15_results.zip", "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert "if SCENES_DONE and not FAILURE:\n    jobs = {}" in text
    assert "if HARNESS_DONE and not FAILURE:\n    TESTS_DONE = run_stage" in text
    assert 'plan = listed("--replay_jobs") if TESTS_DONE and not FAILURE else None' in text
    assert "if REPLAYS_DONE and not FAILURE:" in text


def test_nothing_of_earlier_steps_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("analyze_step8", "analyze_step14", "CONTROL_IDS", "SEARCH_IDS", "--family sampled", "step14_results.zip",
                 "--eval_policies base", "--record_dir", "--jitter_seeds", "--stage train", "step12"):
        assert gone not in text, gone
