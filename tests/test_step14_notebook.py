"""Step 14 notebook: settings agree with analyze_step14.py, and every job command, parsed by self_filtered_bc's own
argument parser, passes the analysis's provenance check for its folder (the generated cells' helpers are executed)."""
import ast
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))

import analyze_step14 as a  # noqa: E402

SHA = "e" * 40
STEP10_IN = "/kaggle/input/noise-reachability-step-10/step10"
STEP12_IN = "/kaggle/input/expert-iteration-step-12/step12"
OUT_ROOT = "/kaggle/working/step14"


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step14_notebook.py", SHA, str(out)],
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
        keep = (ast.FunctionDef, ast.Assign) if ("def eval_out(" in c or "def sfbc(" in c) else (ast.FunctionDef,)
        body = [n for n in tree.body if isinstance(n, keep)]
        exec(compile(ast.Module(body=body, type_ignores=[]), "cell", "exec"), ns)
    ns.update(STEP10_IN=STEP10_IN, STEP12_IN=STEP12_IN)
    return ns


def parse(cmd, ns):
    import self_filtered_bc as sfbc

    argv = shlex.split(cmd.split(" && ", 1)[1])
    assert argv[:3] == [ns["PY"], "-u", f'{ns["REPO"]}/self_filtered_bc/self_filtered_bc.py']
    return vars(sfbc.parse_args(argv[3:]))


def check(ns, cmd, folder):
    st = parse(cmd, ns)
    rec = {"settings": st}
    if st["stage"] == "train":
        st["train_episode_sha256"] = {}
        st["train_episodes"] = []
    a.check_settings(OUT_ROOT, Path(folder) / "provenance.jsonl", rec)
    assert st["task_id"] == a.TASK and st["n_action_steps"] == 10 and st["max_steps"] == 520
    assert st["policy_path"] == "/ckpt"
    return st


def test_cells_parse_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    ns = namespace(cells)
    assert ns["REPO_REF"] == SHA and ns["OUT_NAME"] == "step14" and ns["OUT_ROOT"] == OUT_ROOT
    assert ns["TASK"] == a.TASK and ns["CHECK_STATE"] == a.CHECK_STATE and tuple(ns["SEARCHED"]) == a.SEARCHED
    assert ns["TRAIN_STEPS"] == a.L_STEPS and ns["BATCH_SIZE"] == 16 and tuple(ns["TRAIN_SEEDS"]) == a.TRAIN_SEEDS
    assert ns["HARNESS"] == {p: v[0] for p, v in a.HARNESS.items()}
    assert ns["CHECK_PARTS"] == a.CHECK_PARTS and ns["SEARCHED_PARTS"] == a.SEARCHED_PARTS
    assert ns["CHECK_POLICIES"] == a.CHECK_POLICIES and ns["SEARCHED_POLICIES"] == a.SEARCHED_POLICIES
    assert ns["ADAPTERS"] == ["L0_0", "L0_1", "L0_2"] and ns["CHECKPOINT_REVISION"].startswith(a.POLICY_REVISION)
    assert ns["STOP_AT_H"] == 11.0


def test_first_line_counts_the_episodes(tmp_path, capsys):
    namespace(build(tmp_path))
    line = capsys.readouterr().out.strip()
    assert line == (f"step 14 at {SHA}: task 0, check state 8, searched [2, 5, 6, 11], adapters ['L0_0', 'L0_1', 'L0_2'] "
                    "(2671 steps), at most 1256 episodes (8 harness, 864 on 0/8, 384 on the searched states), stop after "
                    "11.0 h, output /kaggle/working/step14")


def test_part_sizes_add_up_to_the_analysis_episodes(tmp_path):
    ns = namespace(build(tmp_path))
    for which, (states, seeds, parts, _) in a.SETS.items():
        assert sum(ns["part_size"](which, j) for j in range(len(parts))) == len(states) * len(seeds)


def test_eval_commands_pass_the_provenance_check(tmp_path):
    ns = namespace(build(tmp_path))
    n = 0
    for which, (_, _, parts, policies) in a.SETS.items():
        for policy in policies:
            for j in range(len(parts)):
                cmd = ns["eval_cmd"](which, policy, j)
                st = check(ns, cmd, ns["eval_out"](which, policy, j))
                assert st["eval_output_dir"] == f"{OUT_ROOT}/{which}/{policy}/part{j}" and st["eval_policies"] == "lora"
                n += 1
    assert n == 6 * 9 + 4 * 6
    assert parse(ns["eval_cmd"]("eval_check", "N2", 0), ns)["lora_dir"] == f"{STEP10_IN}/runs/N0_2/adapter"
    assert parse(ns["eval_cmd"]("eval_check", "R1", 0), ns)["lora_dir"] == f"{STEP12_IN}/runs/R0_1/adapter"
    assert parse(ns["eval_cmd"]("eval_searched", "L0", 0), ns)["lora_dir"] == f"{OUT_ROOT}/runs/L0_0/adapter"


def test_train_commands_pass_the_provenance_check(tmp_path):
    ns = namespace(build(tmp_path))
    for s in a.TRAIN_SEEDS:
        st = check(ns, ns["train_cmd"](f"L0_{s}"), f"{OUT_ROOT}/runs/L0_{s}")
        assert st["seed"] == s and st["train_steps"] == 2671 and st["data_dir"] == "/tmp/step14/task_0/round2_data"
        assert st["val_init_states"] == ""


def test_harness_commands_pass_the_provenance_check(tmp_path):
    cells = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    ns.update(read_csv=lambda p: [], run_stage=lambda name, jobs, total: captured.update(jobs) or False)
    exec(cell(cells, "HARNESS_DONE = False"), ns)
    assert sorted(captured) == ["harness_N0", "harness_R0"]
    for name, (cmd, gpu) in captured.items():
        policy = name.split("_")[1]
        st = check(ns, cmd, f"{OUT_ROOT}/harness/{policy}")
        assert st["eval_output_dir"] == f"{OUT_ROOT}/harness/{policy}"
    assert {captured["harness_N0"][1], captured["harness_R0"][1]} == {0, 1}


def test_evaluate_skips_finished_parts_and_unready_adapters(tmp_path):
    ns = namespace(build(tmp_path))
    captured = {}
    done_part = f"{OUT_ROOT}/eval_check/R1/part2/per_episode_lora.csv"
    ns.update(read_csv=lambda p: [{}] * 16 if p == done_part else [],
              run_stage=lambda name, jobs, total: captured.update(jobs),
              adapter_done=lambda n: n == "L0_1")
    ns["evaluate"]("NR")
    assert len(captured) == 6 * 6 + 4 * 3 - 1 and "eval_check_R1_part2" not in captured
    captured.clear()
    ns["evaluate"]("L")
    assert sorted({k.split("_")[2] for k in captured}) == ["L1"] and len(captured) == 6 + 4


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("RESUMING from", "STEP12_IN = find_input", 'HARNESS_DONE = run_gate("harness")',
                                     'evaluate("NR")', "os.symlink(f, target)", "check_step14_data.py", "TIMING = Path(",
                                     "while ready and hours_left() > TRAIN_NEEDS_H", 'evaluate("L")', "ALL_DONE = (",
                                     "--out {OUT_ROOT}/step14_report.txt", "WORK REMAINS", "step14_results.zip",
                                     "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert "ready = [n for n in ADAPTERS if not adapter_done(n)] if HARNESS_DONE and not FAILURE else []" in text
    assert 'if HARNESS_DONE and not FAILURE:\n    evaluate("NR")' in text


def test_nothing_of_earlier_steps_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("analyze_step8", "analyze_step12", "CONTROL_IDS", "SEARCH_IDS", "--family", "check_scene",
                 "step12_results.zip", "--eval_policies base", "deep_noise_search"):
        assert gone not in text, gone
