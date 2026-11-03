"""Step 10 notebook: settings agree with analyze_step10.py, and its job commands and part sizes are right (the helper
functions of the generated cells are executed here, not only compiled)."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step10 as a10  # noqa: E402
from vla_core.envs import parse_index_ranges  # noqa: E402

SHA = "c" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step10_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"] if c["cell_type"] == "code"]


def cell(cells, marker):
    found = [c for c in cells if marker in c]
    assert len(found) == 1, marker
    return found[0]


def namespace(cells):
    """The settings cell, then every function defined in the later cells, with Kaggle paths stubbed."""
    ns = {"POLICY_PATH": "/ckpt", "PY": "/venv/python", "REPO": "/repo", "RUN_ENV": {}}
    exec(cell(cells, "T_SESSION = time.time()"), ns)
    for c in cells:
        tree = ast.parse("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))))
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=defs, type_ignores=[]), "cell", "exec"), ns)
    ns["OUT_ROOT"] = "/out"
    ns["MAIN"] = "/repo/self_filtered_bc/self_filtered_bc.py"
    ns["DATA_A"] = "/out/task_0/collect_data"
    return ns


def test_cells_compile_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    ns = namespace(cells)
    assert ns["REPO_REF"] == SHA and ns["N_ACTION_STEPS"] == 10 and ns["OUT_NAME"] == "step10"
    assert ns["TASKS"] == list(a10.TASKS)
    assert ns["CHECKPOINT_REVISION"] == a10.POLICY_REVISION
    for t in a10.TASKS:
        assert tuple(parse_index_ranges(ns["EVAL_STATES"][t])) == a10.EVAL_STATES[t]
        assert [s for part in ns["EVAL_SEED_PARTS"][t] for s in parse_index_ranges(part)] == list(a10.EVAL_SEEDS[t])
        assert ns["POLICIES"][t] == a10.policies(t)
    assert ns["TRAIN_SEEDS"] == list(a10.TRAIN_SEEDS) and (ns["TRAIN_STEPS"], ns["BATCH_SIZE"]) == (2000, 16)
    assert sorted(ns["ADAPTERS"]) == sorted([f"A{s}" for s in range(3)] + [f"N{t}_{s}" for t in a10.TASKS for s in range(3)])
    assert ns["ADAPTERS"][:3] == ["A0", "A1", "A2"]


def test_part_sizes_add_up_to_the_analysis_episodes(tmp_path):
    ns = namespace(build(tmp_path))
    for t in a10.TASKS:
        total = sum(ns["part_size"](t, j) for j in range(len(ns["EVAL_SEED_PARTS"][t])))
        assert total == len(a10.EVAL_STATES[t]) * len(a10.EVAL_SEEDS[t])


def test_eval_commands(tmp_path):
    ns = namespace(build(tmp_path))
    base = ns["eval_cmd"](6, "base", 5)
    assert "--task_id 6 --n_action_steps 10 --policy_path /ckpt --stage eval --eval_seeds 36-39" in base
    assert "--eval_init_states 0,8,12,14,20" in base and "--eval_policies base" in base
    assert "--eval_output_dir /out/task_6/eval/base/part5" in base
    n = ns["eval_cmd"](7, "N2", 0)
    assert "--task_id 7" in n and "--eval_policies lora --output_dir /out/runs/N7_2" in n
    assert "--eval_output_dir /out/task_7/eval/N2/part0" in n
    a = ns["eval_cmd"](0, "A1", 3)
    assert "--task_id 0" in a and "--output_dir /out/runs/A1" in a and "--eval_seeds 28-31" in a
    assert [ns["adapter_of"](0, p) for p in ("N0", "A2")] == ["N0_0", "A2"]


def test_train_commands(tmp_path):
    ns = namespace(build(tmp_path))
    a = ns["train_cmd"]("A1")
    assert "--task_id 0 " in a and "--seed 1 " in a and "--data_dir /out/task_0/collect_data --val_init_states 45-49" in a
    assert "--selection success --train_steps 2000 --batch_size 16" in a and "--output_dir /out/runs/A1" in a
    n = ns["train_cmd"]("N6_2")
    assert "--task_id 6 " in n and "--seed 2 " in n and "--data_dir /out/task_6/replay_data --output_dir /out/runs/N6_2" in n
    assert "val_init_states" not in n


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("CHECK_DONE = True", "HARNESS_DONE = CHECK_DONE and", "REPLAYS_DONE = False",
                                     "COLLECTED = len(", "TIMING = Path(", "while ready and hours_left() > TRAIN_NEEDS_H",
                                     "ALL_DONE = (", "--out {OUT_ROOT}/step10_report.txt", "WORK REMAINS",
                                     "step10_results.zip", "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert 'run_stage("harness", jobs, 3 * len(TASKS))' in text
    assert 'MAIN = f"{REPO}/self_filtered_bc/self_filtered_bc.py"' in text
    assert "if HARNESS_DONE:\n    r = sh(f\"cd {REPO} && {PY} -S noise_reachability/analyze_step10.py --replay_jobs\"" in text
    assert "--record_dir {task_dir(t)}/replay_data" in text


def test_nothing_of_step_8_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("step8", "analyze_step8", "CONTROL_IDS", "SCREEN", "TEST1", "--hits", "--order", "check_scene"):
        assert gone not in text, gone


def test_every_process_gets_n_action_steps_from_one_variable(tmp_path):
    text = "\n".join(build(tmp_path))
    for entry in ("deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--task_id {task} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}",
                  "check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--init_states 0-9 --n_action_steps {N_ACTION_STEPS}"):
        assert entry in text


def test_resume_and_short_commit(tmp_path):
    text = "\n".join(build(tmp_path))
    assert "assert len(previous) <= 1" in text and "RESUMING from" in text
    r = subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step10_notebook.py", "abc1234",
                        str(tmp_path / "nb.ipynb")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "full commit" in r.stderr
