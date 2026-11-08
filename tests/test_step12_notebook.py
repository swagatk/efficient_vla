"""Step 12 notebook: settings agree with analyze_step12.py, and its job commands match what the analysis's provenance
check accepts (the helper functions of the generated cells are executed here, not only compiled)."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step12 as a  # noqa: E402
from vla_core.envs import parse_index_ranges  # noqa: E402

SHA = "c" * 40
STEP10_IN = "/kaggle/input/noise-reachability-step-10/step10"


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_step12_notebook.py", SHA, str(out)],
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
    ns["OUT_ROOT"] = "/kaggle/working/step12"
    ns["STEP10_IN"] = STEP10_IN
    ns["MAIN"] = "/repo/self_filtered_bc/self_filtered_bc.py"
    return ns


def flag(cmd, name):
    m = re.search(rf"--{name} (\S+)", cmd)
    return m.group(1) if m else None


def test_cells_compile_and_settings_match_the_analysis(tmp_path):
    cells = build(tmp_path)
    for c in cells:
        compile("\n".join(l for l in c.split("\n") if not l.lstrip().startswith(("!", "%"))), "cell", "exec")
    ns = namespace(cells)
    assert ns["REPO_REF"] == SHA and ns["N_ACTION_STEPS"] == 10 and ns["OUT_NAME"] == "step12"
    assert ns["TASKS"] == list(a.TASKS) and ns["CHECKPOINT_REVISION"] == a.POLICY_REVISION
    assert {t: tuple(v) for t, v in ns["SEARCHED"].items()} == a.SEARCHED and ns["HELD_OUT"] == a.HELD_OUT
    assert ns["CHECK"] == a.CHECK and ns["EXPLORER"] == a.EXPLORER and ns["POLICIES"] == a.POLICIES
    assert tuple(parse_index_ranges(ns["SEARCH_IDS"])) == a.SEARCH_IDS
    assert tuple(parse_index_ranges(ns["HARNESS_SEEDS"])) == a.HARNESS_SEEDS and ns["HARNESS_STATE"] == a.HARNESS_STATE
    for t in a.TASKS:
        assert tuple(parse_index_ranges(ns["EVAL_STATES"][t])) == a.eval_states(t)
        assert [s for part in ns["EVAL_SEED_PARTS"][t] for s in parse_index_ranges(part)] == list(a.EVAL_SEEDS[t])
    assert (ns["TRAIN_STEPS"], ns["BATCH_SIZE"]) == (2000, 16) and ns["TRAIN_SEEDS"] == list(a.TRAIN_SEEDS)
    assert sorted(ns["ADAPTERS"]) == sorted(f"R{t}_{s}" for t in a.TASKS for s in a.TRAIN_SEEDS)
    assert ns["STOP_AT_H"] == 11.0


def test_first_line_counts_the_episodes(tmp_path):
    text = cell(build(tmp_path), "T_SESSION = time.time()")
    assert "2236 episodes (12 harness, 400 search, 1824 evaluation)" in text


def test_part_sizes_add_up_to_the_analysis_episodes(tmp_path):
    ns = namespace(build(tmp_path))
    for t in a.TASKS:
        total = sum(ns["part_size"](t, j) for j in range(len(ns["EVAL_SEED_PARTS"][t])))
        assert total == len(a.eval_states(t)) * len(a.EVAL_SEEDS[t])


def test_eval_commands_pass_the_provenance_check(tmp_path):
    ns = namespace(build(tmp_path))
    for t, policy, j in ((6, "N1", 5), (7, "R2", 0), (0, "R0", 3), (0, "N0", 0)):
        cmd = ns["eval_cmd"](t, policy, j)
        assert f"--task_id {t} --n_action_steps 10 --policy_path /ckpt --stage eval --eval_policies lora" in cmd
        settings = {"stage": "eval", "eval_policies": "lora", "eval_init_states": flag(cmd, "eval_init_states"),
                    "eval_seeds": flag(cmd, "eval_seeds"), "lora_dir": flag(cmd, "lora_dir")}
        out = Path(flag(cmd, "eval_output_dir"))
        assert out == Path(f"/kaggle/working/step12/task_{t}/eval/{policy}/part{j}")
        a.check_settings("/kaggle/working/step12", t, out / "provenance.jsonl", {"settings": settings})
    assert flag(ns["eval_cmd"](6, "N1", 0), "lora_dir") == f"{STEP10_IN}/runs/N6_1/adapter"
    assert flag(ns["eval_cmd"](6, "R1", 0), "lora_dir") == "/kaggle/working/step12/runs/R6_1/adapter"


def test_train_commands_pass_the_provenance_check(tmp_path):
    ns = namespace(build(tmp_path))
    cmd = ns["train_cmd"]("R6_2")
    assert "--task_id 6 " in cmd and "--seed 2 " in cmd and "--selection success --train_steps 2000 --batch_size 16" in cmd
    assert "val_init_states" not in cmd
    settings = dict(a.TRAIN_SETTINGS, stage="train", seed=2, data_dir=flag(cmd, "data_dir"), val_init_states="")
    a.check_settings("/kaggle/working/step12", 6, Path("/kaggle/working/step12/runs/R6_2/provenance.jsonl"),
                     {"settings": settings})
    assert flag(cmd, "data_dir") == "/tmp/step12/task_6/round2_data"


def test_search_and_harness_commands_pass_the_provenance_check(tmp_path):
    text = "\n".join(build(tmp_path))
    assert ('f"{SEARCH} --task_id {t} --family repeat --init_state {s} --noise_ids {SEARCH_IDS} "\n'
            '                    f"--lora_dir {n_adapter(t, EXPLORER)} --output_dir {task_dir(t)}/search "\n'
            '                    f"--record_dir {task_dir(t)}/search_data --record_successes_only 1", k % 2)') in text
    assert 'name = f"repeat_n{k}_init{s:02d}_seed{r[\'seed\']}.npz"' in text   # deep_noise_search.record_name's form
    assert a.record_name("repeat", 7012, None, 6, 8012) == "repeat_n7012_init06_seed8012.npz"
    ns = namespace(build(tmp_path))
    lora = ns["n_adapter"](7, ns["EXPLORER"])
    settings = {"family": "repeat", "jitter_seeds": None, "noise_ids": ns["SEARCH_IDS"], "init_state": 38,
                "lora_dir": lora, "record_dir": "/kaggle/working/step12/task_7/search_data",
                "record_successes_only": 1, "frame_stride": 2}
    a.check_settings("/kaggle/working/step12", 7, Path("/kaggle/working/step12/task_7/search/provenance.jsonl"),
                     {"settings": settings, "git_dirty_sfbc": False})
    harness = {"stage": "eval", "eval_policies": "lora", "eval_init_states": str(ns["HARNESS_STATE"][7]),
               "eval_seeds": ns["HARNESS_SEEDS"], "lora_dir": lora}
    a.check_settings("/kaggle/working/step12", 7, Path("/kaggle/working/step12/task_7/harness/provenance.jsonl"),
                     {"settings": harness})
    assert ('--lora_dir {n_adapter(t, EXPLORER)} "\n                                        f"--eval_seeds {HARNESS_SEEDS} '
            '--eval_init_states {HARNESS_STATE[t]} "') in text


def test_stages_come_in_order_and_guard_what_follows(tmp_path):
    text = "\n".join(build(tmp_path))
    order = [text.index(s) for s in ("STEP10_IN = found[0]", "CHECK_DONE = True", "HARNESS_DONE = False",
                                     'HARNESS_DONE = run_gate("harness")', "SEARCH_DONE = False", "os.symlink(f, target)",
                                     "TIMING = Path(", "while ready and hours_left() > TRAIN_NEEDS_H", "ALL_DONE = (",
                                     "--out {OUT_ROOT}/step12_report.txt", "WORK REMAINS", "step12_results.zip",
                                     "raise RuntimeError(FAILURE)")]
    assert order == sorted(order)
    assert "if HARNESS_DONE and not FAILURE:\n    jobs, k = {}, 0\n    for t in TASKS:\n        for s in SEARCHED[t]:" in text
    assert "ready = [n for n in ADAPTERS if not adapter_done(n)] if SEARCH_DONE and not FAILURE else []" in text


def test_nothing_of_step_8_or_10_leaks_in(tmp_path):
    text = "\n".join(build(tmp_path))
    for gone in ("analyze_step8", "analyze_step10", "CONTROL_IDS", "SCREEN", "TEST1", "--hits", "check_scene",
                 "collect_data", "--replay_jobs", "step10_results.zip", "--eval_policies base"):
        assert gone not in text, gone


def test_every_process_gets_n_action_steps_from_one_variable(tmp_path):
    text = "\n".join(build(tmp_path))
    for entry in ("deep_noise_search.py --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}",
                  "--task_id {task} --n_action_steps {N_ACTION_STEPS} --policy_path {POLICY_PATH}",
                  "check_step5a.py --task_id {t} --policy_path {POLICY_PATH} --n_action_steps {N_ACTION_STEPS}"):
        assert entry in text
