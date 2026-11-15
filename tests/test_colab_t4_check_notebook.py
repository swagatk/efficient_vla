"""Colab T4 check notebook: its settings agree with analyze_colab_t4_check.py, and the job cell's commands, parsed by
self_filtered_bc's own argument parser, have exactly the settings the analysis requires (cells executed with stubs)."""
import ast
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "self_filtered_bc"))

import analyze_colab_t4_check as a  # noqa: E402

SHA = "d" * 40


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-I", "-S", "noise_reachability/build_colab_t4_check_notebook.py", SHA, str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    nb = json.loads(out.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"], nb


def cell(cells, marker):
    found = [c for c in cells if marker in c]
    assert len(found) == 1, marker
    return found[0]


def namespace(cells):
    ns = {"POLICY_PATH": "/ckpt", "PY": "/venv/python", "REPO": "/repo", "RUN_ENV": {}, "JOBS": 2}
    exec(cell(cells, "T_SESSION = time.time()"), ns)
    for c in cells:
        tree = ast.parse(c)
        keep = (ast.FunctionDef, ast.Assign) if "def sfbc(" in c else (ast.FunctionDef,)   # sfbc's cell also sets MAIN
        defs = [n for n in tree.body if isinstance(n, keep)]
        exec(compile(ast.Module(body=defs, type_ignores=[]), "cell", "exec"), ns)
    return ns


def test_every_code_cell_parses_and_the_commit_is_pinned(tmp_path):
    cells, nb = build(tmp_path)
    for c in cells:
        ast.parse(c)
    assert f'REPO_REF = "{SHA}"' in cell(cells, "T_SESSION = time.time()")
    assert nb["metadata"]["colab"]["gpuType"] == "T4"


def test_settings_agree_with_the_analysis(tmp_path):
    ns = namespace(build(tmp_path)[0])
    assert ns["TASKS"] == list(a.TASKS) and ns["HARNESS_STATE"] == dict(a.HARNESS_STATE)
    assert ns["HARNESS_SEEDS"] == f"{a.HARNESS_SEEDS[0]}-{a.HARNESS_SEEDS[-1]}" and ns["N_ACTION_STEPS"] == a.N_ACTION_STEPS
    assert ns["POLICIES"] == dict(a.POLICIES) and ns["CHECKPOINT_REVISION"].startswith(a.POLICY_REVISION)
    assert sorted(ns["ADAPTER_SHA256"]) == sorted(a.TASKS)


def test_job_commands_have_the_settings_the_analysis_requires(tmp_path):
    import self_filtered_bc as sfbc

    cells, _ = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    ns.update(read_csv=lambda p: [], run_jobs=lambda jobs, max_parallel: captured.update(jobs) or {},
              failed=lambda codes: {}, fail=lambda m: None)
    exec(cell(cells, "jobs[f\"task{t}_{policy}\"]"), ns)
    assert len(captured) == len(a.TASKS) * len(a.POLICIES)
    for name, (cmd, gpu) in captured.items():
        assert gpu == 0
        argv = shlex.split(cmd.split(" && ", 1)[1])
        assert argv[:3] == [ns["PY"], "-u", f'{ns["REPO"]}/self_filtered_bc/self_filtered_bc.py']
        args = sfbc.parse_args(argv[3:])
        t, policy = int(name.split("_")[0][4:]), name.split("_")[1]
        flag = a.POLICIES[policy][0]
        assert (args.stage, args.task_id, args.eval_policies) == ("eval", t, flag)
        assert (args.n_action_steps, args.max_steps) == (a.N_ACTION_STEPS, a.MAX_STEPS)
        assert args.eval_seeds == "16-19" and args.eval_init_states == str(a.HARNESS_STATE[t])
        assert args.policy_path == "/ckpt" and args.eval_output_dir.endswith(f"/run/task_{t}/{policy}")
        if flag == "lora":
            assert Path(args.lora_dir).parent.name == f"N{t}_0"


def test_resume_skips_finished_jobs(tmp_path):
    cells, _ = build(tmp_path)
    ns = namespace(cells)
    captured = {}
    done = [{}] * 4
    ns.update(read_csv=lambda p: done if p.endswith("task_0/base/per_episode_base.csv") else [],
              run_jobs=lambda jobs, max_parallel: captured.update(jobs) or {}, failed=lambda c: {}, fail=lambda m: None)
    exec(cell(cells, "jobs[f\"task{t}_{policy}\"]"), ns)
    assert "task0_base" not in captured and len(captured) == len(a.TASKS) * len(a.POLICIES) - 1
