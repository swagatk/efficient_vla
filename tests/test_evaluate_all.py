import csv
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "baseline_performance"))
sys.path.insert(0, str(REPO_ROOT))

from evaluate_all import build_command, generate_report, run_all  # noqa: E402
from vla_core.provenance import append_jsonl, collect, git_state  # noqa: E402

# Stand-in for lerobot-eval: writes eval_info.json in the real layout. Successes follow the init state
# index and the seed; FAKE_FAIL_TASK makes that task crash, FAKE_SLEEP delays every run. It runs with -S
# so the child does not import robosuite_mujoco_patch through sitecustomize, which took minutes and
# hundreds of MB per process and exhausted WSL memory when the tests were first run without it.
FAKE_EVAL = '''#!{python} -S
import json, os, sys, time
args = dict(a[2:].split("=", 1) for a in sys.argv[1:])
task = int(args["env.task_ids"].strip("[]"))
seed = int(args["seed"])
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
with open(os.path.join(args["output_dir"], "gpu.txt"), "w") as f:
    f.write(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
if os.environ.get("FAKE_FAIL_TASK") == str(task):
    sys.exit(139)
n = int(args["eval.n_episodes"])
successes = [(i + seed + task) % 3 == 0 for i in range(n)]
info = {{"per_task": [{{"task_group": "libero_10", "task_id": task, "metrics": {{"successes": successes}}}}],
        "overall": {{"pc_success": 100.0 * sum(successes) / n, "n_episodes": n}}}}
with open(os.path.join(args["output_dir"], "eval_info.json"), "w") as f:
    json.dump(info, f)
'''


@pytest.fixture
def fake_eval(tmp_path):
    path = tmp_path / "fake_lerobot_eval"
    path.write_text(FAKE_EVAL.format(python=sys.executable))
    path.chmod(0o755)
    return str(path)


def kwargs(fake, n=4):
    return {"lerobot_eval": fake, "n_episodes": n}


def test_default_command_matches_results_300926_protocol():
    cmd = build_command(3, 1, "out/task_3_seed_1")
    assert cmd == [
        "lerobot-eval", "--policy.path=HuggingFaceVLA/smolvla_libero", "--policy.device=cuda", "--env.type=libero",
        "--env.task=libero_10", "--env.render_mode=rgb_array", "--env.max_parallel_tasks=1",
        "--env.observation_height=256", "--env.observation_width=256", "--env.episode_length=520",
        "--eval.batch_size=1", "--eval.use_async_envs=false", "--eval.n_episodes=10",
        "--output_dir=out/task_3_seed_1", "--env.task_ids=[3]", "--seed=1",
    ]


def test_run_report_and_resume(tmp_path, fake_eval):
    out = tmp_path / "out"
    codes = run_all(out, [0, 1], [0, 1], kwargs(fake_eval), jobs=3, launch_delay_s=0, poll_s=0.05, log=lambda m: None)
    assert codes == {(t, s): 0 for t in (0, 1) for s in (0, 1)}
    assert generate_report(out, [0, 1], [0, 1], 4) == []

    with open(out / "consolidated_report.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["task_id", "seed", "success_rate"]
    assert {(r["task_id"], r["seed"]): float(r["success_rate"]) for r in rows}[("1", "1")] == 25.0
    with open(out / "per_episode.csv") as f:
        eps = list(csv.DictReader(f))
    assert len(eps) == 16
    assert [e["init_state"] for e in eps if e["task_id"] == "0" and e["seed"] == "0"] == ["0", "1", "2", "3"]
    assert (out / "consolidated_summary.txt").read_text().splitlines()[2].startswith("Task  0 | Mean Success:")

    # Finished runs are skipped on resume.
    assert run_all(out, [0, 1], [0, 1], kwargs(fake_eval), poll_s=0.05, log=lambda m: None) == {}


def test_crashed_run_is_reported_missing_not_dropped_silently(tmp_path, fake_eval, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_TASK", "1")
    out = tmp_path / "out"
    codes = run_all(out, [0, 1], [0], kwargs(fake_eval), launch_delay_s=0, poll_s=0.05, log=lambda m: None)
    assert codes[(1, 0)] == 139
    assert generate_report(out, [0, 1], [0], 4) == [(1, 0)]
    summary = (out / "consolidated_summary.txt").read_text()
    assert "INCOMPLETE: 1 of 2 runs missing" in summary and "task 1 seed 0" in summary


def test_wrong_episode_count_is_missing(tmp_path, fake_eval):
    out = tmp_path / "out"
    run_all(out, [0], [0], kwargs(fake_eval, n=3), poll_s=0.05, log=lambda m: None)
    assert generate_report(out, [0], [0], 10) == [(0, 0)]


def test_gpus_round_robin(tmp_path, fake_eval):
    out = tmp_path / "out"
    run_all(out, [0, 1, 2], [0], kwargs(fake_eval), jobs=3, gpus=["0", "1"], launch_delay_s=0, poll_s=0.05, log=lambda m: None)
    assert [(out / f"task_{t}_seed_0" / "gpu.txt").read_text() for t in (0, 1, 2)] == ["0", "1", "0"]


def test_launch_deadline_and_hard_stop(tmp_path, fake_eval, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "30")
    out = tmp_path / "out"
    t0 = time.monotonic()
    codes = run_all(out, [0, 1, 2], [0], kwargs(fake_eval), jobs=1, launch_deadline_s=0.2, hard_stop_s=1.0,
                    launch_delay_s=0, poll_s=0.05, log=lambda m: None)
    assert time.monotonic() - t0 < 10
    assert list(codes) == [(0, 0)] and codes[(0, 0)] != 0
    assert generate_report(out, [0, 1, 2], [0], 4) == [(0, 0), (1, 0), (2, 0)]


def test_report_reproduces_results_300926():
    # The run folder is gitignored, so this only runs on the machine that produced results_300926.
    root = REPO_ROOT / "baseline_performance" / "outputs" / "evaluate_all_20260929_140518"
    if not root.exists():
        pytest.skip("original output folder is not on this machine")
    reference = (REPO_ROOT / "baseline_performance" / "results_300926" / "consolidated_summary.txt").read_text()
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "copy"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns("videos", "*.log"))
        assert generate_report(copy, list(range(10)), [0, 1, 2], 10) == []
        assert (copy / "consolidated_summary.txt").read_text() == reference
        with open(copy / "consolidated_report.csv") as f, open(REPO_ROOT / "baseline_performance" / "results_300926" / "consolidated_report.csv") as g:
            assert [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)] == [{k: float(v) for k, v in r.items()} for r in csv.DictReader(g)]


def test_provenance_record(tmp_path):
    state = git_state(str(REPO_ROOT))
    assert len(state["git_commit"]) == 40 and isinstance(state["git_dirty"], bool)
    assert git_state(str(tmp_path)) == {"git_commit": None, "git_dirty": None}
    record = collect(str(REPO_ROOT), {"jobs": 2})
    assert record["settings"] == {"jobs": 2} and "lerobot" in record["packages"]
    path = tmp_path / "provenance.jsonl"
    append_jsonl(str(path), record)
    append_jsonl(str(path), record)
    assert [json.loads(line)["git_commit"] for line in path.read_text().splitlines()] == [state["git_commit"]] * 2
