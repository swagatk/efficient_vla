"""check_step12_session.py: interim counts, the round-1 reproduction check and the report, on synthetic folders."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location(
    "check_step12_session", ROOT / "week6/noise_reachability_step12_kaggle/check_step12_session.py")
cs = importlib.util.module_from_spec(spec)
sys.modules["check_step12_session"] = cs
spec.loader.exec_module(cs)

from test_analyze_step12 import _rewrite_train, make_run, write_csv  # noqa: E402

a = cs.a


def history(path: Path, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"step": 50 * (i + 1), "loss": x, "sec_per_step": 5.0} for i, x in enumerate(losses)]))


def session(tmp_path, keep=("R0_0", "R6_0", "R7_0")):
    """A synthetic run with only some round-2 runs finished; R7_0 trained on round 1's set only."""
    run, s10 = tmp_path / "run", tmp_path / "s10"
    make_run(run, s10)
    for d in sorted((run / "runs").iterdir()):
        if d.name not in keep:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    for t in a.TASKS:
        for s in a.TRAIN_SEEDS:
            history(s10 / f"runs/N{t}_{s}/train_history.json", [0.5, 0.25])
    for name in keep:
        history(run / f"runs/{name}/train_history.json", [0.5, 0.25] if name == "R7_0" else [0.5, 0.2])
    write_csv(run / "task_7/search/per_episode_repeat_state38.csv",     # task 7's search finds nothing, as in session 1
              [{"task_id": 7, "seed": 1000 + k, "init_state": 38, "success": 0, "steps": 520} for k in a.SEARCH_IDS])
    _rewrite_train(run / "runs/R7_0/provenance.jsonl", a.train_episodes(s10 / "runs/N7_0/provenance.jsonl"))
    (run / "repo_commit.txt").write_text("c" * 40 + "\n")
    return run, s10


def test_same_as_round_1_reports_only_runs_without_new_data(tmp_path):
    run, s10 = session(tmp_path)
    assert cs.same_as_round_1(run, s10) == {"R7_0": (2, 2)}
    history(run / "runs/R7_0/train_history.json", [0.5, 0.3])          # one logged loss differs
    assert cs.same_as_round_1(run, s10) == {"R7_0": (2, 1)}
    history(run / "runs/R7_0/train_history.json", [0.5])                # a different number of logged steps
    with pytest.raises(ValueError, match="R7_0"):
        cs.same_as_round_1(run, s10)


def test_eval_counts_count_partial_policies_and_refuse_repeats_and_strays(tmp_path):
    run, _ = session(tmp_path)
    f = run / "task_6/eval/N0/part1/per_episode_lora.csv"
    f.unlink()
    got = cs.eval_counts(run)
    assert got[(6, "N0")] == (128 - 16, 128) and got[(7, "R2")] == (96, 96) and len(got) == 18
    write_csv(f, [{"task_id": 6, "seed": 48, "init_state": 8, "success": 0, "steps": 9}])   # seed 48 is in part 0
    with pytest.raises(ValueError, match="task 6 N0"):
        cs.eval_counts(run)
    write_csv(f, [{"task_id": 6, "seed": 52, "init_state": 9, "success": 0, "steps": 9}])   # an unplanned state
    with pytest.raises(ValueError, match="task 6 N0"):
        cs.eval_counts(run)


def test_report_on_a_partial_session(tmp_path):
    run, s10 = session(tmp_path)
    text = cs.report(run, step10=s10)
    assert "harness gate: passed, 12/12 episodes" in text
    assert "search 6/8: 5/50 successes, noise ids [7000, 7010, 7020, 7030, 7040]" in text
    assert "search 7/38: 0/50 successes, noise ids []" in text and "search total: 35/400" in text
    assert "training set R6_0: round 1's 2 episodes plus 15 search successes" in text
    assert "training set R7_0: round 1's 2 episodes plus 0 search successes" in text
    assert "R7_0 trained on round 1's set only: loss equal to N7_0's at 2/2 logged steps" in text
    assert "eval episodes run: 1824/1824 (counts only; no success read)" in text
    assert "R0_1" not in text


def test_report_refuses_a_finished_run_with_the_wrong_training_set(tmp_path):
    run, s10 = session(tmp_path)
    f = run / "runs/R6_0/provenance.jsonl"
    _rewrite_train(f, a.train_episodes(f)[:-1])
    with pytest.raises(ValueError, match="R6_0"):
        cs.report(run, step10=s10)
