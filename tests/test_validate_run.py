import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from validate_run import main, parse_range, validate, values_match  # noqa: E402


def make_run(root, tasks=(0, 4), seeds=(0, 1), states=range(3), settings=None, dirty=False, log="n_action_steps': 10"):
    root.mkdir(parents=True, exist_ok=True)
    rec = {"git_commit": "abc", "git_dirty": dirty, "settings": settings or {"n_action_steps": 10, "tasks": list(tasks)}}
    (root / "provenance.jsonl").write_text(json.dumps(rec) + "\n")
    with open(root / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success"])
        w.writeheader()
        w.writerows({"task_id": t, "seed": s, "init_state": i, "success": 1} for t in tasks for s in seeds for i in states)
    (root / "task_0_seed_0").mkdir(exist_ok=True)
    with gzip.open(root / "task_0_seed_0" / "eval.log.gz", "wt") as f:
        f.write(f"config {{'{log}}}\n")
    return root


def run(root, *extra):
    return validate(root, [0, 4], [0, 1], [0, 1, 2], {"n_action_steps": "10", "tasks": "0 4"}, ["n_action_steps': 10"]
                    ) if not extra else validate(root, *extra)


def test_ranges_and_values():
    assert parse_range("3-5") == [3, 4, 5] and parse_range("0 4,6") == [0, 4, 6] and parse_range([4, 0]) == [0, 4]
    assert parse_range("lerobot") is None and parse_range(True) is None
    assert values_match([0, 4, 6], "0 4 6") and values_match("10-19", "10-19") and not values_match(1, "10")
    assert values_match("lerobot", "lerobot") and not values_match("medoid_n8", "medoid_n16")


def test_a_correct_run_passes(tmp_path):
    assert run(make_run(tmp_path / "ok")) == []


def test_wrong_setting_dirty_tree_missing_log_pattern_are_caught(tmp_path):
    errors = run(make_run(tmp_path / "bad", settings={"n_action_steps": 1, "tasks": [0, 4]}, dirty=True,
                          log="n_action_steps': 1"))
    text = "\n".join(errors)
    assert "settings.n_action_steps should be 10" in text and "git_dirty" in text and "not found" in text


def test_missing_extra_and_duplicate_episodes_are_caught(tmp_path):
    root = make_run(tmp_path / "eps", seeds=(0,))
    assert any("missing" in e and "6 distinct of 12" in e for e in run(root))
    with open(root / "per_episode.csv", "a") as f:
        f.write("0,0,0,1\n9,0,0,1\n")
    e = "\n".join(validate(root, [0, 4], [0], [0, 1, 2], {}, []))
    assert "1 unexpected" in e and "1 duplicated" in e


def test_underscore_folders_are_not_part_of_the_run_and_traceback_fails(tmp_path):
    root = make_run(tmp_path / "t")
    make_run(root / "_timing", tasks=(3,), seeds=(0,), states=range(1), settings={"n_action_steps": 10, "tasks": [3]})
    assert run(root) == []
    (root / "task_4_seed_1").mkdir()
    (root / "task_4_seed_1" / "eval.log").write_text("n_action_steps': 10\nTraceback (most recent call last):\n")
    assert any("traceback" in e for e in run(root))


def test_cli_exit_code(tmp_path, capsys):
    root = make_run(tmp_path / "cli")
    assert main([str(root), "--tasks", "0 4", "--seeds", "0-1", "--init_states", "0-2", "--expect", "n_action_steps=10"]) == 0
    assert main([str(root), "--tasks", "0 4", "--seeds", "0-2", "--init_states", "0-2"]) == 1
    assert "FAIL" in capsys.readouterr().out
