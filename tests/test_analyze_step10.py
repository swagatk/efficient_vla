"""Step 10 analysis: the replay list against the committed results, McNemar, both tests, the reading and refusals."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_step8 as a8  # noqa: E402
import analyze_step10 as a10  # noqa: E402
from test_analyze_step8 import write  # noqa: E402


# ---- the replay list --------------------------------------------------------------------------------------------

def test_replay_plan_counts_and_content():
    plan = a10.replay_plan()
    assert {t: len(v) for t, v in plan.items()} == {0: 56, 6: 33, 7: 33}
    assert sum(1 for e in plan[7] if e[3] is None) == 25 and sum(1 for e in plan[7] if e[3] is not None) == 8
    assert ("repeat", 33, 1459, None) in plan[7] and ("repeat", 33, 1459, 5000) in plan[7]
    assert {e[1] for e in plan[0]} == {2, 5, 6, 7, 8, 11} and {e[1] for e in plan[6]} == {8, 12}
    assert {e[0] for e in plan[0]} == {"full", "repeat"} and {e[0] for e in plan[6] + plan[7]} == {"repeat"}
    for t, eps in plan.items():                              # only resistant states, only the trained ones
        assert {e[1] for e in eps} == set(a10.TRAINED[t])


def test_replay_plan_reads_every_success_of_its_sources():
    w = ROOT / "week6/noise_reachability_step8_kaggle/run/task_7/test"
    hits = {int(r["seed"]) - 1000 for r in csv.DictReader(open(w / "per_episode_repeat_state33.csv")) if int(r["success"])}
    assert {e[2] for e in a10.replay_plan()[7] if e[3] is None} == hits


def expand(jobs):
    out = set()
    for t, f, s, ids, js in jobs:
        for k in map(int, ids.split(",")):
            for j in ([None] if js is None else map(int, js.split(","))):
                out.add((t, f, s, k, j))
    return out


def test_replay_jobs_cover_the_plan_exactly():
    plan = a10.replay_plan()
    want = {(t, *e) for t, eps in plan.items() for e in eps}
    jobs = a10.replay_jobs(plan)
    assert expand(jobs) == want
    assert all(len(js.split(",")) >= 1 and "," not in ids for _, _, _, ids, js in jobs if js)
    assert len(jobs) == 36


# ---- statistics -------------------------------------------------------------------------------------------------

def test_mcnemar_greater_known_answers():
    assert a10.mcnemar_greater(0, 0) == 1.0
    assert a10.mcnemar_greater(3, 0) == pytest.approx(1 / 8)
    assert a10.mcnemar_greater(5, 0) == pytest.approx(1 / 32)
    assert a10.mcnemar_greater(2, 1) == pytest.approx(4 / 8)
    assert a10.mcnemar_greater(0, 3) == 1.0
    assert a10.mcnemar_greater(6, 1) == pytest.approx(8 / 128)


def test_paired_counts_discordant_episodes():
    first = {1: 1, 2: 1, 3: 0, 4: 0, 5: 1}
    second = {1: 1, 2: 0, 3: 1, 4: 0, 5: 0}
    r = a10.paired(first, second, [1, 2, 3, 4, 5])
    assert (r["first"], r["second"], r["first_only"], r["second_only"], r["n"]) == (3, 2, 2, 1, 5)
    assert r["p"] == pytest.approx(0.5)


def test_reading():
    assert a10.reading(False, [0], True, True).startswith("not read: a harness")
    assert a10.reading(True, [], None, None).startswith("not read: no task")
    assert a10.reading(True, [0, 7], False, None).startswith("not shown")
    assert a10.reading(True, [0, 7], True, True).startswith("constant-noise successes make")
    assert a10.reading(True, [0, 7], True, False).startswith("learnable, but")
    assert a10.reading(True, [7], True, None).startswith("learnable, but")


# ---- end to end on a synthetic run ------------------------------------------------------------------------------

SMALL_PLAN = {0: [("repeat", 2, k, None) for k in range(20)] + [("full", 6, 29, 5000), ("full", 6, 29, 5001)],
              6: [("repeat", 8, k, None) for k in range(25)],
              7: [("repeat", 33, k, None) for k in range(30)] + [("repeat", 33, 1, 5002)]}


POLICY = f"/k/snapshots/{a8.POLICY_REVISION}"


def record(folder, task, flags=("git_dirty_noise_reachability",), **settings):
    """One provenance line as the entry point writing ``folder`` would append it."""
    r = {"git_commit": "c" * 40, "git_dirty": False, **{k: False for k in flags},
         "settings": {"task_id": task, "n_action_steps": 10, "max_steps": 520, "policy_path": POLICY, **settings}}
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def train_record(root, name, **change):
    t = a10.adapter_task(name)
    seed = int(name[1:]) if name.startswith("A") else int(name.split("_")[1])
    data = "/k/step10/task_0/collect_data" if name.startswith("A") else f"/k/step10/task_{t}/replay_data"
    settings = {"stage": "train", "seed": seed, "data_dir": data, "val_init_states": "45-49" if name.startswith("A") else "",
                **a10.TRAIN_SETTINGS}
    settings.update(change)
    record(root / "runs" / name, t, flags=("git_dirty_sfbc",), **settings)


def make_run(root, plan=SMALL_PLAN, harness=None, fail_replays=None, wins=None, drop=None, extra=None):
    """wins: {(task, policy): set of (seed, state) that succeed}; everything else fails."""
    harness, fail_replays, wins = harness or {}, fail_replays or {}, wins or {}
    for t in a10.TASKS:
        folder = root / f"task_{t}"
        for seed in range(3):
            n = harness.get(t, a8.HARNESS_REFERENCE[t])
            rows = [{"task_id": t, "seed": seed, "init_state": i, "success": int(seed * 10 + i < n), "steps": 520}
                    for i in range(10)]
            write(folder / "harness" / f"seed{seed}" / "per_episode_base.csv", rows)
            record(folder / "harness" / f"seed{seed}", t, policies="base")
        files = {}
        for n_fail, (f, s, k, j) in enumerate(plan[t]):
            name = f"per_episode_{f}_state{s}.csv" if j is None else f"confirm_{f}_n{k}_state{s}.csv"
            seed = 1000 + k if j is None else j
            files.setdefault(name, []).append({"task_id": t, "seed": seed, "init_state": s,
                                               "success": int(n_fail >= fail_replays.get(t, 0)), "steps": 300})
        if extra == t:
            files[next(iter(files))].append({"task_id": t, "seed": 99999, "init_state": 2, "success": 1, "steps": 1})
        for name, rows in files.items():
            write(folder / "replay" / name, rows)
        record(folder / "replay", t, flags=("git_dirty_noise_reachability", "git_dirty_sfbc"), family="repeat",
               frame_stride=2, record_dir=f"/k/step10/task_{t}/replay_data")
        for p in a10.policies(t):
            name = "base" if p == "base" else "lora"
            good = wins.get((t, p), set())
            for j, part in enumerate(a10.EVAL_SEED_PARTS[t]):
                lo, hi = map(int, part.split("-"))
                keys = [(s, i) for s in range(lo, hi + 1) for i in a10.EVAL_STATES[t]]
                if drop == (t, p) and j == len(a10.EVAL_SEED_PARTS[t]) - 1:
                    keys = keys[:-1]
                rows = [{"task_id": t, "seed": s, "init_state": i, "success": int((s, i) in good), "steps": 520}
                        for s, i in keys]
                out = folder / "eval" / p / f"part{j}"
                write(out / f"per_episode_{name}.csv", rows)
                record(out, t, flags=("git_dirty_sfbc",), stage="eval", eval_init_states=a10.EVAL_STATE_SPECS[t],
                       eval_seeds=part, eval_policies=name, policy=name,
                       lora_dir=None if p == "base" else f"/k/step10/runs/{a10.adapter_of(t, p)}/adapter")
            if p != "base":
                train_record(root, a10.adapter_of(t, p))
    for k, part in enumerate(a10.COLLECT_PARTS):
        record(root / "task_0" / "collect" / f"part{k}", 0, flags=("git_dirty_sfbc",), stage="collect", hard_reset=0,
               collect_seeds="0-2", frame_stride=2, init_states=part)
    return root


def trained_wins(task, n_per_state):
    return {(s, i) for i in a10.TRAINED[task] for s in a10.EVAL_SEEDS[task][:n_per_state]}


def test_learned_and_beyond_own_successes(tmp_path):
    wins = {}
    for i in range(3):
        for t in a10.TASKS:
            wins[(t, f"N{i}")] = trained_wins(t, 2)                 # 2 wins per trained state, base 0
    root = make_run(tmp_path, wins=wins)
    r = a10.analyse(root, SMALL_PLAN)
    t1 = r["tests"]["test 1"]
    assert t1["per_seed"][0]["n"] == 6 * 16 + 2 * 32 + 32
    assert t1["per_seed"][0]["first_only"] == 2 * 9 and t1["per_seed"][0]["p"] == pytest.approx(0.5 ** 18)
    assert t1["passes"] and r["tests"]["test 2"]["n"] == 240 and r["tests"]["test 2"]["first_only"] == 30
    assert r["decision"].startswith("constant-noise successes make")
    assert r["replays"] == {0: {"planned": 22, "succeeded": 22}, 6: {"planned": 25, "succeeded": 25},
                            7: {"planned": 31, "succeeded": 31}}
    text = a10.format_report(r)
    assert "test 1 passes" in text and "test 2, N against A" in text and "decision: constant-noise" in text


def test_two_of_three_seeds_is_enough_and_one_is_not(tmp_path):
    wins = {(t, "N0"): trained_wins(t, 2) for t in a10.TASKS} | {(t, "N1"): trained_wins(t, 2) for t in a10.TASKS}
    r = a10.analyse(make_run(tmp_path / "a", wins=wins), SMALL_PLAN)
    assert [v["passes"] for v in r["tests"]["test 1"]["per_seed"].values()] == [True, True, False]
    assert r["tests"]["test 1"]["passes"]
    wins = {(t, "N0"): trained_wins(t, 2) for t in a10.TASKS}
    r = a10.analyse(make_run(tmp_path / "b", wins=wins), SMALL_PLAN)
    assert not r["tests"]["test 1"]["passes"] and "test 2" not in r["tests"]
    assert r["decision"].startswith("not shown")


def test_test1_threshold_is_strict(tmp_path):
    # 5 N-only wins against 0: p = 1/32 < 0.05; 4 against 0: p = 1/16 > 0.05
    win5 = {(16, 2), (17, 2), (18, 2), (19, 2), (20, 2)}
    wins = {(0, f"N{i}"): win5 for i in range(3)}
    r = a10.analyse(make_run(tmp_path / "a", wins=wins), SMALL_PLAN)
    assert r["tests"]["test 1"]["per_seed"][0]["p"] == pytest.approx(1 / 32) and r["tests"]["test 1"]["passes"]
    wins = {(0, f"N{i}"): set(list(sorted(win5))[:4]) for i in range(3)}
    r = a10.analyse(make_run(tmp_path / "b", wins=wins), SMALL_PLAN)
    assert not r["tests"]["test 1"]["passes"]


def test_a_matching_n_does_not_pass_test2(tmp_path):
    wins = {}
    for i in range(3):
        for t in a10.TASKS:
            wins[(t, f"N{i}")] = trained_wins(t, 2)
        wins[(0, f"A{i}")] = trained_wins(0, 2)
    r = a10.analyse(make_run(tmp_path, wins=wins), SMALL_PLAN)
    assert r["tests"]["test 1"]["passes"] and not r["tests"]["test 2"]["passes"]
    assert r["tests"]["test 2"]["p"] == 1.0
    assert r["decision"].startswith("learnable, but")


def test_test2_ignores_state_11(tmp_path):
    wins = {(t, f"N{i}"): {(s, 11) for s in a10.EVAL_SEEDS[0]} for i in range(3) for t in (0,)}
    wins |= {(t, f"N{i}"): trained_wins(t, 3) for i in range(3) for t in (6, 7)}
    r = a10.analyse(make_run(tmp_path, wins=wins), SMALL_PLAN)
    assert r["tests"]["test 1"]["passes"] and r["tests"]["test 2"]["first_only"] == 0


def test_task_with_too_few_replays_is_left_out(tmp_path):
    # task 0 has 22 replays: 3 failing leaves 19, so it leaves the tests; 2 failing leaves 20, so it stays
    r = a10.analyse(make_run(tmp_path / "a", fail_replays={0: 3}), SMALL_PLAN)
    assert r["read_tasks"] == [6, 7] and r["replays"][0]["succeeded"] == 19
    assert "test 2" not in r["tests"]
    assert r["tests"]["test 1"]["per_seed"][0]["n"] == 2 * 32 + 32
    r = a10.analyse(make_run(tmp_path / "b", fail_replays={0: 2}), SMALL_PLAN)
    assert r["read_tasks"] == [0, 6, 7]


def test_no_task_read(tmp_path):
    r = a10.analyse(make_run(tmp_path, fail_replays={0: 99, 6: 99, 7: 99}), SMALL_PLAN)
    assert r["decision"].startswith("not read: no task")


def test_harness_failure(tmp_path):
    root = make_run(tmp_path, harness={6: 13})
    assert a10.analyse(root, SMALL_PLAN)["decision"].startswith("not read: a harness")
    assert a10.gate(root).startswith("harness failed: task 6")
    assert a10.main(["--root", str(root), "--gate", "harness"]) == 1


def test_refuses_missing_eval_episode(tmp_path):
    with pytest.raises(ValueError, match="finish the run"):
        a10.analyse(make_run(tmp_path, drop=(7, "N2")), SMALL_PLAN)


def test_refuses_missing_or_extra_replay(tmp_path):
    root = make_run(tmp_path / "a")
    (root / "task_7" / "replay" / "confirm_repeat_n1_state33.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        a10.analyse(root, SMALL_PLAN)
    with pytest.raises(ValueError, match="not planned"):
        a10.analyse(make_run(tmp_path / "b", extra=0), SMALL_PLAN)


def test_refuses_duplicate_eval_episode(tmp_path):
    root = make_run(tmp_path)
    p = root / "task_6" / "eval" / "base" / "part1" / "per_episode_base.csv"
    rows = list(csv.DictReader(open(p)))
    write(p, rows + rows[:1])
    with pytest.raises(ValueError, match="twice"):
        a10.analyse(root, SMALL_PLAN)


def test_reported_comparisons(tmp_path):
    wins = {(0, "N0"): {(s, 14) for s in range(16, 20)}, (0, "A1"): {(16, 2)}}
    r = a10.analyse(make_run(tmp_path, wins=wins), SMALL_PLAN)
    assert r["reported"]["task 0 held out, N0 against base"]["first"] == 4
    assert r["reported"]["task 0 held out, N0 against base"]["n"] == 2 * 16
    assert r["reported"]["task 7 easy, N2 against base"]["n"] == 32
    assert r["reported"]["task 0 systematic, A1 against base"]["first_only"] == 1
    assert not any(k.startswith("task 6") and " A" in k for k in r["reported"])


def test_policies_per_task():
    assert a10.policies(0) == ["base", "N0", "N1", "N2", "A0", "A1", "A2"]
    assert a10.policies(7) == ["base", "N0", "N1", "N2"]


def test_design_sizes():
    n = {t: len(a10.EVAL_STATES[t]) * len(a10.EVAL_SEEDS[t]) for t in a10.TASKS}
    assert n == {0: 208, 6: 160, 7: 128}
    assert sum(n[t] * len(a10.policies(t)) for t in a10.TASKS) == 2608
    for t in a10.TASKS:
        roles = set(a10.TRAINED[t]) | set(a10.HELD_OUT[t]) | set(a10.EASY[t])
        assert roles <= set(a10.EVAL_STATES[t])
        assert not set(a10.TRAINED[t]) & set(a10.HELD_OUT[t]) and not set(a10.TRAINED[t]) & set(a10.EASY[t])
    assert set(range(10)) == set(a10.EASY[0]) | set(a10.SYSTEMATIC_0)


def test_held_out_are_those_of_step_9():
    import analyze_step9 as a9
    assert {t: a10.HELD_OUT[t] for t in (0, 6, 7)} == {t: a9.HELD_OUT[t] for t in (0, 6, 7)}


def test_cli(tmp_path, capsys):
    assert a10.main(["--replay_jobs"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 36 and lines[-1].startswith("7 repeat 33 1473 ")
    root = make_run(tmp_path / "run")
    out = tmp_path / "r.txt"
    import unittest.mock as m
    steps = {(t, *e): 300 for t, eps in SMALL_PLAN.items() for e in eps}
    with m.patch.object(a10, "replay_plan", return_value=SMALL_PLAN), m.patch.object(a10, "source_steps", return_value=steps):
        assert a10.main(["--root", str(root), "--out", str(out)]) == 0
    assert out.read_text().endswith("decision: not shown: one round of filtered BC on constant-noise successes did "
                                    "not make these states solvable by sampling\n")
    assert json.loads(Path(str(out) + ".json").read_text())["read_tasks"] == [0, 6, 7]


def test_paired_does_not_count_shared_successes():
    first = {1: 1, 2: 1, 3: 1, 4: 0}
    second = {1: 1, 2: 1, 3: 0, 4: 1}
    r = a10.paired(first, second, [1, 2, 3, 4])
    assert (r["first_only"], r["second_only"]) == (1, 1)


def test_state_roles_and_report_lines(tmp_path):
    wins = {(7, "N1"): {(16, 33), (17, 33), (18, 33)}}
    root = make_run(tmp_path, wins=wins)
    r = a10.analyse(root, SMALL_PLAN)
    roles = {(v["task"], v["state"]): v["role"] for v in r["states"]}
    assert roles[(0, 2)] == roles[(0, 11)] == roles[(7, 33)] == "trained"
    assert roles[(0, 14)] == roles[(6, 20)] == roles[(7, 47)] == "held out"
    assert roles[(0, 9)] == roles[(6, 0)] == roles[(7, 4)] == "easy"
    text = a10.format_report(r)
    zero, three = (f"({100 * a:.1f}-{100 * b:.1f}%)" for a, b in (a10.clopper_pearson(0, 32), a10.clopper_pearson(3, 32)))
    assert f"task 7 state 33 (trained), of 32: base 0 {zero}, N0 0 {zero}, N1 3 {three}, N2 0 {zero}" in text
    assert zero == "(0.0-10.9%)"
    assert "tasks read: [0, 6, 7]" in text
    assert a10.main(["--root", str(root), "--gate", "harness"]) == 0


def test_report_after_a_harness_failure(tmp_path):
    r = a10.analyse(make_run(tmp_path, harness={0: 0}), SMALL_PLAN)
    text = a10.format_report(r)
    assert "tasks read" not in text and "test 1" not in text
    assert text.endswith("decision: not read: a harness failed\n")



def test_provenance_of_a_complete_run_passes(tmp_path):
    root = make_run(tmp_path)
    n = 3 * 3 + 3 + sum(len(a10.policies(t)) * len(a10.EVAL_SEED_PARTS[t]) for t in a10.TASKS) + 12 + 4
    assert a10.check_provenance(root)["records"] == n


@pytest.mark.parametrize("change", [{"git_dirty_sfbc": True}, {"git_dirty": True}, {"git_dirty_sfbc": None},
                                    {"git_commit": "d" * 40}])
def test_provenance_refuses_dirty_or_other_commit(tmp_path, change):
    root = make_run(tmp_path)
    f = root / "runs" / "N7_0" / "provenance.jsonl"
    r = json.loads(f.read_text())
    r.update(change)
    f.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError):
        a10.check_provenance(root)


def test_provenance_refuses_a_record_with_no_dirty_flag(tmp_path):
    root = make_run(tmp_path)
    f = root / "task_7" / "eval" / "base" / "part0" / "provenance.jsonl"
    r = json.loads(f.read_text())
    del r["git_dirty_sfbc"]
    f.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        a10.check_provenance(root)


def test_provenance_refuses_a_record_dirty_under_either_of_two_flags(tmp_path):
    root = make_run(tmp_path)
    f = root / "task_6" / "replay" / "provenance.jsonl"
    r = json.loads(f.read_text())
    r["git_dirty_noise_reachability"] = True
    f.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        a10.check_provenance(root)


@pytest.mark.parametrize("name, change, match", [
    ("N6_1", {"train_steps": 1000}, "training settings"),
    ("N6_1", {"lora_r": 8}, "training settings"),
    ("N6_1", {"selection": "all"}, "training settings"),
    ("N6_1", {"seed": 2}, "seed"),
    ("N6_1", {"data_dir": "/k/step10/task_7/replay_data"}, "data"),
    ("N6_1", {"val_init_states": "45-49"}, "validation"),
    ("A0", {"val_init_states": ""}, "validation"),
    ("A0", {"data_dir": "/k/step10/task_0/replay_data"}, "data"),
])
def test_provenance_refuses_wrong_training(tmp_path, name, change, match):
    root = make_run(tmp_path)
    (root / "runs" / name / "provenance.jsonl").unlink()
    train_record(root, name, **change)
    with pytest.raises(ValueError, match=match):
        a10.check_provenance(root)


def edit(path, **change):
    r = json.loads(path.read_text().splitlines()[-1])
    r["settings"].update(change)
    path.write_text(json.dumps(r) + "\n")


@pytest.mark.parametrize("where, change, match", [
    ("task_6/eval/N1/part0", {"lora_dir": "/k/step10/runs/N6_2/adapter"}, "adapter"),
    ("task_6/eval/N1/part0", {"lora_dir": "/k/step10/runs/N7_1/adapter"}, "adapter"),
    ("task_0/eval/A2/part1", {"lora_dir": "/k/step10/runs/A1/adapter"}, "adapter"),
    ("task_0/eval/base/part1", {"eval_policies": "lora", "policy": "lora"}, "base"),
    ("task_7/eval/N0/part3", {"eval_seeds": "16-19"}, "episodes"),
    ("task_7/eval/N0/part3", {"eval_init_states": "4,33,38"}, "episodes"),
    ("task_0/collect/part2", {"hard_reset": 1}, "collection"),
    ("task_0/collect/part2", {"init_states": "10-19"}, "collection"),
    ("task_6/replay", {"record_dir": "/k/step10/task_7/replay_data"}, "record_dir"),
    ("task_6/replay", {"frame_stride": 1}, "replay settings"),
])
def test_provenance_refuses_wrong_settings(tmp_path, where, change, match):
    root = make_run(tmp_path)
    edit(root / where / "provenance.jsonl", **change)
    with pytest.raises(ValueError, match=match):
        a10.check_provenance(root)


def test_provenance_refuses_a_record_in_an_unknown_folder(tmp_path):
    root = make_run(tmp_path)
    record(root / "task_0", 0)
    with pytest.raises(ValueError, match="unexpected folder"):
        a10.check_provenance(root)


def test_adapter_task_and_adapter_of():
    assert [a10.adapter_task(n) for n in ("A0", "A2", "N0_1", "N6_0", "N7_2")] == [0, 0, 0, 6, 7]
    assert [a10.adapter_of(6, "N1"), a10.adapter_of(0, "A2"), a10.adapter_of(0, "N0")] == ["N6_1", "A2", "N0_0"]


def test_seed_parts_cover_the_seeds():
    for t in a10.TASKS:
        seeds = [s for p in a10.EVAL_SEED_PARTS[t] for s in range(int(p.split("-")[0]), int(p.split("-")[1]) + 1)]
        assert seeds == list(a10.EVAL_SEEDS[t])


def test_replays_with_the_original_step_count_are_counted(tmp_path):
    steps = {(t, *e): 300 for t, eps in SMALL_PLAN.items() for e in eps}
    steps[(7, "repeat", 33, 1, 5002)] = 301
    steps[(6, "repeat", 8, 0, None)] = 299
    r = a10.analyse(make_run(tmp_path, fail_replays={0: 1}), SMALL_PLAN, steps)
    assert {t: v["same_steps"] for t, v in r["replays"].items()} == {0: 21, 6: 24, 7: 30}
    assert "replays task 0: 21 of 22 succeeded again; 21 in the original's number of steps" in a10.format_report(r)


def test_source_steps_are_the_committed_originals():
    steps = a10.source_steps()
    assert steps[(7, "repeat", 33, 1459, None)] == 327 and steps[(7, "repeat", 33, 1459, 5000)] == 257
    assert set(steps) == {(t, *e) for t, eps in a10.replay_plan().items() for e in eps}


def test_arm_a_is_not_reported_on_states_it_collected_on(tmp_path):
    r = a10.analyse(make_run(tmp_path), SMALL_PLAN)
    rows = {(v["task"], v["state"]): v for v in r["states"]}
    assert "A0" in rows[(0, 2)] and "A0" not in rows[(0, 11)] and "A2" not in rows[(0, 27)]
    assert "task 0 state 14 (held out), of 16: base 0" in a10.format_report(r)
    assert "A0" not in [l for l in a10.format_report(r).splitlines() if l.startswith("task 0 state 14")][0]


def test_report_prints_both_interval_ends(tmp_path):
    wins = {(7, "N1"): {(s, 33) for s in range(16, 32)}}
    r = a10.analyse(make_run(tmp_path, wins=wins), SMALL_PLAN)
    lo, hi = a10.clopper_pearson(16, 32)
    assert f"N1 16 ({100 * lo:.1f}-{100 * hi:.1f}%)" in a10.format_report(r)
    assert f"{100 * lo:.1f}" == "31.9"
