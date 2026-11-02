import csv
import json
import os
import sys
from math import comb
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_ROOT", ROOT))
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step8 as a8  # noqa: E402

COMMIT = "c" * 40


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def episodes(task, state, ids, hits=()):
    return [{"task_id": task, "seed": 1000 + k, "init_state": state, "success": int(k in hits), "steps": 520} for k in ids]


def provenance(folder, task, **change):
    r = {"git_commit": COMMIT, "git_dirty": False, "git_dirty_noise_reachability": False,
         "settings": {"task_id": task, "n_action_steps": 10, "max_steps": 520, "policy_path": f"/s/{a8.POLICY_REVISION}"}}
    r["settings"].update(change.pop("settings", {}))
    r.update(change)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def scene(root, task, state, same=True):
    folder = root / f"task_{task}" / "scene"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"scene_state{state}.json").write_text(json.dumps({"task_id": task, "init_state": state, "same": same}))


def gates(root, control=None, harness=None):
    control, harness = control or {}, harness or {}
    for t in a8.TASKS:
        folder = root / f"task_{t}"
        provenance(folder, t)
        for seed in range(3):
            n = harness.get(t, a8.HARNESS_REFERENCE[t])
            rows = [{"task_id": t, "seed": seed, "init_state": i, "success": int(seed * 10 + i < n), "steps": 520} for i in range(10)]
            write(folder / "harness" / f"seed{seed}" / "per_episode_base.csv", rows)
            provenance(folder / "harness" / f"seed{seed}", t)
        for f in a8.FAMILIES:
            n = control.get(t, {}).get(f, 10)
            write(folder / "control" / f"per_episode_{f}_state{a8.CONTROL[t]}.csv",
                  episodes(t, a8.CONTROL[t], a8.CONTROL_IDS, a8.CONTROL_IDS[:n]))


def screen(root, zero=None, direct_hits=()):
    """zero: {task: states at 0 of 8}; each gets the second part. Other states hit once in the first part."""
    zero = {0: [12, 15], 4: [11], 6: [20]} if zero is None else zero
    for t in a8.SCREEN_TASKS:
        for s in a8.SCREEN_STATES:
            if s in zero.get(t, ()):
                ids = a8.ALL_SCREEN_IDS
                hits = ()
            else:
                ids, hits = a8.FIRST_IDS, a8.FIRST_IDS[:1]
            write(root / f"task_{t}" / "screen" / f"per_episode_sampled_state{s}.csv", episodes(t, s, ids, hits))
    for t, states in a8.DIRECT.items():
        for s in states:
            hits = a8.ALL_SCREEN_IDS[:1] if s in direct_hits else ()
            write(root / f"task_{t}" / "screen" / f"per_episode_sampled_state{s}.csv", episodes(t, s, a8.ALL_SCREEN_IDS, hits))


def write_test(root, task, state, ids, sampled=0, repeat=0, jitter=2):
    folder = root / f"task_{task}" / "test"
    for f, n in (("sampled", sampled), ("repeat", repeat)):
        write(folder / f"per_episode_{f}_state{state}.csv", episodes(task, state, ids, ids[:n]))
    for k in ids[:repeat][:3]:
        write(folder / f"confirm_repeat_n{k}_state{state}.csv",
              [{"task_id": task, "seed": s, "init_state": state, "success": int(i < jitter), "steps": 520}
               for i, s in enumerate((5000, 5001, 5002, 5003))])


def full_run(root, t1=(0, 0), t2=(0, 0)):
    """test 1 (sampled, repeat); test 2 the same counts on its first state, 0 on the others."""
    gates(root)
    screen(root)
    scene(root, *a8.TEST1)
    for t, s in a8.eligible_order(root)[:4]:
        scene(root, t, s)
    plan = a8.test_plan(root)
    write_test(root, *a8.TEST1, a8.TEST1_IDS, *t1)
    for i, (t, s, ids) in enumerate(plan["test2"]):
        write_test(root, t, s, ids, *(t2 if i == 0 else (0, 0)))
    return plan


def test_round_robin_takes_each_task_in_turn_lowest_state_first():
    assert a8.round_robin({6: [20], 0: [15, 12, 18], 4: [11, 13]}) == [(0, 12), (4, 11), (6, 20), (0, 15), (4, 13), (0, 18)]


def test_round_robin_stops_at_the_limit_even_inside_a_round():
    assert a8.round_robin({0: [1, 2], 4: [3, 4], 6: [5]}, 2) == [(0, 1), (4, 3)]
    assert a8.round_robin({0: [1, 2], 4: [3]}, 3) == [(0, 1), (4, 3), (0, 2)]
    assert a8.round_robin({}, 3) == []


def test_test_ids_are_disjoint_and_miss_every_other_range():
    blocks = [set(a8.test2_ids(i)) for i in range(a8.MAX_SELECTED)]
    assert [len(b) for b in blocks] == [150] * 4
    assert a8.test2_ids(0)[0] == 1000 and a8.test2_ids(3)[-1] == 1599
    assert len(set().union(*blocks)) == 600
    others = set(a8.CONTROL_IDS) | set(a8.TEST1_IDS) | set(a8.ALL_SCREEN_IDS) | set(range(70))
    assert not others & set().union(*blocks)
    assert not set(a8.TEST1_IDS) & set(a8.ALL_SCREEN_IDS) and len(a8.ALL_SCREEN_IDS) == 32 and len(a8.TEST1_IDS) == 300


def test_constants_match_the_readme():
    assert a8.HARNESS_REFERENCE == {0: 7, 4: 14, 6: 20, 7: 18} and a8.CONTROL == {0: 9, 4: 5, 6: 0, 7: 4}
    assert a8.CONTROL_IDS == tuple(range(35, 70)) and a8.TEST1 == (6, 8) and a8.DIRECT == {7: (33, 38, 47)}
    assert (a8.MAX_SECOND, a8.MAX_SELECTED, a8.MIN_SELECTED) == (20, 4, 2)
    assert a8.SCREEN_TASKS == (0, 4, 6) and a8.SCREEN_STATES == tuple(range(10, 30))
    assert a8.FIRST_IDS == tuple(range(400, 408)) and a8.SECOND_IDS == tuple(range(408, 432))


def test_harness_references_equal_step_0():
    rows = list(csv.DictReader(open(RESULTS / "week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv")))
    for t, n in a8.HARNESS_REFERENCE.items():
        wanted = [r for r in rows if int(r["task_id"]) == t and int(r["seed"]) <= 2 and int(r["init_state"]) <= 9]
        assert len(wanted) == 30 and sum(int(float(r["success"])) for r in wanted) == n


def test_holm_needs_the_smaller_under_half_and_the_other_under_alpha():
    assert a8.holm({"a": 0.02, "b": 0.04}) == {"a": True, "b": True}
    assert a8.holm({"a": 0.04, "b": 0.02}) == {"a": True, "b": True}
    assert a8.holm({"a": 0.02, "b": 0.05}) == {"a": True, "b": False}
    assert a8.holm({"a": 0.025, "b": 0.03}) == {"a": False, "b": False}
    assert a8.holm({"a": 0.0249, "b": 0.0499}) == {"a": True, "b": True}


def test_one_test_alone_is_at_alpha():
    assert a8.holm({"a": 0.049}) == {"a": True}
    assert a8.holm({"a": 0.05}) == {"a": False}


def test_holm_ties_are_not_order_dependent():
    assert a8.holm({"b": 0.02, "a": 0.02}) == {"a": True, "b": True}
    assert a8.holm({"b": 0.03, "a": 0.03}) == {"a": False, "b": False}


@pytest.mark.parametrize("t1,t2,text", [
    (None, None, "not read"), (True, True, "fixed noise beats sampling"), (True, False, "6/8 is confirmed, but"),
    (False, True, "did not replicate"), (True, None, "too few resistant"), (None, True, "6/8 was not tested"),
    (False, False, "not shown"), (False, None, "not shown"), (None, False, "not shown")])
def test_reading_of_every_pair(t1, t2, text):
    assert text in a8.reading(t1, t2)


def test_reading_says_only_what_its_branch_may(tmp_path):
    assert "beyond task 0" not in a8.reading(True, False)


def test_gates_stop_on_harness_and_sampled_control_only(tmp_path):
    gates(tmp_path / "a", harness={7: 11})
    assert "task 7" in a8.gate(tmp_path / "a", "harness")
    gates(tmp_path / "b", harness={7: 12})
    assert a8.gate(tmp_path / "b", "harness") is None
    gates(tmp_path / "c", control={4: {"sampled": 2}})
    assert "task 4" in a8.gate(tmp_path / "c", "control")
    gates(tmp_path / "d", control={4: {"sampled": 3, "repeat": 0}})
    assert a8.gate(tmp_path / "d", "control") is None
    assert a8.read_tasks(tmp_path / "d") == [0, 6, 7]


def test_a_task_with_a_failed_repeat_control_leaves_screen_and_tests(tmp_path):
    gates(tmp_path, control={0: {"repeat": 2}})
    screen(tmp_path)
    assert all(t != 0 for t, _ in a8.second_states(tmp_path))
    assert all(t != 0 for t, _ in a8.eligible_order(tmp_path))


def test_a_task_6_control_failure_removes_test_1(tmp_path):
    gates(tmp_path, control={6: {"repeat": 2}})
    screen(tmp_path)
    for t, s in a8.eligible_order(tmp_path):
        scene(tmp_path, t, s)
    plan = a8.test_plan(tmp_path)
    assert plan["test1"] is None and plan["test2"]


def test_second_part_is_capped_in_rounds(tmp_path):
    gates(tmp_path)
    zero = {0: list(range(10, 22)), 4: list(range(10, 22)), 6: list(range(10, 22))}
    screen(tmp_path, zero)
    states = a8.second_states(tmp_path)
    assert len(states) == 20 and states[:3] == [(0, 10), (4, 10), (6, 10)]
    assert sum(t == 0 for t, _ in states) == 7 and sum(t == 6 for t, _ in states) == 6


def test_a_state_with_a_late_hit_is_not_eligible(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    write(tmp_path / "task_0" / "screen" / "per_episode_sampled_state12.csv", episodes(0, 12, a8.ALL_SCREEN_IDS, (431,)))
    assert (0, 12) not in a8.eligible_order(tmp_path) and (0, 15) in a8.eligible_order(tmp_path)


def test_a_first_part_hit_is_not_taken_on(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    assert (0, 10) not in a8.second_states(tmp_path)


def test_direct_states_need_their_own_0_of_32(tmp_path):
    gates(tmp_path)
    screen(tmp_path, direct_hits=(33,))
    order = a8.eligible_order(tmp_path)
    assert (7, 33) not in order and (7, 38) in order and (7, 47) in order


def test_eligible_order_goes_round_the_tasks(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    assert a8.eligible_order(tmp_path)[:5] == [(0, 12), (4, 11), (6, 20), (7, 33), (0, 15)]


def test_an_unfinished_screen_is_refused(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    write(tmp_path / "task_0" / "screen" / "per_episode_sampled_state12.csv", episodes(0, 12, a8.FIRST_IDS))
    with pytest.raises(ValueError, match="finish the run"):
        a8.eligible_order(tmp_path)


def test_a_screen_with_a_foreign_id_is_refused(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    write(tmp_path / "task_0" / "screen" / "per_episode_sampled_state10.csv", episodes(0, 10, a8.FIRST_IDS + (500,)))
    with pytest.raises(ValueError, match="finish the run"):
        a8.second_states(tmp_path)


def test_selection_passes_over_a_failed_scene_and_asks_for_the_next(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    assert a8.selection(tmp_path) == ([], (0, 12))
    scene(tmp_path, 0, 12, same=False)
    assert a8.selection(tmp_path) == ([], (4, 11))
    for k in ((4, 11), (6, 20), (7, 33), (0, 15)):
        scene(tmp_path, *k)
    assert a8.selection(tmp_path) == ([(4, 11), (6, 20), (7, 33), (0, 15)], None)


def test_selection_stops_at_four_without_more_checks(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    for k in a8.eligible_order(tmp_path)[:4]:
        scene(tmp_path, *k)
    assert len(a8.selection(tmp_path)[0]) == 4 and a8.selection(tmp_path)[1] is None


def test_a_scene_file_of_another_state_is_refused(tmp_path):
    scene(tmp_path, 0, 12)
    p = tmp_path / "task_0" / "scene" / "scene_state12.json"
    p.write_text(json.dumps({"task_id": 0, "init_state": 13, "same": True}))
    with pytest.raises(ValueError, match="another state"):
        a8.scene_of(tmp_path, 0, 12)


def test_ids_follow_the_final_position_after_a_replacement(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    scene(tmp_path, *a8.TEST1)
    scene(tmp_path, 0, 12, same=False)
    for k in ((4, 11), (6, 20), (7, 33), (0, 15)):
        scene(tmp_path, *k)
    t = a8.test_plan(tmp_path)["test2"]
    assert [(x[0], x[1], x[2][0]) for x in t] == [(4, 11, 1000), (6, 20, 1150), (7, 33, 1300), (0, 15, 1450)]


def test_one_eligible_state_runs_no_test_2(tmp_path):
    gates(tmp_path)
    screen(tmp_path, {0: [12]}, direct_hits=(33, 38, 47))
    scene(tmp_path, *a8.TEST1)
    scene(tmp_path, 0, 12)
    assert a8.test_plan(tmp_path) == {"test1": a8.TEST1, "test2": []}


def test_two_eligible_states_run_test_2(tmp_path):
    gates(tmp_path)
    screen(tmp_path, {0: [12], 4: [11]}, direct_hits=(33, 38, 47))
    scene(tmp_path, *a8.TEST1)
    scene(tmp_path, 0, 12)
    scene(tmp_path, 4, 11)
    assert len(a8.test_plan(tmp_path)["test2"]) == 2


def test_a_failed_scene_on_6_8_removes_test_1(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    scene(tmp_path, *a8.TEST1, same=False)
    for k in a8.eligible_order(tmp_path)[:4]:
        scene(tmp_path, *k)
    assert a8.test_plan(tmp_path)["test1"] is None


def test_a_missing_scene_check_of_6_8_is_refused(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    for k in a8.eligible_order(tmp_path)[:4]:
        scene(tmp_path, *k)
    with pytest.raises(ValueError, match="6/8"):
        a8.test_plan(tmp_path)


def test_full_run_both_pass(tmp_path):
    full_run(tmp_path, (0, 12), (1, 15))
    r = a8.analyse(tmp_path)
    assert r["tests"]["test 1"]["passes"] and r["tests"]["test 2"]["passes"]
    assert "fixed noise beats" in r["decision"]
    assert r["pooled"]["repeat"] == 15 and r["pooled"]["sampled"] == 1 and r["pooled"]["episodes"] == 600
    assert r["pooled"]["states_with_a_sampled_hit"] == 1 and r["pooled"]["states"] == 4


def test_pooled_p_is_on_all_tested_episodes(tmp_path):
    full_run(tmp_path, (0, 0), (0, 5))
    r = a8.analyse(tmp_path)
    assert r["tests"]["test 2"]["p"] == pytest.approx(comb(600, 5) / comb(1200, 5))


def test_test_1_p_known_answer(tmp_path):
    full_run(tmp_path, (0, 5), (0, 0))
    r = a8.analyse(tmp_path)
    assert r["tests"]["test 1"]["p"] == pytest.approx(comb(300, 5) / comb(600, 5))


def test_holm_is_applied_in_the_analysis(tmp_path):
    full_run(tmp_path, (0, 5), (0, 0))        # p about 0.03: under .05, not under .025
    r = a8.analyse(tmp_path)
    assert 0.025 < r["tests"]["test 1"]["p"] < 0.05 and not r["tests"]["test 1"]["passes"]
    assert "not shown" in r["decision"]


def test_alone_test_1_is_at_alpha(tmp_path):
    gates(tmp_path)
    screen(tmp_path, {0: [12]}, direct_hits=(33, 38, 47))
    scene(tmp_path, *a8.TEST1)
    scene(tmp_path, 0, 12)
    write_test(tmp_path, *a8.TEST1, a8.TEST1_IDS, 0, 5)
    r = a8.analyse(tmp_path)
    assert r["tests"]["test 1"]["passes"] and "too few resistant" in r["decision"]


def test_sampled_hits_count_against_repeat(tmp_path):
    full_run(tmp_path, (6, 8), (0, 0))
    assert not a8.analyse(tmp_path)["tests"]["test 1"]["passes"]


def test_jittered_counts_and_bound(tmp_path):
    full_run(tmp_path, (0, 12), (0, 2))
    r = a8.analyse(tmp_path)
    assert r["jittered"] == {"successes": 10, "episodes": 20}
    first = r["states"][0]
    assert first["group"] == "test 1" and first["sampled_bound_if_zero"] == pytest.approx(1 - 0.05 ** (1 / 300), abs=1e-4)
    assert len(first["confirmed"]) == 3


def test_an_unfinished_test_is_refused(tmp_path):
    full_run(tmp_path)
    write(tmp_path / "task_6" / "test" / "per_episode_repeat_state8.csv", episodes(6, 8, a8.TEST1_IDS[:-1]))
    with pytest.raises(ValueError, match="finish the run"):
        a8.analyse(tmp_path)


def test_a_missing_confirmation_is_refused(tmp_path):
    full_run(tmp_path, (0, 2), (0, 0))
    (tmp_path / "task_6" / "test" / f"confirm_repeat_n{a8.TEST1_IDS[0]}_state8.csv").unlink()
    with pytest.raises(FileNotFoundError):
        a8.analyse(tmp_path)


def test_failed_harness_is_not_read(tmp_path):
    gates(tmp_path, harness={0: 14})
    assert a8.analyse(tmp_path)["decision"] == "not read: a harness failed"


def test_failed_sampled_control_is_not_read(tmp_path):
    gates(tmp_path, control={7: {"sampled": 2}})
    assert a8.analyse(tmp_path)["decision"] == "not read: a sampled control failed"


@pytest.mark.parametrize("change,match", [
    (dict(git_dirty=True), "dirty"), (dict(git_dirty_noise_reachability=True), "dirty"),
    (dict(git_commit="d" * 40), "commits"), (dict(settings={"n_action_steps": 50}), "settings"),
    (dict(settings={"max_steps": 300}), "settings"), (dict(settings={"policy_path": "/s/other"}), "policy"),
    (dict(settings={"task_id": 4}), "task 4")])
def test_provenance_refuses(tmp_path, change, match):
    gates(tmp_path)
    provenance(tmp_path / "task_0" / "scene", 0, **change)
    with pytest.raises(ValueError, match=match):
        a8.check_provenance(tmp_path)


def test_provenance_refuses_another_revision(tmp_path):
    gates(tmp_path)
    for f in tmp_path.rglob("provenance.jsonl"):
        f.write_text(f.read_text().replace(a8.POLICY_REVISION, "other"))
    with pytest.raises(ValueError, match="revision"):
        a8.check_provenance(tmp_path)


def test_report_and_cli(tmp_path, capsys):
    full_run(tmp_path, (0, 12), (1, 15))
    out = tmp_path / "r.txt"
    assert a8.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert "test 1, task 6 state 8: sampled 0/300" in text and "repeat 12/300" in text and "decision: fixed noise" in text
    assert json.loads((tmp_path / "r.txt.json").read_text())["pooled"]["repeat"] == 15
    capsys.readouterr()
    a8.main(["--root", str(tmp_path), "--tests"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "6 8 100-399" and lines[1] == "0 12 1000-1149" and lines[4] == "7 33 1450-1599"
    a8.main(["--root", str(tmp_path), "--hits"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[:3] == ["6 8 100", "6 8 101", "6 8 102"] and lines[3:] == ["0 12 1000", "0 12 1001", "0 12 1002"]
    a8.main(["--root", str(tmp_path), "--order"])
    assert capsys.readouterr().out.splitlines()[0] == "0 12"
    a8.main(["--root", str(tmp_path), "--second"])
    assert capsys.readouterr().out.splitlines()[0] == "0 12"


def test_gate_cli_exit_code(tmp_path, capsys):
    gates(tmp_path, harness={0: 14})
    assert a8.main(["--root", str(tmp_path), "--gate", "harness"]) == 1
    assert "STOP" in capsys.readouterr().out


def test_descriptive_fields_of_the_report(tmp_path):
    gates(tmp_path, control={0: {"sampled": 12, "repeat": 9}})
    screen(tmp_path)
    scene(tmp_path, *a8.TEST1)
    scene(tmp_path, 0, 12, same=False)
    for k in ((4, 11), (6, 20), (7, 33), (0, 15)):
        scene(tmp_path, *k)
    for i, (t, s, ids) in enumerate(a8.test_plan(tmp_path)["test2"]):
        write_test(tmp_path, t, s, ids)
    write_test(tmp_path, *a8.TEST1, a8.TEST1_IDS)
    r = a8.analyse(tmp_path)
    assert r["control"][0] == {"sampled": 12, "repeat": 9}
    assert r["screen"]["first_part_zero"] == {0: 2, 4: 1, 6: 1} and r["screen"]["second_part"] == 4
    assert r["screen"]["scene_failed"] == [[0, 12]]
    assert r["test1_status"] == "run"


def test_a_lost_test_1_is_explained(tmp_path):
    gates(tmp_path)
    screen(tmp_path)
    scene(tmp_path, *a8.TEST1, same=False)
    for k in a8.eligible_order(tmp_path)[:4]:
        scene(tmp_path, *k)
    for t, s, ids in a8.test_plan(tmp_path)["test2"]:
        write_test(tmp_path, t, s, ids)
    r = a8.analyse(tmp_path)
    assert r["test1_status"] == "not run: the scene moved" and "test 1 not run: the scene moved" in a8.format_report(r)


def test_test_1_lost_at_the_control_is_explained(tmp_path):
    gates(tmp_path, control={6: {"repeat": 2}})
    screen(tmp_path)
    for k in a8.eligible_order(tmp_path)[:4]:
        scene(tmp_path, *k)
    for t, s, ids in a8.test_plan(tmp_path)["test2"]:
        write_test(tmp_path, t, s, ids)
    assert a8.analyse(tmp_path)["test1_status"] == "not run: task 6's repeat control failed"


def test_packages_that_change_between_sessions_are_refused(tmp_path):
    gates(tmp_path)
    provenance(tmp_path / "task_0" / "test", 0, packages={"torch": {"version": "other"}})
    with pytest.raises(ValueError, match="packages"):
        a8.check_provenance(tmp_path)
