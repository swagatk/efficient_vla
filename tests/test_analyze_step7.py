import csv
import json
import os
import sys
from math import comb
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))

import analyze_step7 as a7  # noqa: E402

RESULTS = Path(os.environ.get("RESULTS_ROOT", ROOT))
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
         "settings": {"task_id": task, "n_action_steps": 10, "max_steps": 520, "policy_path": f"/s/{a7.POLICY_REVISION}"}}
    r["settings"].update(change.pop("settings", {}))
    r.update(change)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def confirm(folder, family, k, state, wins=2):
    write(folder / f"confirm_{family}_n{k}_state{state}.csv",
          [{"task_id": 0, "seed": s, "init_state": state, "success": int(i < wins), "steps": 520}
           for i, s in enumerate((5000, 5001, 5002, 5003))])


def run(tmp_path, hits=None, control=None, harness=None):
    """A complete run folder. hits: {task: {family: ids}}. control: {task: {family: count}}. harness: {task: count}."""
    hits, control, harness = hits or {}, control or {}, harness or {}
    for t, s in a7.SYSTEMATIC.items():
        folder = tmp_path / f"task_{t}"
        provenance(folder, t)
        for seed in range(3):
            n = harness.get(t, a7.HARNESS_REFERENCE[t])
            rows = [{"task_id": t, "seed": seed, "init_state": i,
                     "success": int(seed * 10 + i < n), "steps": 520} for i in range(10)]
            write(folder / "harness" / f"seed{seed}" / "per_episode_base.csv", rows)
            provenance(folder / "harness" / f"seed{seed}", t)
        for f in a7.FAMILIES:
            write(folder / f"per_episode_{f}_state{a7.CONTROL[t]}.csv",
                  episodes(t, a7.CONTROL[t], range(35), range(control.get(t, {}).get(f, 10))))
            ids = hits.get(t, {}).get(f, ())
            write(folder / f"per_episode_{f}_state{s}.csv", episodes(t, s, range(70), ids))
            if f != "sampled":
                for k in sorted(ids)[:3]:
                    confirm(folder, f, k, s)
    return tmp_path


def test_settings_match_the_protocol():
    assert a7.SYSTEMATIC == {4: 2, 6: 8, 8: 4} and a7.CONTROL == {4: 5, 6: 0, 8: 8}
    assert a7.HARNESS_REFERENCE == {4: 14, 6: 20, 8: 15} and a7.HARNESS_TOLERANCE == 6
    counts = {}
    for r in csv.DictReader(open(RESULTS / "week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv")):
        t = int(r["task_id"])
        if t in a7.SYSTEMATIC and int(r["seed"]) in (0, 1, 2):
            counts[t] = counts.get(t, 0) + int(r["success"])
    assert counts == a7.HARNESS_REFERENCE


def test_controls_follow_the_written_rule():
    rows = {}
    for f, seeds in (("week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv", range(3)),
                     ("week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv", range(3, 16)),
                     ("week6/noise_reachability_step6_kaggle/run/eval/per_episode_base.csv", range(16, 32))):
        for r in csv.DictReader(open(RESULTS / f)):
            if int(r["seed"]) in seeds:
                k = (int(r["task_id"]), int(r["init_state"]))
                rows[k] = rows.get(k, 0) + int(r["success"])
    low = {4: (1, 5, 6, 7, 8), 6: (0, 1, 9), 8: (1, 2, 3, 8, 9)}
    for t, states in low.items():
        best = min(states, key=lambda s: (abs(rows[(t, s)] - 32 / 3), s))
        assert best == a7.CONTROL[t]
    assert [rows[(t, s)] for t, s in a7.CONTROL.items()] == [9, 11, 10]


def test_fisher_known_values():
    assert a7.fisher_greater(5, 0, 210) == pytest.approx(comb(210, 5) / comb(420, 5))
    assert a7.fisher_greater(5, 0, 210) < 0.05 < a7.fisher_greater(4, 0, 210)
    assert a7.fisher_greater(7, 1, 210) < 0.05 < a7.fisher_greater(6, 1, 210)
    assert a7.fisher_greater(0, 0, 210) == 1
    assert a7.fisher_greater(3, 1, 4) == pytest.approx(comb(4, 3) * comb(4, 1) / comb(8, 4) + 1 / comb(8, 4))
    assert a7.fisher_greater(1, 0, 1) == pytest.approx(0.5)


@pytest.mark.parametrize("passes, only, reached, text", [
    (True, 3, 3, "extends beyond task 0"), (True, 2, 3, "extends beyond task 0"), (True, 1, 3, "too few states"),
    (True, 0, 1, "too few states"), (False, 3, 3, "existence only"), (False, 0, 1, "existence only"),
    (False, 0, 0, "unreached at this depth")])
def test_reading_order_and_edges(passes, only, reached, text):
    assert text in a7.reading(passes, only, reached)


def test_no_fixed_family_read_is_not_a_negative_result():
    assert "not read" in a7.reading(False, 0, 0, False) and "not read" in a7.reading(True, 3, 3, False)


def test_clopper_pearson_known_values():
    lo, hi = a7.clopper_pearson(1, 48)
    assert lo == pytest.approx(0.000527, abs=1e-5) and hi == pytest.approx(0.1107, abs=1e-3)
    assert a7.clopper_pearson(0, 210) == (0.0, pytest.approx(1 - 0.025 ** (1 / 210), abs=1e-9))
    assert a7.clopper_pearson(210, 210) == (pytest.approx(0.025 ** (1 / 210), abs=1e-9), 1.0)
    lo, hi = a7.clopper_pearson(5, 210)
    assert lo < 5 / 210 < hi


def test_rate_passes_and_all_states_fixed_only(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1)}, 6: {"repeat": (2, 3)}, 8: {"repeat": (4,)}}))
    q = r["rate"]
    assert (q["repeat"], q["sampled"], q["episodes"], q["passes"]) == (5, 0, 210, True)
    assert q["p"] == pytest.approx(a7.fisher_greater(5, 0, 210)) and q["repeat_noises_that_hit"] == 5
    assert q["repeat_interval"] == a7.clopper_pearson(5, 210) and q["sampled_interval"] == a7.clopper_pearson(0, 210)
    assert r["spread"] == 3 and r["fixed_only_states"] == 3 and "extends beyond" in r["decision"]
    assert r["states"][4]["repeat_hit_ids"] == [0, 1]


def test_sampled_hits_do_not_decide_the_reading_when_the_rate_passes(tmp_path):
    ids = tuple(range(8))
    r = a7.analyse(run(tmp_path, {t: {"repeat": ids, "sampled": (50,)} for t in (4, 6)} | {8: {"repeat": ids}}))
    assert r["rate"]["passes"] and r["spread"] == 3 and r["fixed_only_states"] == 1
    assert "extends beyond" in r["decision"]


def test_the_secondary_rate_leaves_out_task_8(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0,)}, 6: {"repeat": (1, 2), "sampled": (3,)}, 8: {"repeat": (4, 5, 6)}}))
    s = r["rate_without_task_8"]
    assert (s["repeat"], s["sampled"], s["episodes"]) == (3, 1, 140)
    assert s["p"] == pytest.approx(a7.fisher_greater(3, 1, 140)) and r["rate"]["repeat"] == 6
    assert "secondary, without task 8 (stove not fixed): repeat 3/140 against sampled 1/140" in a7.format_report(r)


def test_the_same_noise_on_two_states_counts_once(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1, 2)}, 6: {"repeat": (0, 1)}}))
    assert r["rate"]["repeat"] == 5 and r["rate"]["repeat_noises_that_hit"] == 3


def test_one_full_hit_does_not_make_a_concentrated_rate_general(tmp_path):
    r = a7.analyse(run(tmp_path, {8: {"repeat": (0, 1, 2, 3, 4)}, 4: {"full": (1,)}}))
    assert r["rate"]["passes"] and r["spread"] == 1 and "too few states" in r["decision"]


def test_one_hit_short_of_the_rate_is_existence_only(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1)}, 6: {"repeat": (2,)}, 8: {"repeat": (4,)}}))
    assert not r["rate"]["passes"] and r["rate"]["repeat"] == 4 and "existence only" in r["decision"]


def test_a_sampled_hit_stops_a_state_being_fixed_only_and_raises_the_bar(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1, 2), "sampled": (9,)}, 6: {"repeat": (2, 3)}, 8: {"repeat": (4,)}}))
    assert r["rate"]["sampled"] == 1 and r["rate"]["repeat"] == 6 and not r["rate"]["passes"]
    assert r["fixed_only_states"] == 2 and not r["states"][4]["fixed_only"] and r["spread"] == 3


def test_rate_on_a_single_state_is_called_concentrated(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1, 2, 3, 4)}}))
    assert r["rate"]["passes"] and r["spread"] == 1 and "too few states" in r["decision"]


def test_sampled_hits_alone_are_not_a_reach(tmp_path):
    r = a7.analyse(run(tmp_path, {8: {"sampled": (1,)}}))
    assert r["states"][8]["class"] == "sampled" and r["fixed_only_states"] == 0
    assert "unreached at this depth" in r["decision"]


def test_full_counts_as_a_reach_but_not_in_the_rate(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"full": (0, 1, 2, 3, 4, 5, 6)}, 6: {"full": (1,)}}))
    assert r["rate"]["repeat"] == 0 and not r["rate"]["passes"] and r["fixed_only_states"] == 2 and r["spread"] == 0
    assert "existence only" in r["decision"]


def test_a_task_with_repeat_unread_leaves_the_pool(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0, 1, 2, 3, 4)}, 6: {"repeat": (1,)}}, control={6: {"repeat": 2}}))
    assert r["rate"]["episodes"] == 140 and r["rate"]["repeat"] == 5
    assert r["family_read"][6]["repeat"] is False and r["unread_hits"] == {"task 6": {"repeat": 1}}
    assert "repeat" not in r["states"][6]["per_family"] and r["spread"] == 1


def test_control_of_exactly_3_is_read(tmp_path):
    r = a7.analyse(run(tmp_path, {}, control={8: {"sampled": 3, "repeat": 3, "full": 3}}))
    assert r["family_read"][8] == {"sampled": True, "full": True, "repeat": True}


@pytest.mark.parametrize("n, ok", [(14 + 6, True), (14 + 7, False), (14 - 6, True), (14 - 7, False)])
def test_harness_edges(tmp_path, n, ok):
    root = run(tmp_path, harness={4: n})
    assert (a7.gate(root, "harness") is None) == ok
    assert ("not read" in a7.analyse(root)["decision"]) != ok


def test_sampled_control_stops_the_run(tmp_path):
    root = run(tmp_path, control={6: {"sampled": 2}})
    assert "task 6" in a7.gate(root, "control") and "not read" in a7.analyse(root)["decision"]
    assert a7.main(["--root", str(root), "--gate", "control"]) == 1
    assert a7.main(["--root", str(root), "--gate", "harness"]) == 0


def test_control_gate_passes_at_3(tmp_path):
    assert a7.gate(run(tmp_path, control={6: {"sampled": 3}}), "control") is None


def test_a_missing_episode_is_refused(tmp_path):
    root = run(tmp_path)
    f = root / "task_6" / "per_episode_repeat_state8.csv"
    write(f, episodes(6, 8, range(69)))
    with pytest.raises(ValueError, match="finish the run"):
        a7.analyse(root)


def test_a_missing_confirmation_is_refused(tmp_path):
    root = run(tmp_path, {4: {"repeat": (0,)}})
    (root / "task_4" / "confirm_repeat_n0_state2.csv").unlink()
    with pytest.raises(FileNotFoundError):
        a7.analyse(root)


def test_incomplete_or_wrong_harness_is_refused(tmp_path):
    root = run(tmp_path)
    (root / "task_8" / "harness" / "seed2" / "per_episode_base.csv").unlink()
    with pytest.raises(ValueError, match="harness task 8"):
        a7.harness_check(root, 8)


def test_harness_rows_of_another_task_are_refused(tmp_path):
    root = run(tmp_path)
    f = root / "task_8" / "harness" / "seed0" / "per_episode_base.csv"
    rows = list(csv.DictReader(open(f)))
    write(f, [dict(r, task_id=4) for r in rows])
    with pytest.raises(ValueError, match="another task"):
        a7.harness_check(root, 8)


@pytest.mark.parametrize("change, message", [
    ({"git_commit": "d" * 40}, "commits"), ({"git_dirty": True}, "dirty"), ({"git_dirty_noise_reachability": True}, "dirty"),
    ({"settings": {"n_action_steps": 1}}, "settings"), ({"settings": {"max_steps": 300}}, "settings"),
    ({"settings": {"policy_path": "/other"}}, "policy paths"), ({"settings": {"task_id": 0}}, "provenance: task")])
def test_provenance_refusals(tmp_path, change, message):
    root = run(tmp_path)
    provenance(root / "task_6", 6, **change)
    with pytest.raises(ValueError, match=message):
        a7.check_provenance(root)


def test_another_revision_is_refused(tmp_path):
    for t in a7.SYSTEMATIC:
        for f in (tmp_path / f"task_{t}").rglob("provenance.jsonl"):
            f.unlink()
    root = run(tmp_path)
    for f in root.rglob("provenance.jsonl"):
        f.write_text(f.read_text().replace(a7.POLICY_REVISION, "b" * 40))
    with pytest.raises(ValueError, match="not revision"):
        a7.check_provenance(root)


def test_hits_lists_first_three_of_read_fixed_families(tmp_path, capsys):
    root = run(tmp_path, {4: {"repeat": (9, 2, 5, 7), "sampled": (1,)}, 6: {"full": (3,), "repeat": (4,)}},
               control={6: {"repeat": 1}})
    assert a7.main(["--root", str(root), "--hits"]) == 0
    assert capsys.readouterr().out.split("\n")[:-1] == ["4 repeat 2 2", "4 repeat 2 5", "4 repeat 2 7", "6 full 8 3"]


def test_report_and_out_files(tmp_path, capsys):
    root = run(tmp_path / "r", {4: {"repeat": (0, 1)}, 6: {"repeat": (2, 3)}, 8: {"repeat": (4,)}})
    out = tmp_path / "report.txt"
    assert a7.main(["--root", str(root), "--out", str(out)]) == 0
    text = out.read_text()
    assert capsys.readouterr().out == text
    assert "rate: repeat 5/210 against sampled 0/210" in text and "states with a repeat hit: 3 of 3 (required 2)" in text
    assert "95% intervals: repeat " in text and "5 distinct repeat noises hit" in text
    assert "control task 4, state 5, of 35" in text and "harness task 6: base 20/30 against 20 (gap +0, ok)" in text
    assert json.loads((tmp_path / "report.txt.json").read_text())["rate"]["repeat"] == 5


def test_report_when_not_read(tmp_path):
    text = a7.format_report(a7.analyse(run(tmp_path, harness={4: 0})))
    assert "FAILED" in text and "not read" in text and "rate:" not in text


def test_a_single_fixed_hit_counts_as_reached_and_the_bound_appears_only_with_no_hits(tmp_path):
    r = a7.analyse(run(tmp_path, {6: {"full": (1,)}}, control={4: {"sampled": 9}}))
    assert "existence only" in r["decision"] and r["states"][6]["upper_if_zero"] is None
    assert r["states"][4]["upper_if_zero"] == pytest.approx(a7.zero_upper(210))
    assert r["control"][4]["sampled"] == 9
    text = a7.format_report(r)
    assert f"bound on the mixed rate {100 * a7.zero_upper(210):.1f}%" in text and text.count("bound on the mixed rate") == 2


def test_no_pool_when_repeat_is_unread_everywhere(tmp_path):
    r = a7.analyse(run(tmp_path, {}, control={t: {"repeat": 0} for t in a7.SYSTEMATIC}))
    assert r["rate"]["episodes"] == 0 and r["rate"]["p"] == 1 and not r["rate"]["passes"]
    assert r["rate"]["repeat_interval"] is None and "95% intervals" not in a7.format_report(r)
    assert r["rate_without_task_8"]["p"] == 1


def test_no_fixed_family_read_gives_not_read(tmp_path):
    r = a7.analyse(run(tmp_path, {}, control={t: {"repeat": 0, "full": 0} for t in a7.SYSTEMATIC}))
    assert "no fixed family" in r["decision"]


def test_a_stopped_harness_reads_without_control_files(tmp_path):
    root = run(tmp_path, harness={4: 0})
    (root / "task_4" / "per_episode_sampled_state5.csv").unlink()
    r = a7.analyse(root)
    assert r["decision"] == "not read: a harness failed" and "FAILED" in a7.format_report(r)


def test_gate_refuses_a_wrong_setting_early(tmp_path):
    root = run(tmp_path)
    provenance(root / "task_6", 6, settings={"n_action_steps": 1})
    with pytest.raises(ValueError, match="settings"):
        a7.gate(root, "harness")


def test_gate_works_before_the_task_folders_have_search_records(tmp_path):
    root = run(tmp_path)
    for t in a7.SYSTEMATIC:
        (root / f"task_{t}" / "provenance.jsonl").unlink()
    assert a7.gate(root, "harness") is None


def test_root_is_required():
    with pytest.raises(SystemExit):
        a7.main([])


def test_single_hits_count_towards_spread(tmp_path):
    r = a7.analyse(run(tmp_path, {4: {"repeat": (0,)}, 6: {"repeat": (1,)}}))
    assert r["spread"] == 2


def test_one_fixed_family_read_on_one_task_is_enough_to_read(tmp_path):
    control = {t: {"repeat": 0, "full": 0} for t in a7.SYSTEMATIC}
    control[6] = {"repeat": 0, "full": 3}
    r = a7.analyse(run(tmp_path, {}, control=control))
    assert "unreached at this depth" in r["decision"]
