import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step6 as a  # noqa: E402
from reachability import pool  # noqa: E402

DATA = Path(os.environ.get("RESULTS_ROOT", ROOT))     # the mutation check copies no week6/
STEP0 = DATA / "week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv"
STEP1 = DATA / "week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv"
STEP4 = DATA / "week6/noise_reachability_step4_kaggle/run/eval"


def real_labels():
    return {k: sum(v) for k, v in pool([str(STEP0), str(STEP1)], list(a.ALL_TASKS), list(a.LABEL_SEEDS)).items()}


def episodes(tasks, rate):
    """Episode dict with rate(task, state) successes of 16 on seeds 16-31."""
    out = {}
    for t in tasks:
        for i in a.INIT_STATES:
            n = rate(t, i)
            for j, s in enumerate(a.EVAL_SEEDS):
                out[(t, s, i)] = int(j < n)
    return out


def run(base_rate, b_rate, labels=None):
    labels = labels or real_labels()
    return a.analyse(labels, episodes(a.NEW_TASKS, base_rate), episodes(a.NEW_TASKS, b_rate),
                     episodes((0,), base_rate), episodes((0,), b_rate))


def test_labels_match_the_table_in_the_readme():
    labels = real_labels()
    for t, want in {4: [13, 4, 0, 16, 16, 5, 1, 8, 5, 11], 6: [7, 5, 15, 9, 9, 16, 12, 16, 0, 8],
                    7: [8, 16, 3, 6, 7, 1, 13, 16, 4, 16], 8: [11, 8, 2, 1, 0, 10, 14, 13, 5, 6]}.items():
        assert [labels[(t, i)] for i in range(10)] == want


def test_groups_of_the_protocol():
    assert a.groups(real_labels(), a.NEW_TASKS) == {4: ([2], [1, 5, 6, 7, 8]), 6: ([8], [0, 1, 9]),
                                                    8: ([4], [1, 2, 3, 8, 9])}


def test_groups_boundary_8_in_9_out_and_task_without_systematic_left_out():
    labels = {(1, i): 16 for i in a.INIT_STATES}
    labels[(1, 0)], labels[(1, 1)], labels[(1, 2)], labels[(1, 3)] = 0, 1, 8, 9
    assert a.groups(labels, (1,)) == {1: ([0], [1, 2])}
    labels[(1, 0)] = 2
    assert a.groups(labels, (1,)) == {}


def test_step4_files_as_task_0_reference():
    base = a.load(STEP4 / "per_episode_base.csv", (0,))
    b = a.load(STEP4 / "per_episode_B.csv", (0,))
    a.check_complete(base, (0,), "base")
    a.check_complete(b, (0,), "B")
    assert (sum(base.values()), sum(b.values())) == (37, 64)


def test_uneven_gain_known_answer_by_hand():
    # Task 1: systematic 0 gains 0; low 1, 2 gain 1, 0.5 -> D 0.75. Other assignments: 1 -> 0.25 - 1 = -0.75, 2 -> 0.
    gains = {(1, 0): 0.0, (1, 1): 1.0, (1, 2): 0.5}
    r = a.uneven_gain(gains, {1: ([0], [1, 2])})
    assert r["D"] == pytest.approx(0.75) and r["assignments"] == 3 and r["p"] == pytest.approx(1 / 3)


def test_uneven_gain_two_strata_average_and_enumerate():
    gains = {(1, 0): 0.0, (1, 1): 1.0, (2, 0): 0.0, (2, 1): 0.5}
    r = a.uneven_gain(gains, {1: ([0], [1]), 2: ([0], [1])})
    assert r["D"] == pytest.approx(0.75) and r["assignments"] == 4 and r["p"] == pytest.approx(0.25)


def test_uneven_gain_ties_count_against():
    gains = {(1, i): 0.25 for i in range(3)}
    assert a.uneven_gain(gains, {1: ([0], [1, 2])})["p"] == 1


def test_uneven_gain_several_systematic_states():
    gains = {(1, 0): 0.0, (1, 1): 0.0, (1, 2): 1.0}
    r = a.uneven_gain(gains, {1: ([0, 1], [2])})
    assert r["D"] == pytest.approx(1.0) and r["assignments"] == 3 and r["p"] == pytest.approx(1 / 3)


def test_primary_has_144_assignments_and_best_p_is_one_over_144():
    labels = real_labels()
    r = run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)
    assert r["primary"]["assignments"] == 144 and r["primary"]["p"] == pytest.approx(1 / 144)


def test_task_level_gain_alone_gives_no_uneven_gain():
    r = run(lambda t, i: 0, lambda t, i: 8)
    assert r["primary"]["D"] == pytest.approx(0) and r["primary"]["p"] == 1 and not r["uneven"]
    assert r["reading"].startswith("labels hold, uneven gain not replicated")


def test_reading_replicates():
    labels = real_labels()
    r = run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)
    assert r["labels_hold"] and r["informative"] and r["uneven"]
    assert r["primary"]["D"] == pytest.approx(0.5)
    assert r["reading"].startswith("labels hold, uneven gain holds")


def test_label_hits_edge():
    hits = {(4, 2): 2, (6, 8): 2, (8, 4): 0}
    r = run(lambda t, i: hits.get((t, i), 0), lambda t, i: 16)
    assert r["label_hits"] == 4 and r["label_episodes"] == 48 and r["labels_hold"]
    hits[(8, 4)] = 1
    r = run(lambda t, i: hits.get((t, i), 0), lambda t, i: 16)
    assert r["label_hits"] == 5 and not r["labels_hold"] and r["reading"].startswith("labels fail")


def test_informative_edge():
    # 13 low states; gain 2/16 on all of them = 0.125 passes; 1/16 on all = 0.0625 does not.
    labels = real_labels()
    assert run(lambda t, i: 0, lambda t, i: 2 if labels[(t, i)] else 0)["informative"]
    r = run(lambda t, i: 0, lambda t, i: 1 if labels[(t, i)] else 0)
    assert not r["informative"] and r["reading"].startswith("not informative")


def test_informative_exactly_at_threshold():
    labels = real_labels()
    low = a.groups(labels, a.NEW_TASKS)
    states = [(t, i) for t in low for i in low[t][1]]
    assert len(states) == 13
    # A mean of 0.10 needs a total of 20.8 successes over 13 states: 21 passes, 20 does not.
    r = run(lambda t, i: 0, lambda t, i: 2 if (t, i) in states[:8] else 1 if (t, i) in states else 0)
    assert r["low_gain"] == pytest.approx(21 / 13 / 16) and r["informative"]
    r = run(lambda t, i: 0, lambda t, i: 2 if (t, i) in states[:7] else 1 if (t, i) in states else 0)
    assert r["low_gain"] == pytest.approx(20 / 13 / 16) and not r["informative"]


def test_uneven_needs_both_size_and_p():
    assert a.reading(True, True, True).startswith("labels hold, uneven gain holds")
    labels = real_labels()
    # Small but perfectly ordered: every low state gains 1/16, systematic 0. D = 0.0625 < 0.10.
    r = run(lambda t, i: 0, lambda t, i: 1 if labels[(t, i)] else 0)
    assert r["primary"]["D"] == pytest.approx(1 / 16) and not r["uneven"]


def test_reading_order_labels_first():
    assert a.reading(False, False, False).startswith("labels fail")
    assert a.reading(False, True, True).startswith("labels fail")
    assert a.reading(True, False, True).startswith("not informative")


def test_harness_edges():
    labels = real_labels()
    counts = {(t, i): labels[(t, i)] for t in a.NEW_TASKS for i in a.INIT_STATES}
    counts[(4, 3)] -= 16
    counts[(4, 4)] -= 2
    r = a.harness(counts, labels, a.NEW_TASKS)
    assert r[4]["gap"] == -18 and r[4]["ok"] and r[6]["gap"] == 0
    counts[(4, 0)] -= 1
    assert not a.harness(counts, labels, a.NEW_TASKS)[4]["ok"]
    counts[(4, 0)] += 37
    assert a.harness(counts, labels, a.NEW_TASKS)[4]["gap"] == 18 and a.harness(counts, labels, a.NEW_TASKS)[4]["ok"]
    assert a.harness(counts, labels, a.NEW_TASKS)[4]["seeds_0_15"] == 79


def test_report_flags_failed_harness():
    r = run(lambda t, i: 16, lambda t, i: 16)
    assert not r["harness_ok"]
    assert "HARNESS FAILED" in a.format_report(r)


def test_refusals(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("task_id,seed,init_state,success\n4,16,0,1\n4,16,0,0\n")
    with pytest.raises(ValueError, match="twice"):
        a.load(p, a.NEW_TASKS)
    p.write_text("task_id,seed,init_state,success\n1,16,0,1\n")
    with pytest.raises(ValueError, match="task 1"):
        a.load(p, a.NEW_TASKS)
    eps = episodes(a.NEW_TASKS, lambda t, i: 1)
    del eps[(4, 16, 0)]
    with pytest.raises(ValueError, match="finish the run"):
        a.check_complete(eps, a.NEW_TASKS, "base")
    eps = episodes(a.NEW_TASKS, lambda t, i: 1)
    eps[(4, 3, 0)] = 1
    with pytest.raises(ValueError, match="1 others"):
        a.check_complete(eps, a.NEW_TASKS, "base")


def test_gain_uses_b_minus_base_over_16_and_includes_task_0():
    r = run(lambda t, i: 2, lambda t, i: 6)
    rows = r["per_state"]
    assert len(rows) == 50 and {x["gain"] for x in rows} == {0.25}
    assert rows[0] == {"task": 0, "init_state": 0, "seeds_0_15": 2, "base": 2, "b": 6, "gain": 0.25}
    assert r["secondary"]["per_task"]["0"]["run"] == 60 and r["secondary"]["per_task"]["0"]["ref"] == 20
    assert r["secondary"]["with_task_0"]["assignments"] == 144 * 56


def test_ranks_and_spearman_known_answers():
    assert a.ranks([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]
    assert a.ranks([3, 1, 2]) == [3, 1, 2]
    assert a.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1)
    assert a.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    # By hand: ranks 1, 2, 3, 4 and 1, 3.5, 3.5, 2: sum of products 1.5, sums of squares 5 and 4.5.
    assert a.spearman([1, 2, 3, 4], [1, 5, 5, 2]) == pytest.approx(1.5 / (5 * 4.5) ** 0.5)
    assert a.spearman([1, 1], [1, 2]) is None
    assert a.spearman([1, 2], [1, 1]) is None


def test_main_writes_report_and_json(tmp_path, monkeypatch):
    labels = real_labels()
    fields = ["task_id", "seed", "init_state", "success", "steps"]

    def write(eps, name):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for (t, s, i), ok in eps.items():
                w.writerow({"task_id": t, "seed": s, "init_state": i, "success": ok, "steps": 520})
        return str(tmp_path / name)

    base = write(episodes(a.NEW_TASKS, lambda t, i: labels[(t, i)]), "base.csv")
    b = write(episodes(a.NEW_TASKS, lambda t, i: min(16, labels[(t, i)] + 8) if labels[(t, i)] else 0), "b.csv")
    out = tmp_path / "report.txt"
    monkeypatch.setattr(sys, "argv", ["x", "--labels", str(STEP0), str(STEP1), "--base", base, "--b", b, "--task0_base",
                                      str(STEP4 / "per_episode_base.csv"), "--task0_b", str(STEP4 / "per_episode_B.csv"),
                                      "--out", str(out)])
    assert a.main() == 0
    assert "labels hold, uneven gain holds" in out.read_text()
    j = json.loads(Path(str(out) + ".json").read_text())
    assert len(j["inputs"]) == 6 and j["harness_ok"] and j["secondary"]["per_task"]["0"]["run"] == 64


def test_size_rule_at_and_below_the_threshold(monkeypatch):
    labels = real_labels()
    primary = {"D": a.MIN_GAIN, "p": 0.01, "assignments": 144}
    monkeypatch.setattr(a, "uneven_gain", lambda gains, grp: primary)
    assert run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)["uneven"]
    primary = {"D": a.MIN_GAIN - 0.001, "p": 0.01, "assignments": 144}
    assert not run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)["uneven"]
    primary = {"D": 0.5, "p": a.ALPHA, "assignments": 144}
    assert not run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)["uneven"]


def test_harness_only_exit_codes(tmp_path, monkeypatch, capsys):
    labels = real_labels()
    fields = ["task_id", "seed", "init_state", "success"]

    def write(rate):
        path = tmp_path / "base.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for (t, s, i), ok in episodes(a.NEW_TASKS, rate).items():
                w.writerow({"task_id": t, "seed": s, "init_state": i, "success": ok})
        return str(path)

    argv = ["x", "--harness_only", "--labels", str(STEP0), str(STEP1), "--base"]
    monkeypatch.setattr(sys, "argv", argv + [write(lambda t, i: labels[(t, i)])])
    assert a.main() == 0
    assert "harness task 4: base 79/160 against 79" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", argv + [write(lambda t, i: 0)])
    assert a.main() == 1
    assert "FAILED" in capsys.readouterr().out


def test_full_run_needs_all_inputs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--labels", str(STEP0), "--base", "b.csv"])
    with pytest.raises(SystemExit):
        a.main()


def test_several_systematic_states_are_averaged():
    gains = {(1, 0): 0.2, (1, 1): 0.4, (1, 2): 1.0}
    assert a.uneven_gain(gains, {1: ([0, 1], [2])})["D"] == pytest.approx(0.7)


def test_one_failing_task_fails_the_harness():
    labels = real_labels()

    def rate(t, i):
        return 0 if t == 4 else labels[(t, i)]

    r = run(rate, lambda t, i: 16)
    assert not r["harness_ok"] and r["harness"]["6"]["ok"]
    assert a.harness_gate(labels, episodes(a.NEW_TASKS, rate))[1] is False


def test_run_needs_every_input_not_just_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--labels", str(STEP0), "--base", "b.csv", "--b", "c.csv", "--task0_base", "d.csv",
                                      "--task0_b", "e.csv"])
    with pytest.raises(SystemExit):
        a.main()


def test_one_state_above_the_limit_fails_the_labels():
    hits = {(4, 2): 2, (6, 8): 0, (8, 4): 0}
    assert run(lambda t, i: hits.get((t, i), 0), lambda t, i: 16)["labels_hold"]
    hits[(4, 2)] = 3
    r = run(lambda t, i: hits.get((t, i), 0), lambda t, i: 16)
    assert r["label_hits"] == 3 and not r["labels_hold"] and r["label_hits_per_state"]["4/2"] == 3


def test_label_interval_known_answer():
    r = run(lambda t, i: 0, lambda t, i: 16)
    assert r["label_rate_ci95"][0] == 0 and r["label_rate_ci95"][1] == pytest.approx(1 - 0.025 ** (1 / 48))


def test_logit_gain_known_answer_and_floor():
    assert a.logit_gain(8, 8) == 0
    assert a.logit_gain(16, 0) == pytest.approx(2 * math.log(33))
    assert a.logit_gain(1, 0) == pytest.approx(math.log(1.5 / 15.5) - math.log(0.5 / 16.5))


def test_log_odds_d_uses_the_counts():
    labels = real_labels()
    r = run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)
    assert r["secondary"]["log_odds"]["D"] == pytest.approx(a.logit_gain(8, 0))


def test_task_gaps_in_the_primary_result():
    labels = real_labels()
    r = run(lambda t, i: 0, lambda t, i: 8 if labels[(t, i)] else 0)
    assert r["primary"]["per_task"] == {"4": 0.5, "6": 0.5, "8": 0.5}


def test_bootstrap_interval_is_a_point_when_every_seed_agrees():
    labels = real_labels()
    grp = a.groups(labels, a.NEW_TASKS)
    base, b = episodes(a.NEW_TASKS, lambda t, i: 0), episodes(a.NEW_TASKS, lambda t, i: 16 if labels[(t, i)] else 0)
    assert a.bootstrap_interval(base, b, grp, reps=50) == [1.0, 1.0]


def test_bootstrap_interval_covers_d_is_ordered_and_repeats():
    labels = real_labels()
    grp = a.groups(labels, a.NEW_TASKS)
    base, b = episodes(a.NEW_TASKS, lambda t, i: 0), episodes(a.NEW_TASKS, lambda t, i: 8 if labels[(t, i)] else 0)
    lo, hi = a.bootstrap_interval(base, b, grp, reps=2000)
    assert 0.3 < lo < 0.5 < hi < 0.7
    assert a.bootstrap_interval(base, b, grp, reps=2000) == [lo, hi]


def test_bootstrap_keeps_seeds_paired():
    grp = {4: ([0], [1])}
    base = {(4, s, i): s % 2 for s in a.EVAL_SEEDS for i in (0, 1)}
    # Identical arms: every seed's difference is 0, so a paired resample cannot move D.
    assert a.bootstrap_interval(base, dict(base), grp, reps=50) == [0.0, 0.0]


def test_bootstrap_interval_takes_the_2_5_and_97_5_percentiles():
    grp = {4: ([0], [1])}
    base = {(4, s, i): 0 for s in a.EVAL_SEEDS for i in (0, 1)}
    b = {(4, s, i): int(i == 1 and s % 2 == 0) for s in a.EVAL_SEEDS for i in (0, 1)}
    rng = random.Random(a.BOOTSTRAP_SEED)
    diffs = [b[(4, s, 0)] - base[(4, s, 0)] for s in a.EVAL_SEEDS], [b[(4, s, 1)] - base[(4, s, 1)] for s in a.EVAL_SEEDS]
    values = []
    for _ in range(1000):
        sys_gain, low_gain = sum(rng.choices(diffs[0], k=16)) / 16, sum(rng.choices(diffs[1], k=16)) / 16
        values.append(low_gain - sys_gain)
    values.sort()
    assert a.bootstrap_interval(base, b, grp, reps=1000) == [values[25], values[974]]
