"""Step 9 analysis: exact permutation test, decision rule, refusals, and the library against the committed results."""
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_step8 as a8  # noqa: E402
import analyze_step9 as a9  # noqa: E402
from check_scene import SCENE_NOISE_ID  # noqa: E402
from test_analyze_step8 import episodes, provenance, scene, write  # noqa: E402

WEEK6 = ROOT / "week6"


def brute_p(per_task):
    """Enumerates every within-task relabelling; tiny inputs only."""
    observed = sum(sum(lib) for lib, _ in per_task.values())
    splits = []
    for lib, ran in per_task.values():
        pool = list(lib) + list(ran)
        splits.append([sum(pool[i] for i in c) for c in itertools.combinations(range(len(pool)), len(lib))])
    totals = [sum(x) for x in itertools.product(*splits)]
    return sum(t >= observed for t in totals) / len(totals)


# ---- the exact test ----------------------------------------------------------------------------------------------

def test_subset_sum_counts_known_answer():
    assert a9.subset_sum_counts([0, 1, 2], 2) == {1: 1, 2: 1, 3: 1}
    assert a9.subset_sum_counts([1, 1, 0, 0], 2) == {2: 1, 1: 4, 0: 1}
    assert a9.subset_sum_counts([2, 0], 0) == {0: 1}


def test_permutation_p_known_answer():
    assert a9.permutation_p({0: ([1], [0, 0, 0])}) == pytest.approx(0.25)
    assert a9.permutation_p({0: ([0], [0, 0, 0])}) == 1.0
    # two tasks, each 1 of 2 noises in the library and one hit in the task: P(both hits land in the library) = 1/4
    assert a9.permutation_p({0: ([1], [0]), 4: ([1], [0])}) == pytest.approx(0.25)
    assert a9.permutation_p({0: ([1], [0]), 4: ([0], [1])}) == pytest.approx(0.75)


def test_permutation_p_matches_enumeration():
    rng = random.Random(1)
    for _ in range(200):
        per_task = {}
        for t in rng.sample([0, 4, 6, 7], rng.randint(1, 3)):
            lib = [rng.choice([0, 0, 1, 2]) for _ in range(rng.randint(1, 3))]
            ran = [rng.choice([0, 0, 0, 1, 2]) for _ in range(rng.randint(1, 4))]
            per_task[t] = (lib, ran)
        assert a9.permutation_p(per_task) == pytest.approx(brute_p(per_task), abs=1e-12)


def test_one_state_one_task_is_fisher():
    """With 0/1 outcomes on one state, relabelling noises is the hypergeometric test on the two counts."""
    for a, L, b, R in ((3, 20, 1, 50), (0, 6, 0, 50), (5, 25, 2, 50), (6, 6, 50, 50)):
        lib = [1] * a + [0] * (L - a)
        ran = [1] * b + [0] * (R - b)
        assert a9.permutation_p({7: (lib, ran)}) == pytest.approx(a9.fisher_unequal(a, L, b, R), rel=1e-12)


def test_noise_hitting_both_states_counts_once_as_a_unit():
    """A library noise that hits both states is one lucky unit, not two independent hits: the p is larger than a
    per-episode test that treats the 2 x 3 library episodes as independent."""
    per_task = {0: ([2, 0, 0], [0] * 50)}
    p = a9.permutation_p(per_task)
    assert p == pytest.approx(3 / 53)                       # the one noise holding both hits lands in the library
    assert p > a9.fisher_unequal(2, 6, 0, 100)


def test_fisher_unequal_known_answer():
    # 2 of 3 against 0 of 3: P(the 2 successes both in the first group) = C(3,2)/C(6,2)
    assert a9.fisher_unequal(2, 3, 0, 3) == pytest.approx(3 / 15)
    assert a9.fisher_unequal(0, 5, 3, 5) == 1.0


@pytest.mark.parametrize("p, want", [(0.0499999, "carry over"), (0.05, "not shown"), (0.0500001, "not shown")])
def test_reading_threshold(p, want):
    assert want in a9.reading(True, 8, p)


def test_reading_not_read():
    assert a9.reading(False, 8, 0.001).startswith("not read: a harness")
    assert a9.reading(True, 0, None).startswith("not read: no held-out")


# ---- the library and the design --------------------------------------------------------------------------------

def test_library_sizes_and_disjointness():
    assert {t: len(v) for t, v in a9.LIBRARY.items()} == {0: 20, 4: 6, 6: 20, 7: 25}
    for t, v in a9.LIBRARY.items():
        assert len(set(v)) == len(v)
        assert not set(v) & set(a9.RANDOM_IDS)
        assert not {SCENE_NOISE_ID} & set(v)
    assert len(a9.RANDOM_IDS) == 50 and min(a9.RANDOM_IDS) == 6000
    assert set(a9.HELD_OUT) == set(a9.TASKS)


def hit_ids(path):
    return {int(r["seed"]) - 1000 for r in csv.DictReader(open(path)) if int(r["success"])}


def test_library_is_every_repeat_hit_of_the_committed_results():
    lib = {0: set(), 4: set(), 6: set(), 7: set()}
    for s in (2, 5, 6, 7, 8):
        lib[0] |= hit_ids(WEEK6 / f"noise_reachability_step5a_colab/run/per_episode_repeat_state{s}.csv")
    for t, s in ((4, 2), (6, 8)):
        lib[t] |= hit_ids(WEEK6 / f"noise_reachability_step7_kaggle/run/task_{t}/per_episode_repeat_state{s}.csv")
    for t, s in ((0, 11), (4, 11), (6, 8), (6, 12), (7, 33)):
        lib[t] |= hit_ids(WEEK6 / f"noise_reachability_step8_kaggle/run/task_{t}/test/per_episode_repeat_state{s}.csv")
    assert {t: tuple(sorted(v)) for t, v in lib.items()} == a9.LIBRARY


def test_held_out_are_step8_eligible_states_it_did_not_test():
    r = json.load(open(WEEK6 / "noise_reachability_step8_kaggle/run/step8_report.txt.json"))
    eligible = {tuple(k) for k in r["screen"]["eligible"]}
    tested = {(v["task"], v["state"]) for v in r["states"]}
    assert {(t, s) for t, v in a9.HELD_OUT.items() for s in v} == eligible - tested


# ---- end to end on a synthetic run -----------------------------------------------------------------------------

def make_run(root, harness=None, scenes=None, lib_hits=None, ran_hits=None, drop=None):
    """A complete step-9 folder. lib_hits / ran_hits: {(task, state): noise ids that succeed}."""
    harness, scenes = harness or {}, scenes or {}
    lib_hits, ran_hits = lib_hits or {}, ran_hits or {}
    for t in a9.TASKS:
        folder = root / f"task_{t}"
        provenance(folder, t)
        for seed in range(3):
            n = harness.get(t, a8.HARNESS_REFERENCE[t])
            rows = [{"task_id": t, "seed": seed, "init_state": i, "success": int(seed * 10 + i < n), "steps": 520}
                    for i in range(10)]
            write(folder / "harness" / f"seed{seed}" / "per_episode_base.csv", rows)
        for s in a9.HELD_OUT[t]:
            scene(root, t, s, scenes.get((t, s), True))
            for arm, hits in (("library", lib_hits), ("random", ran_hits)):
                ids = list(a9.arm_ids(t, arm))
                if drop == (t, s, arm):
                    ids = ids[:-1]
                write(folder / "test" / arm / f"per_episode_repeat_state{s}.csv", episodes(t, s, ids, hits.get((t, s), ())))
                provenance(folder / "test" / arm, t)
    return root


def test_analyse_counts_and_jobs(tmp_path):
    lib = {(7, 38): a9.LIBRARY[7][:4], (7, 47): a9.LIBRARY[7][:2], (0, 14): a9.LIBRARY[0][:1]}
    ran = {(7, 38): a9.RANDOM_IDS[:1]}
    root = make_run(tmp_path, lib_hits=lib, ran_hits=ran)
    r = a9.analyse(root)
    assert r["pooled"]["library"] == 7 and r["pooled"]["random"] == 1
    assert r["pooled"]["library_episodes"] == 2 * 71 and r["pooled"]["random_episodes"] == 400
    assert r["tasks"][7]["library_ids_on_every_state"] == 2
    assert r["tasks"][7]["library_hits"][0] == [a9.LIBRARY[7][0], "step 8, 7/33", [38, 47]]
    assert r["p"] < 0.05 and "carry over" in r["decision"]
    assert len(a9.jobs(root)) == 16
    text = a9.format_report(r)
    assert "decision: hit noises carry over" in text
    assert "pooled: library 7/142 (2.0-9.9%), random 1/400 (0.0-1.4%)" in text


def test_no_effect_does_not_pass(tmp_path):
    lib = {(7, 38): a9.LIBRARY[7][:1]}
    ran = {(7, 38): a9.RANDOM_IDS[:2], (7, 47): a9.RANDOM_IDS[2:4]}
    r = a9.analyse(make_run(tmp_path, lib_hits=lib, ran_hits=ran))
    assert r["p"] > 0.05 and r["decision"].startswith("not shown")


def test_scene_failure_drops_only_that_state(tmp_path):
    root = make_run(tmp_path, scenes={(6, 14): False})
    r = a9.analyse(root)
    assert r["scene_failed"] == [[6, 14]]
    assert r["tasks"][6]["states"] == [20]
    assert r["tasks"][6]["library_episodes"] == 20
    assert (6, 14) not in {(t, s) for t, s, _, _ in a9.jobs(root)}


def test_all_scenes_fail_is_not_read(tmp_path):
    scenes = {(t, s): False for t, v in a9.HELD_OUT.items() for s in v}
    r = a9.analyse(make_run(tmp_path, scenes=scenes))
    assert r["decision"].startswith("not read: no held-out") and r["p"] is None


def test_harness_failure_is_not_read(tmp_path):
    r = a9.analyse(make_run(tmp_path, harness={6: 13}))
    assert r["decision"].startswith("not read: a harness")
    assert a9.gate(tmp_path).startswith("harness failed: task 6")


def test_harness_passes_at_the_edge(tmp_path):
    make_run(tmp_path, harness={6: 14, 0: 13})
    assert a9.gate(tmp_path) is None


def test_refuses_missing_episode(tmp_path):
    with pytest.raises(ValueError, match="finish the run"):
        a9.analyse(make_run(tmp_path, drop=(4, 26, "library")))


def test_refuses_missing_scene(tmp_path):
    root = make_run(tmp_path)
    (root / "task_4" / "scene" / "scene_state18.json").unlink()
    with pytest.raises(ValueError, match="scene check of 4/18"):
        a9.jobs(root)


def test_refuses_dirty_or_wrong_setting(tmp_path):
    root = make_run(tmp_path)
    provenance(root / "task_0" / "test" / "random", 0, settings={"n_action_steps": 1})
    with pytest.raises(ValueError, match="settings"):
        a9.analyse(root)


def test_cli_jobs_and_out(tmp_path, capsys):
    root = make_run(tmp_path / "run")
    assert a9.main(["--root", str(root), "--jobs"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "0 14 library " + ",".join(map(str, a9.LIBRARY[0]))
    assert lines[1] == "0 14 random " + ",".join(map(str, a9.RANDOM_IDS))
    out = tmp_path / "report.txt"
    assert a9.main(["--root", str(root), "--out", str(out)]) == 0
    assert out.read_text().endswith("not shown: no carry-over found; the test had about 0.25-0.55 power for the hypothesised 2x gain\n")
    assert json.loads(Path(str(out) + ".json").read_text())["p"] == 1.0
    assert a9.main(["--root", str(root), "--gate", "harness"]) == 0


def test_one_state_read_counts_no_id_as_on_every_state(tmp_path):
    root = make_run(tmp_path, scenes={(7, 47): False}, lib_hits={(7, 38): a9.LIBRARY[7][:3]})
    assert a9.analyse(root)["tasks"][7]["library_ids_on_every_state"] == 0


def test_report_lines(tmp_path):
    r = a9.analyse(make_run(tmp_path, lib_hits={(4, 18): a9.LIBRARY[4][:1]}))
    text = a9.format_report(r)
    assert "task 0 state 14: library 0/20 (0.0-16.8%), random 0/50 (0.0-7.1%)" in text
    assert "task 4 state 18: library 1/6 (0.4-64.1%)" in text
    assert "held-out states that failed the scene check: []" in text
    assert "does not pass at 0.05" in text
    assert "pooled: library 1/142 (0.0-3.9%), random 0/400 (0.0-0.9%)" in text


def test_report_when_harness_failed(tmp_path):
    r = a9.analyse(make_run(tmp_path / "a", harness={0: 0}))
    text = a9.format_report(r)
    assert "scene check" not in text and "pooled" not in text
    assert text.endswith("decision: not read: a harness failed\n")
    assert a9.main(["--root", str(tmp_path / "a"), "--gate", "harness"]) == 1


def test_sources_name_every_state_an_id_was_found_on():
    assert a9.source_of(0, 30) == "step 5a, 0/5; step 5a, 0/6; step 5a, 0/7"
    assert a9.source_of(0, 15) == "step 5a, 0/7; step 5a, 0/8"
    assert a9.source_of(6, 30) == "step 7, 6/8"


def test_step5a_sources_match_its_files():
    for src, ids in a9.LIBRARY_SOURCES[0].items():
        if src.startswith("step 5a"):
            s = src.split("/")[1]
            assert hit_ids(WEEK6 / f"noise_reachability_step5a_colab/run/per_episode_repeat_state{s}.csv") == set(ids)
