"""Step 15 analysis: known answers, the edges of both rules, the harness threshold, refusals and readings."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step15 as a  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d"
N_LORA = "/kaggle/input/noise-reachability-step-10/step10/runs/N7_{s}/adapter"


def write_csv(path, task, rows):
    """rows: {(seed, state): (success, steps)}; appended as the scripts do, header once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in sorted(rows.items()):
            w.writerow([task, seed, state, succ, steps])


def record(settings, extra=None, dirty_key="git_dirty_noise_reachability"):
    st = dict({"n_action_steps": 10, "max_steps": 520, "policy_path": POLICY}, **settings)
    return dict({"git_commit": "abc", "git_dirty": False, dirty_key: False, "packages": {"torch": "2.11.0"},
                 "gpus": ["Tesla T4", "Tesla T4"],
                 "settings": st}, **(extra or {}))


def add_prov(folder, rec):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def search(task, state, ids, lora=None):
    return {"task_id": task, "family": "repeat", "init_state": state, "noise_ids": ids, "jitter_seeds": None,
            "record_dir": None, "lora_dir": lora}


def rows_of(task, state, ids, success, steps=520):
    return {(1000 + k, state): (int(success(k)), 300 if success(k) else steps) for k in ids}


def make_run(root, a_success=None, replay_success=None, b_success=None, harness_change=None, scene=None):
    """A complete step-15 run. a_success(state, id), replay_success(state, id), b_success(policy, id) -> bool."""
    a_success = a_success or (lambda s, k: False)
    replay_success = replay_success or a_success
    b_success = b_success or (lambda p, k: False)
    scene = scene or {}
    for t, states in a.SCENE_STATES.items():
        for s in states:
            d = root / f"task_{t}" / "scene"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"scene_state{s}.json").write_text(json.dumps({"task_id": t, "init_state": s,
                                                                 "same": scene.get((t, s), True)}))
            add_prov(d, record({"task_id": t, "init_state": s, "output_dir": str(d)}))
    old = a.committed_harness()
    for policy, rows in old.items():
        rows = dict(rows)
        rows.update((harness_change or {}).get(policy, {}))
        if policy == "base":
            d = root / "harness" / "base"
            write_csv(d / f"per_episode_repeat_state{a.STATE_B}.csv", 7, rows)
            add_prov(d, record(search(7, a.STATE_B, a.ids_spec(a.HARNESS_BASE_IDS))))
        else:
            d = root / "harness" / "N0"
            write_csv(d / "per_episode_lora.csv", 7, rows)
            add_prov(d, record({"task_id": 7, "stage": "eval", "eval_policies": "lora", "eval_init_states": "33",
                                "eval_seeds": "16-19", "lora_dir": N_LORA.format(s=0)}, dirty_key="git_dirty_sfbc"))
    for s in a.STATES_A:
        if not scene.get((6, s), True):
            continue
        for j, part in enumerate(a.A_PARTS):
            d = root / "task_6" / "search" / f"part{j}"
            write_csv(d / f"per_episode_repeat_state{s}.csv", 6, rows_of(6, s, a.parse_ids(part), lambda k: a_success(s, k)))
            add_prov(d, record(search(6, s, part)))
        wins = [k for k in a.A_IDS if a_success(s, k)]
        if wins:
            d = root / "task_6" / "replay"
            write_csv(d / f"per_episode_repeat_state{s}.csv", 6, rows_of(6, s, wins, lambda k: replay_success(s, k)))
            add_prov(d, record(search(6, s, a.ids_spec(wins))))
    if scene.get((7, a.STATE_B), True):
        for p in a.B_POLICIES:
            d = root / "task_7" / "search" / p
            write_csv(d / f"per_episode_repeat_state{a.STATE_B}.csv", 7,
                      rows_of(7, a.STATE_B, a.B_IDS, lambda k, p=p: b_success(p, k)))
            add_prov(d, record(search(7, a.STATE_B, a.ids_spec(a.B_IDS), None if p == "base" else N_LORA.format(s=p[1]))))
    return root


def by_offset(spec):
    """b_success from {policy: set of id offsets (id - 9000) that succeed}."""
    return lambda p, k: (k - 9000) in spec.get(p, set())


# ---- constants and helpers ---------------------------------------------------------------------------------------

def test_ids_are_fresh_and_the_protocols():
    committed = set(range(0, 70)) | set(range(100, 400)) | set(range(1000, 1600)) | set(range(6000, 6050)) | set(range(7000, 7050))
    assert not committed & set(a.A_IDS) and len(a.A_IDS) == 400 and len(a.B_IDS) == 200
    assert set(a.B_IDS) <= set(a.A_IDS) and a.parse_ids(",".join(a.A_PARTS)) == list(a.A_IDS)
    assert a.STATES_A == (14, 20) and (a.TASK_B, a.STATE_B) == (7, 38) and a.HARNESS_MIN_SAME == 13


def test_ids_spec_round_trips():
    assert a.ids_spec(a.HARNESS_BASE_IDS) == "6000-6002,6017,6019,6021,6023,6025,6030,6040"
    assert a.ids_spec(range(9000, 9200)) == "9000-9199" and a.ids_spec([5]) == "5" and a.ids_spec([3, 1, 2]) == "1-3"
    for ids in ([9001, 9003, 9004], list(a.HARNESS_BASE_IDS), [7]):
        assert a.parse_ids(a.ids_spec(ids)) == sorted(ids)


def test_harness_reads_step9s_seven_successes_and_step10s_rows():
    h = a.committed_harness()
    assert sum(v[0] for v in h["base"].values()) == 7 and len(h["base"]) == 10
    assert sorted(h["N0"]) == [(s, 33) for s in range(16, 20)]


# ---- known answers -----------------------------------------------------------------------------------------------

def test_known_answers_for_both_tests(tmp_path):
    # base wins on offsets 0-11, N0 on 10-11 only (base-only 10 -> p = 2^-10), N1 never, N2 on 0-11 (equal)
    spec = {"base": set(range(12)), "N0": {10, 11}, "N2": set(range(12))}
    r = a.analyse(make_run(tmp_path, a_success=lambda s, k: s == 14 and k in (9005, 9300), b_success=by_offset(spec)))
    t14, t20 = r["test A"]["14"], r["test A"]["20"]
    assert t14["successes"] == 2 and t14["replayed_ids"] == [9005, 9300] and t14["passes"]
    assert t20["successes"] == 0 and not t20["passes"] and t20["upper_bound_if_none"] == pytest.approx(0.0074614, abs=1e-6)
    b = r["test B"]["per_seed"]
    assert (b["0"]["first_only"], b["0"]["second_only"], b["0"]["p"]) == (10, 0, 2 ** -10)
    assert (b["1"]["first_only"], b["1"]["p"]) == (12, 2 ** -12)
    assert (b["2"]["first_only"], b["2"]["second_only"], b["2"]["p"]) == (0, 0, 1.0)
    assert r["test B"]["passes"] and r["test B"]["counts"]["base"][0] == 12
    assert "rare" in r["decision"]["6/14"] and "0.75%" in r["decision"]["6/20"]
    assert "both task-6 states" not in r["decision"] and "removed" in r["decision"]["7/38"]


def test_a_success_that_does_not_replay_does_not_count(tmp_path):
    r = a.analyse(make_run(tmp_path, a_success=lambda s, k: k == 9100, replay_success=lambda s, k: s == 20))
    assert not r["test A"]["14"]["passes"] and r["test A"]["14"]["successes"] == 1
    assert r["test A"]["14"]["replayed_ids"] == [] and r["test A"]["20"]["passes"]


def test_both_states_solved_gives_the_combined_reading(tmp_path):
    r = a.analyse(make_run(tmp_path, a_success=lambda s, k: k == 9399))
    assert r["decision"]["both task-6 states"].startswith("every resistant state")


def test_b_needs_two_of_three_seeds_and_p_strictly_below(tmp_path):
    # 5 base-only pairs give p = 1/32 < 0.05; 4 give 1/16 > 0.05
    five, four = set(range(5)), set(range(4))
    r = a.analyse(make_run(tmp_path, b_success=lambda p, k: (k - 9000) in (five if p == "base" else set())
                           if p in ("base", "N0") else (k - 9000) in (four if p == "base" else set())))
    # base succeeds on offsets 0-4 for every comparison (base is shared): N0, N1, N2 all lose 5 -> 3 of 3 pass
    assert r["test B"]["passes"]
    for spec, passes in (({"base": five, "N1": five, "N2": five}, False),      # only seed 0 passes
                         ({"base": five, "N2": five}, True),                     # seeds 0 and 1
                         ({"base": five, "N0": {0}, "N1": {0}, "N2": {0}}, False)):  # 4 base-only each: 1/16
        r = a.analyse(make_run(tmp_path / str(len(list(tmp_path.iterdir()))), b_success=by_offset(spec)))
        assert r["test B"]["passes"] is passes, spec


def test_b_is_one_sided_a_better_adapter_does_not_pass(tmp_path):
    spec = {p: set(range(20)) for p in ("N0", "N1", "N2")}
    r = a.analyse(make_run(tmp_path, b_success=by_offset(spec)))
    assert not r["test B"]["passes"] and all(v["p"] == 1.0 for v in r["test B"]["per_seed"].values())
    assert all(v["p_two_sided"] < 1e-5 for v in r["test B"]["per_seed"].values())


# ---- harness -----------------------------------------------------------------------------------------------------

def test_harness_threshold_13_of_14(tmp_path):
    one = {"base": {(7017, 38): (1, 429)}}
    two = {"base": {(7017, 38): (1, 429)}, "N0": {(16, 33): (0, 520)}}
    assert a.gate(make_run(tmp_path / "one", harness_change=one)) is None
    assert "12 of 14" in a.gate(make_run(tmp_path / "two", harness_change=two))
    r = a.analyse(tmp_path / "two")
    assert r["decision"] == "not read: the harness failed" and "test A" not in r


def test_harness_refuses_missing_episodes(tmp_path):
    root = make_run(tmp_path)
    p = root / "harness" / "base" / "per_episode_repeat_state38.csv"
    p.write_text("".join(p.read_text().splitlines(True)[:-1]))
    with pytest.raises(ValueError, match="finish the run"):
        a.gate(root)


# ---- scenes ------------------------------------------------------------------------------------------------------

def test_a_failed_scene_drops_the_state(tmp_path):
    r = a.analyse(make_run(tmp_path, scene={(6, 20): False, (7, 38): False}))
    assert r["test A"]["20"] is None and r["test B"] is None and r["scene_failed"] == [[6, 20], [7, 38]]
    assert "scene" in r["decision"]["6/20"] and "scene" in r["decision"]["7/38"]
    assert a.replay_jobs(tmp_path) == []


def test_a_missing_scene_check_is_refused(tmp_path):
    root = make_run(tmp_path)
    (root / "task_6" / "scene" / "scene_state14.json").unlink()
    with pytest.raises(ValueError, match="scene check of 6/14 is missing"):
        a.analyse(root)


# ---- replays -----------------------------------------------------------------------------------------------------

def test_replay_jobs_list_the_successes(tmp_path):
    root = make_run(tmp_path, a_success=lambda s, k: (s, k) in {(14, 9001), (14, 9002), (14, 9250), (20, 9399)})
    assert a.replay_jobs(root) == [(6, 14, "9001-9002,9250"), (6, 20, "9399")]
    assert a.replay_jobs(make_run(tmp_path / "none")) == []


def test_missing_replay_is_refused(tmp_path):
    root = make_run(tmp_path, a_success=lambda s, k: s == 14 and k in (9001, 9002))
    p = root / "task_6" / "replay" / "per_episode_repeat_state14.csv"
    p.write_text("".join(p.read_text().splitlines(True)[:-1]))
    with pytest.raises(ValueError, match="replay of 6/14"):
        a.analyse(root)


# ---- refusals ----------------------------------------------------------------------------------------------------

def test_incomplete_or_extra_episodes_are_refused(tmp_path):
    root = make_run(tmp_path)
    p = root / "task_6" / "search" / "part1" / "per_episode_repeat_state20.csv"
    p.write_text("".join(p.read_text().splitlines(True)[:-1]))
    with pytest.raises(ValueError, match="finish the run"):
        a.analyse(root)
    root = make_run(tmp_path / "extra")
    write_csv(root / "task_7" / "search" / "N1" / "per_episode_repeat_state38.csv", 7, {(1000 + 9300, 38): (0, 520)})
    with pytest.raises(ValueError, match="1 others"):
        a.analyse(root)


def test_a_repeated_episode_is_refused(tmp_path):
    root = make_run(tmp_path)
    write_csv(root / "task_7" / "search" / "base" / "per_episode_repeat_state38.csv", 7, {(10000, 38): (1, 300)})
    with pytest.raises(ValueError, match="twice"):
        a.analyse(root)


@pytest.mark.parametrize("folder, change, match", [
    ("task_6/search/part0", {"noise_ids": "9000-9399"}, "ids"),
    ("task_6/search/part1", {"family": "sampled"}, "not a plain repeat search"),
    ("task_6/search/part0", {"lora_dir": "/x/step10/runs/N6_0/adapter"}, "an adapter on the base"),
    ("task_7/search/N1", {"lora_dir": N_LORA.format(s=2)}, "adapter"),
    ("task_7/search/base", {"init_state": 47}, "state"),
    ("task_7/search/N0", {"noise_ids": "9000-9198"}, "ids"),
    ("task_6/replay", {"noise_ids": "8999"}, "replay ids"),
    ("harness/base", {"noise_ids": "6000-6009"}, "harness ids"),
    ("harness/N0", {"eval_seeds": "16-20"}, "harness settings"),
    ("task_6/search/part0", {"n_action_steps": 1}, "n_action_steps"),
    ("task_6/search/part0", {"max_steps": 300}, "n_action_steps or max_steps"),
    ("task_6/scene", {"task_id": 7}, "scene check"),
])
def test_wrong_settings_are_refused(tmp_path, folder, change, match):
    root = make_run(tmp_path, a_success=lambda s, k: k == 9000)
    f = root / folder / "provenance.jsonl"
    lines = f.read_text().splitlines()
    r = json.loads(lines[0])
    r["settings"].update(change)
    f.write_text("\n".join([json.dumps(r)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match=match):
        a.analyse(root)


@pytest.mark.parametrize("change, match", [
    ({"git_dirty": True}, "dirty"), ({"git_dirty_noise_reachability": True}, "dirty"),
    ({"git_commit": "def"}, "commits"), ({"packages": {"torch": "2.10.0"}}, "packages"),
    ({"gpus": ["NVIDIA A100-SXM4-40GB"]}, "GPUs"), ({"gpus": []}, "GPUs"), ({"gpus": ["Tesla T4", "NVIDIA A100"]}, "GPUs"),
])
def test_dirty_trees_mixed_commits_and_packages_are_refused(tmp_path, change, match):
    root = make_run(tmp_path)
    add_prov(root / "task_7" / "search" / "N2",
             record(search(7, 38, "9000-9199", N_LORA.format(s=2)), extra=change))
    with pytest.raises(ValueError, match=match):
        a.check_provenance(root)


def test_an_unexpected_folder_is_refused(tmp_path):
    root = make_run(tmp_path)
    add_prov(root / "task_7" / "search" / "R0", record(search(7, 38, "9000-9199")))
    with pytest.raises(ValueError, match="unexpected folder"):
        a.check_provenance(root)


def test_report_and_main(tmp_path, capsys):
    root = make_run(tmp_path, a_success=lambda s, k: s == 20 and k == 9123, b_success=by_offset({"base": {1, 2}}))
    out = tmp_path / "report.txt"
    assert a.main(["--root", str(root), "--out", str(out)]) == 0
    text = out.read_text()
    assert "test A, 6/20: base 1 of 400" in text and "1 replayed" in text and "test A, 6/14: base 0 of 400" in text
    assert "test B, base against N7_0: 2 against 0" in text and "test B does not pass" in text
    assert json.loads(Path(str(out) + ".json").read_text())["test A"]["20"]["passes"] is True
    assert a.main(["--root", str(root), "--gate", "harness"]) == 0
    assert a.main(["--root", str(root), "--replay_jobs"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "6 20 9123"


def test_one_dirty_flag_among_clean_ones_is_refused(tmp_path):
    root = make_run(tmp_path)
    add_prov(root / "harness" / "N0", record({"task_id": 7, "stage": "eval", "eval_policies": "lora",
                                              "eval_init_states": "33", "eval_seeds": "16-19",
                                              "lora_dir": N_LORA.format(s=0)},
                                             extra={"git_dirty_noise_reachability": False, "git_dirty_sfbc": True}))
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(root)


def test_the_committed_step9_file_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setitem(a.STEP9_SHA256, a.STEP9_FILE, "0" * 64)
    with pytest.raises(ValueError, match="not the committed file"):
        a.committed_harness()


def test_harness_lists_the_episodes_that_differ(tmp_path):
    r = a.harness(make_run(tmp_path, harness_change={"N0": {(17, 33): (1, 395)}}))
    assert r["per_policy"]["N0"]["differ"] == ["(17, 33)"] and r["per_policy"]["base"]["differ"] == []
    assert r["same"] == 13 and r["passes"]


def test_main_gate_exit_code_and_message(tmp_path, capsys):
    root = make_run(tmp_path, harness_change={"base": {(7017, 38): (0, 520), (7019, 38): (0, 520)}})
    assert a.main(["--root", str(root), "--gate", "harness"]) == 1
    assert capsys.readouterr().out.startswith("gate harness FAILED: harness: 12 of 14")


def test_steps_of_the_successes_are_reported(tmp_path):
    root = make_run(tmp_path, a_success=lambda s, k: s == 14 and k == 9007, b_success=by_offset({"N1": {3, 4}}))
    r = a.analyse(root)
    assert r["test A"]["14"]["success_steps"] == {"9007": [300, 300]} and r["test A"]["20"]["success_steps"] == {}
    assert r["test A"]["14"]["found_interval"][0] > 0 and r["test B"]["success_steps"]["N1"] == [300, 300]
    assert r["test B"]["success_steps"]["base"] == []
    assert "steps of the successes (search, replay): {'9007': [300, 300]}" in a.format_report(r)
