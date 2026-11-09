"""analyze_step12.py: state selection from step 10, harness, search counts, provenance, test 1 and refusals."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step12 as a  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/" + a.POLICY_REVISION
STEP10_IN = "/kaggle/input/x/step10"
OUT = "/kaggle/working/step12"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "seed", "init_state", "success", "steps"])
        w.writeheader()
        w.writerows(rows)


def record(path: Path, settings: dict, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"task_id": 0, "n_action_steps": 10, "max_steps": 520, "policy_path": POLICY}
    r = {"git_commit": "c" * 40, "git_dirty": False, "git_dirty_sfbc": False, "packages": {"torch": "2"},
         "settings": dict(base, **settings)}
    r.update(extra)
    with open(path, "a") as f:
        f.write(json.dumps(r) + "\n")


def eval_settings(t, policy, j):
    run = f"{STEP10_IN}/runs" if policy.startswith("N") else f"{OUT}/runs"
    return {"task_id": t, "stage": "eval", "eval_policies": "lora", "eval_init_states": a.EVAL_STATE_SPECS[t],
            "eval_seeds": a.EVAL_SEED_PARTS[t][j], "lora_dir": f"{run}/{a.adapter_of(t, policy)}/adapter"}


def make_run(root: Path, step10: Path, success=lambda t, p, seed, s: 0, harness_change=0, train=True):
    """A complete synthetic step-12 folder and the step-10 rows its harness compares with."""
    changed = 0
    for t in a.TASKS:
        s0 = a.HARNESS_STATE[t]
        old = [{"task_id": t, "seed": e, "init_state": s0, "success": 1, "steps": 300 + e} for e in a.HARNESS_SEEDS]
        write_csv(step10 / f"task_{t}/eval/N0/part0/per_episode_lora.csv", old)
        new = []
        for r in old:
            r = dict(r)
            if changed < harness_change:
                r["steps"] += 1
                changed += 1
            new.append(r)
        write_csv(root / f"task_{t}/harness/per_episode_lora.csv", new)
        record(root / f"task_{t}/harness/provenance.jsonl",
               {"task_id": t, "stage": "eval", "eval_policies": "lora", "eval_init_states": str(s0),
                "eval_seeds": "16-19", "lora_dir": f"{STEP10_IN}/runs/N{t}_0/adapter"})
        for s in a.SEARCHED[t]:
            rows = [{"task_id": t, "seed": 1000 + k, "init_state": s, "success": int(k % 10 == 0), "steps": 520}
                    for k in a.SEARCH_IDS]
            write_csv(root / f"task_{t}/search/per_episode_repeat_state{s}.csv", rows)
            record(root / f"task_{t}/search/provenance.jsonl",
                   {"task_id": t, "family": "repeat", "jitter_seeds": None, "noise_ids": "7000-7049", "init_state": s,
                    "lora_dir": f"{STEP10_IN}/runs/N{t}_0/adapter", "record_dir": f"{OUT}/task_{t}/search_data",
                    "record_successes_only": 1, "frame_stride": 2},
                   git_dirty_noise_reachability=False)
        for p in a.POLICIES:
            for j, part in enumerate(a.EVAL_SEED_PARTS[t]):
                lo, hi = map(int, part.split("-"))
                rows = [{"task_id": t, "seed": e, "init_state": s, "success": success(t, p, e, s), "steps": 9}
                        for e in range(lo, hi + 1) for s in a.eval_states(t)]
                write_csv(root / f"task_{t}/eval/{p}/part{j}/per_episode_lora.csv", rows)
                record(root / f"task_{t}/eval/{p}/part{j}/provenance.jsonl", eval_settings(t, p, j))
    for t in a.TASKS:
        old = ["repeat_n12_init08_seed1012.npz", "full_n25_j5002_init11_seed5002.npz"]
        hits = [f"repeat_n{k}_init{s:02d}_seed{1000 + k}.npz" for s in a.SEARCHED[t] for k in a.SEARCH_IDS if k % 10 == 0]
        for s in a.TRAIN_SEEDS:
            record(step10 / f"runs/N{t}_{s}/provenance.jsonl", {"task_id": t, "train_episodes": old})
            if train:
                names = old + hits
                record(root / f"runs/R{t}_{s}/provenance.jsonl",
                       dict(a.TRAIN_SETTINGS, task_id=t, stage="train", seed=s, data_dir=f"{OUT}/task_{t}/round2_data",
                            val_init_states="", train_episodes=names))


def test_selection_rule_picks_the_committed_states_from_step_10():
    report = json.loads((a.STEP10 / "step10_report.txt.json").read_text())
    searched, held = a.select_states(report)
    assert searched == {t: a.SEARCHED[t] for t in a.TASKS} and held == {t: a.HELD_OUT[t] for t in a.TASKS}


def _state(t, s, role, k, n=16):
    return {"task": t, "state": s, "role": role, "episodes": n, **{f"N{i}": [k, [0, 1]] for i in range(3)}}


def test_selection_threshold_is_inclusive_and_skips_easy_states():
    report = {"states": [_state(0, 2, "trained", 5, 50 / 3),          # pooled 15/50 = 0.30: searched
                         _state(0, 6, "trained", 6, 50 / 3),          # 18/50 = 0.36: not
                         _state(0, 1, "easy", 0),                     # easy: never
                         _state(6, 14, "held out", 0), _state(6, 20, "held out", 0), _state(6, 9, "held out", 0)]}
    searched, held = a.select_states(report)
    assert searched[0] == (2,) and held[0] == ()
    assert held[6] == (20,) and searched[6] == (9, 14)               # the highest-numbered low held-out state stays out


def test_full_run_reads_and_test_1_counts_pairs(tmp_path):
    def success(t, p, seed, s):        # R0 and R1 win 4 more searched episodes per task than N; R2 equal to N
        if s in a.SEARCHED[t] and p in ("R0", "R1") and seed < 52:
            return 1
        return 0
    make_run(tmp_path / "run", tmp_path / "s10", success)
    r = a.analyse(tmp_path / "run", tmp_path / "s10")
    per = r["tests"]["test 1"]["per_seed"]
    wins = sum(4 * len(a.SEARCHED[t]) for t in a.TASKS)
    assert per["0"]["first_only"] == wins and per["0"]["second_only"] == 0 and per["0"]["n"] == 192
    assert per["0"]["p"] == pytest.approx(0.5 ** wins) and per["2"]["p"] == 1.0
    assert r["tests"]["test 1"]["passes"] and r["harness"]["passes"]
    assert r["decision"].startswith("a second round")
    assert r["search"]["7/38"] == {"hits": 5, "episodes": 50, "ids": [7000, 7010, 7020, 7030, 7040]}
    assert r["training_data"]["R7_0"] == {"round 1": {8: 1, 11: 1}, "round 2": {38: 5}}   # id 7000 is round 2's first
    assert r["training_sets"]["R0_2"] == (2, 20) and r["training_sets"]["R7_1"] == (2, 5)
    assert r["harness"]["episodes"] == 12 and r["harness"]["same"] == 12
    assert r["harness"]["per_task"][7] == {"same": 4, "episodes": 4, "successes_now": 4, "successes_step10": 4}
    roles = {(x["task"], x["state"]): x["role"] for x in r["states"]}
    assert roles[(0, 2)] == "searched" and roles[(6, 20)] == "held out" and roles[(7, 33)] == "check"
    assert "search_hits" in next(x for x in r["states"] if x["role"] == "searched")
    assert "test 1 passes" in a.format_report(r)


def test_one_seed_of_three_is_not_enough(tmp_path):
    def success(t, p, seed, s):
        return int(s in a.SEARCHED[t] and p == "R0" and seed < 56)
    make_run(tmp_path / "run", tmp_path / "s10", success)
    r = a.analyse(tmp_path / "run", tmp_path / "s10")
    assert [v["passes"] for v in r["tests"]["test 1"]["per_seed"].values()] == [True, False, False]
    assert not r["tests"]["test 1"]["passes"] and r["decision"].startswith("not shown")


def test_harness_threshold(tmp_path):
    make_run(tmp_path / "a", tmp_path / "s10a", harness_change=1)
    assert a.harness(tmp_path / "a", tmp_path / "s10a")["passes"]            # 11 of 12
    make_run(tmp_path / "b", tmp_path / "s10b", harness_change=2)
    h = a.harness(tmp_path / "b", tmp_path / "s10b")
    assert h["same"] == 10 and not h["passes"]


def test_refusals(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    run = tmp_path / "run"
    p = run / "task_7/eval/R1/part0/per_episode_lora.csv"
    rows = list(csv.DictReader(open(p)))
    write_csv(p, rows + rows[:1])
    with pytest.raises(ValueError, match="twice"):
        a.load_eval(run, 7, "R1")
    write_csv(p, rows[1:])
    with pytest.raises(ValueError, match="finish the run"):
        a.load_eval(run, 7, "R1")
    s = run / "task_6/search/per_episode_repeat_state14.csv"
    write_csv(s, list(csv.DictReader(open(s)))[:-1])
    with pytest.raises(ValueError, match="finish the run"):
        a.search_counts(run)


def test_provenance_refuses_a_round_1_policy_evaluated_with_a_round_2_adapter(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    assert a.check_provenance(tmp_path / "run")["records"] > 0
    bad = eval_settings(6, "N1", 0)
    bad["lora_dir"] = f"{OUT}/runs/R6_1/adapter"
    record(tmp_path / "run/task_6/eval/N1/part0/provenance.jsonl", bad)
    with pytest.raises(ValueError, match="evaluation settings"):
        a.check_provenance(tmp_path / "run")


def test_provenance_refuses_search_without_the_explorer_or_on_an_unsearched_state(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    record(tmp_path / "run/task_7/search/provenance.jsonl",
           {"task_id": 7, "family": "repeat", "jitter_seeds": None, "noise_ids": "7000-7049", "init_state": 47,
            "lora_dir": f"{STEP10_IN}/runs/N7_0/adapter", "record_dir": f"{OUT}/task_7/search_data",
            "record_successes_only": 1, "frame_stride": 2},
           git_dirty_noise_reachability=False)
    with pytest.raises(ValueError, match="search settings"):
        a.check_provenance(tmp_path / "run")


def test_provenance_refuses_training_settings_and_dirty_trees(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10", train=False)
    record(tmp_path / "run/runs/R0_0/provenance.jsonl",
           dict(a.TRAIN_SETTINGS, task_id=0, stage="train", seed=0, data_dir=f"{OUT}/task_0/round2_data",
                val_init_states="", train_episodes=[], train_steps=1000))
    with pytest.raises(ValueError, match="training settings"):
        a.check_provenance(tmp_path / "run")
    make_run(tmp_path / "r2", tmp_path / "s10b", train=False)
    record(tmp_path / "r2/task_0/harness/provenance.jsonl",
           {"task_id": 0, "stage": "eval", "eval_policies": "lora", "eval_init_states": "8", "eval_seeds": "16-19",
            "lora_dir": f"{STEP10_IN}/runs/N0_0/adapter"}, git_dirty=True)
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(tmp_path / "r2")


def test_mcnemar_two_sided_known_values():
    assert a.mcnemar_two_sided(3, 0) == 0.25 and a.mcnemar_two_sided(0, 3) == 0.25
    assert a.mcnemar_two_sided(0, 0) == 1.0 and a.mcnemar_two_sided(1, 1) == 1.0
    assert a.mcnemar_two_sided(5, 0) == pytest.approx(2 / 32)


def test_episode_counts_match_the_protocol():
    assert sum(len(a.EVAL_SEEDS[t]) * len(a.SEARCHED[t]) for t in a.TASKS) == 192
    assert sum(len(a.EVAL_SEEDS[t]) * len(a.eval_states(t)) for t in a.TASKS) == 304
    assert a.EVAL_SEED_PARTS[0] == ["48-51", "52-55", "56-59", "60-63"] and len(a.EVAL_SEED_PARTS[6]) == 8
    assert not set(a.EVAL_SEEDS[6]) & set(range(16, 48))           # fresh: none of step 10's evaluation seeds


def test_episodes_won_only_by_round_1_count_against_round_2(tmp_path):
    def success(t, p, seed, s):        # task 7 state 38: R0 wins seeds 48-50, N0 wins seeds 51-52
        if (t, s) != (7, 38):
            return 0
        return int((p == "R0" and seed in (48, 49, 50)) or (p == "N0" and seed in (51, 52)))
    make_run(tmp_path / "run", tmp_path / "s10", success)
    v = a.analyse(tmp_path / "run", tmp_path / "s10")["tests"]["test 1"]["per_seed"]["0"]
    assert (v["first"], v["second"], v["first_only"], v["second_only"]) == (3, 2, 3, 2)
    assert v["p"] == pytest.approx(0.5) and v["p_two_sided"] == 1.0


def _search_record(**change):
    st = {"task_id": 6, "family": "repeat", "jitter_seeds": None, "noise_ids": "7000-7049", "init_state": 14,
          "lora_dir": f"{STEP10_IN}/runs/N6_0/adapter", "record_dir": f"{OUT}/task_6/search_data",
          "record_successes_only": 1, "frame_stride": 2}
    st.update(change)
    return st


def test_search_must_record_successes_only_at_stride_2(tmp_path):
    for change in ({"record_successes_only": 0}, {"frame_stride": 1}):
        make_run(tmp_path / str(change), tmp_path / f"s10{change}")
        record(tmp_path / str(change) / "task_6/search/provenance.jsonl", _search_record(**change),
               git_dirty_noise_reachability=False)
        with pytest.raises(ValueError, match="search settings"):
            a.check_provenance(tmp_path / str(change))


def test_a_record_without_any_dirty_flag_is_refused(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    f = tmp_path / "run/task_7/eval/R0/part0/provenance.jsonl"
    r = json.loads(f.read_text().splitlines()[0])
    del r["git_dirty_sfbc"]
    f.write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(tmp_path / "run")


def _rewrite_train(path: Path, names):
    r = json.loads(path.read_text().splitlines()[0])
    r["settings"]["train_episodes"] = names
    path.write_text(json.dumps(r) + "\n")


def test_round_2_training_set_must_be_round_1_plus_every_search_hit(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    search = a.search_counts(tmp_path / "run")
    assert a.check_training_data(tmp_path / "run", search, tmp_path / "s10")["R6_0"] == (2, 15)
    f = tmp_path / "run/runs/R6_1/provenance.jsonl"
    names = a.train_episodes(f)
    for bad in (names[:-1],                                            # a search success missing
                [n for n in names if "n12_" not in n],                 # a round-1 episode missing
                names + ["repeat_n7001_init20_seed8001.npz"],          # a held-out state's episode
                names + names[:1]):                                    # an episode twice
        _rewrite_train(f, bad)
        with pytest.raises(ValueError, match="R6_1"):
            a.check_training_data(tmp_path / "run", search, tmp_path / "s10")
    _rewrite_train(f, names)
    (tmp_path / "run/runs/R6_2/provenance.jsonl").unlink()
    with pytest.raises(ValueError, match="R6_2: no training run"):
        a.check_training_data(tmp_path / "run", search, tmp_path / "s10")
    got = a.check_training_data(tmp_path / "run", search, tmp_path / "s10", skip_missing=True)
    assert "R6_2" not in got and got["R6_1"] == (2, 15) and len(got) == 8
    _rewrite_train(f, names[:-1])                                      # present runs are still checked
    with pytest.raises(ValueError, match="R6_1"):
        a.check_training_data(tmp_path / "run", search, tmp_path / "s10", skip_missing=True)


def test_round_2_names_are_deep_noise_search_names():
    assert a.record_name("repeat", 7003, None, 2, 8003) == "repeat_n7003_init02_seed8003.npz"


def test_one_dirty_flag_among_clean_ones_is_refused(tmp_path):
    make_run(tmp_path / "run", tmp_path / "s10")
    record(tmp_path / "run/task_6/search/provenance.jsonl", _search_record(), git_dirty_noise_reachability=True)
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(tmp_path / "run")
