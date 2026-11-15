"""Step 14 analysis: known answers, the 2-of-3 rule and its direction, the harness threshold, refusals, readings."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step14 as a  # noqa: E402
from analyze_step12 import train_episodes  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in sorted(rows.items()):
            w.writerow([0, seed, state, succ, steps])


def record(settings, extra=None):
    st = dict({"task_id": 0, "n_action_steps": 10, "max_steps": 520, "policy_path": POLICY}, **settings)
    return dict({"git_commit": "abc", "git_dirty": False, "git_dirty_sfbc": False, "packages": {"torch": "2.11.0"},
                 "settings": st}, **(extra or {}))


def write_prov(folder, rec):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.jsonl").write_text(json.dumps(rec) + "\n")


def lora(policy):
    return f"/kaggle/input/x/{a.adapter_path(policy)}" if policy[0] != "L" else f"/kaggle/working/{a.adapter_path(policy)}"


def make_run(root, success=None, harness_change=None, train_lists=None):
    """A complete step-14 run. success(which, policy, seed, state) -> 0/1 (default 0)."""
    success = success or (lambda w, p, s, i: 0)
    runs = {"N0": a.STEP10, "R0": a.STEP12}
    for p, (seeds, _, folder) in a.HARNESS.items():
        rows = a.committed_rows(runs[p], folder, seeds)
        for k, v in (harness_change or {}).get(p, {}).items():
            rows[k] = v
        write_csv(root / "harness" / p / "per_episode_lora.csv", rows)
        write_prov(root / "harness" / p, record({"stage": "eval", "eval_policies": "lora", "eval_init_states": "8",
                                                  "eval_seeds": seeds, "lora_dir": lora(folder)}))
    for s in a.TRAIN_SEEDS:
        eps = (train_lists or {}).get(s) or train_episodes(a.STEP12 / "runs" / f"R0_{s}" / "provenance.jsonl")
        st = dict(a.TRAIN_SETTINGS, train_steps=a.L_STEPS, stage="train", seed=s, data_dir="/tmp/step14/task_0/round2_data",
                  val_init_states="", train_episodes=eps, train_episode_sha256={e: "0" * 64 for e in eps})
        write_prov(root / "runs" / f"L0_{s}", record(st))
    for w, (states, _, parts, policies) in a.SETS.items():
        for p in policies:
            for j, part in enumerate(parts):
                rows = {(s, i): (success(w, p, s, i), 300) for s in a.seeds_of(part) for i in states}
                d = root / w / p / f"part{j}"
                write_csv(d / "per_episode_lora.csv", rows)
                write_prov(d, record({"stage": "eval", "eval_policies": "lora", "eval_init_states": a.spec(states),
                                      "eval_seeds": part, "lora_dir": lora(p)}))
    return root


def by_index(spec):
    """success from {(which, policy letter): set of seed offsets that succeed}, same for every training seed unless the
    key carries the seed."""
    def f(w, p, s, i):
        for key in ((w, p), (w, p[0])):
            if key in spec:
                return int((s - 80) in spec[key])
        return 0
    return f


def test_known_mcnemar_values_and_direction(tmp_path):
    # N succeeds on the first 12 check episodes, R on episodes 10-11 only for every seed: N-only 10, R-only 0
    spec = {("eval_check", "N"): set(range(12)), ("eval_check", "R"): {10, 11}}
    r = a.analyse(make_run(tmp_path, by_index(spec)))
    v = r["tests"]["test A"]["per_seed"]["0"]
    assert (v["n"], v["first"], v["second"], v["first_only"], v["second_only"]) == (96, 12, 2, 10, 0)
    assert v["p"] == pytest.approx(1 / 1024) and r["tests"]["test A"]["passes"]
    assert not r["tests"]["test B"]["passes"]            # L at 0: never better than R
    assert r["decision"].startswith("round 2's loss on 0/8 is real; longer")


def test_one_sided_the_wrong_way_does_not_pass(tmp_path):
    spec = {("eval_check", "R"): set(range(12))}          # R better than N: test A must not pass
    r = a.analyse(make_run(tmp_path, by_index(spec)))
    assert not r["tests"]["test A"]["passes"] and r["tests"]["test A"]["per_seed"]["0"]["p"] == 1.0


def test_two_of_three_seeds_needed(tmp_path):
    # b=10, c=2 on seeds 0 and 1 (p 0.0193); seed 2 b=6, c=1 (p 0.0625): passes. With seed 1 also weak: fails.
    def make(weak):
        def f(w, p, s, i):
            k = s - 80
            if w != "eval_check":
                return 0
            seed = int(p[1])
            if p[0] == "L":                              # L-only: 10 strong (k 0-9), 6 weak (k 0-5)
                return int(k < (6 if seed in weak else 10))
            if p[0] == "R":                              # R-only: 2 strong, 1 weak
                return int(k in ({12} if seed in weak else {12, 13}))
            return 0
        return f
    r = a.analyse(make_run(tmp_path / "a", make({2})))
    ps = {s: v["p"] for s, v in r["tests"]["test B"]["per_seed"].items()}
    assert ps["0"] == pytest.approx(79 / 4096) and ps["2"] == pytest.approx(8 / 128)
    assert r["tests"]["test B"]["passes"] and not r["tests"]["test A"]["passes"]
    assert r["decision"].startswith("round 2's loss on 0/8 was not replicated at this budget; longer training raises")
    r = a.analyse(make_run(tmp_path / "b", make({1, 2})))
    assert not r["tests"]["test B"]["passes"]


def test_both_pass_reading_and_pooled_counts(tmp_path):
    spec = {("eval_check", "N"): set(range(12)), ("eval_check", "L"): set(range(12)), ("eval_check", "R"): {11},
            ("eval_searched", "R"): {0, 1}, ("eval_searched", "L"): {0}}
    r = a.analyse(make_run(tmp_path, by_index(spec)))
    assert r["tests"]["test A"]["passes"] and r["tests"]["test B"]["passes"]
    assert r["decision"].startswith("round 2's loss on 0/8 is real, and training long enough")
    pooled = r["reported"]["pooled counts (no pooled p: the seeds share their episodes)"]
    assert pooled["eval_check, N seeds pooled"] == 36 and pooled["eval_check, R seeds pooled"] == 3
    assert pooled["eval_searched, R seeds pooled"] == 3 * 2 * 4 and pooled["eval_searched, L seeds pooled"] == 3 * 4
    row = next(s for s in r["states"] if s["state"] == 8)
    assert row["N0"][0] == 12 and row["episodes"] == 96
    v = r["reported"]["L against R on the searched states"]["0"]
    assert (v["n"], v["first"], v["second"]) == (64, 4, 8)


def test_neither_passes_reading(tmp_path):
    r = a.analyse(make_run(tmp_path))
    assert r["decision"] == "round 2's loss on 0/8 was not replicated at this budget, and longer training was not shown to help"


def test_harness_threshold(tmp_path):
    one = {"N0": {(16, 8): (0, 1)}}
    assert a.harness(make_run(tmp_path / "a", harness_change=one))["same"] == 7
    assert a.gate(tmp_path / "a") is None
    two = {"N0": {(16, 8): (0, 1)}, "R0": {(51, 8): (1, 270)}}
    h = a.harness(make_run(tmp_path / "b", harness_change=two))
    assert h["same"] == 6 and not h["passes"] and a.gate(tmp_path / "b").startswith("harness")
    r = a.analyse(tmp_path / "b")
    assert r["decision"] == "not read: the harness failed"


def test_refuses_other_training_sets(tmp_path):
    eps = train_episodes(a.STEP12 / "runs" / "R0_1" / "provenance.jsonl")
    make_run(tmp_path / "a", train_lists={1: eps[:-1]})
    with pytest.raises(ValueError, match="L0_1"):
        a.check_training_data(tmp_path / "a")
    make_run(tmp_path / "b", train_lists={2: eps + eps[:1]})
    with pytest.raises(ValueError, match="L0_2"):
        a.check_training_data(tmp_path / "b")


def edit(path, **changes):
    rec = json.loads(path.read_text())
    for k, v in changes.items():
        if k in rec["settings"] or k in ("train_steps", "eval_seeds", "lora_dir", "eval_init_states", "seed"):
            rec["settings"][k] = v
        else:
            rec[k] = v
    path.write_text(json.dumps(rec) + "\n")


@pytest.mark.parametrize("where,changes", [
    ("runs/L0_0", {"train_steps": 2000}), ("runs/L0_1", {"seed": 0}), ("runs/L0_2", {"train_episode_sha256": {}}),
    ("runs/L0_0", {"data_dir": "/tmp/step12/task_0/round2_data/x"}), ("runs/L0_0", {"lora_r": 8}),
    ("eval_check/L1/part2", {"lora_dir": "/kaggle/input/x/step12/runs/R0_1/adapter"}),
    ("eval_check/N0/part0", {"eval_seeds": "16-31"}), ("eval_searched/R2/part1", {"eval_init_states": "2,5,6"}),
    ("eval_check/R0/part5", {"git_dirty": True}), ("eval_check/R0/part5", {"git_dirty_sfbc": True}),
    ("harness/R0", {"eval_seeds": "16-19"}), ("harness/N0", {"lora_dir": "/x/step10/runs/N0_1/adapter"}),
    ("eval_check/L0/part0", {"git_commit": "def"}), ("eval_check/L0/part0", {"max_steps": 600}),
    ("eval_check/L0/part0", {"policy_path": "/x/other"}), ("eval_check/L0/part0", {"task_id": 6})])
def test_refuses_wrong_provenance(tmp_path, where, changes):
    make_run(tmp_path)
    edit(tmp_path / where / "provenance.jsonl", **changes)
    with pytest.raises(ValueError):
        a.check_provenance(tmp_path)


def test_clean_run_provenance(tmp_path):
    p = a.check_provenance(make_run(tmp_path))
    n_eval = sum(len(parts) * len(pol) for _, _, parts, pol in a.SETS.values())
    assert p == {"records": 2 + 3 + n_eval, "commit": "abc"}


def test_refuses_missing_duplicate_and_extra_episodes(tmp_path):
    make_run(tmp_path)
    f = tmp_path / "eval_check" / "R1" / "part3" / "per_episode_lora.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="finish the run"):
        a.load_eval(tmp_path, "eval_check", "R1")
    f.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="twice"):
        a.load_eval(tmp_path, "eval_check", "R1")
    f.write_text("\n".join(lines + ["0,500,8,1,300"]) + "\n")
    with pytest.raises(ValueError, match="others"):
        a.load_eval(tmp_path, "eval_check", "R1")
    f.write_text("\n".join(lines[:-1] + ["6" + lines[-1][1:]]) + "\n")
    with pytest.raises(ValueError, match="task 6"):
        a.load_eval(tmp_path, "eval_check", "R1")


def test_protocol_constants():
    assert a.L_STEPS == round(2000 * 7309 / 5473)          # frames of R0_s's and N0_s's sets (archived recordings)
    assert not set(a.CHECK_SEEDS) & set(range(16, 80)) and set(a.SEARCHED_SEEDS) <= set(a.CHECK_SEEDS)
    assert sum(len(a.seeds_of(p)) for p in a.CHECK_PARTS) == 96 and sum(len(a.seeds_of(p)) for p in a.SEARCHED_PARTS) == 16
    episodes = 8 + sum(len(st) * len(seeds) * len(pol) for st, seeds, _, pol in a.SETS.values())
    assert episodes == 1256                               # the README's "at most 1,256 episodes"


def test_adapter_paths_are_the_protocols():
    assert a.adapter_path("N1") == "step10/runs/N0_1/adapter" and a.adapter_path("R2") == "step12/runs/R0_2/adapter"
    assert a.adapter_path("L0") == "step14/runs/L0_0/adapter"


def test_exactly_two_seeds_pass_test_a(tmp_path):
    def f(w, p, s, i):
        k, seed = s - 80, int(p[1])
        if w != "eval_check":
            return 0
        if p[0] == "N":
            return int(k < (6 if seed == 2 else 10))
        return int(p[0] == "R" and k in ({12} if seed == 2 else {12, 13}))
    r = a.analyse(make_run(tmp_path, f))
    assert [v["passes"] for v in r["tests"]["test A"]["per_seed"].values()] == [True, True, False]
    assert r["tests"]["test A"]["passes"]


def test_harness_counts_eight_episodes(tmp_path):
    h = a.harness(make_run(tmp_path))
    assert (h["same"], h["episodes"]) == (8, 8) and h["per_policy"]["R0"] == {"same": 4, "episodes": 4}


def test_committed_rows_with_exactly_the_wanted_episodes(tmp_path):
    rows = {(s, 8): (1, 300) for s in (16, 17, 18, 19)}
    write_csv(tmp_path / "task_0" / "eval" / "N0" / "part0" / "per_episode_lora.csv", rows)
    assert a.committed_rows(tmp_path, "N0", "16-19") == rows
    with pytest.raises(ValueError, match="no committed rows"):
        a.committed_rows(tmp_path, "N0", "16-20")


def test_one_dirty_flag_among_clean_ones_is_refused(tmp_path):
    make_run(tmp_path)
    edit(tmp_path / "eval_check" / "N1" / "part1" / "provenance.jsonl", git_dirty_noise_reachability=False,
         git_dirty_sfbc=True)
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(tmp_path)


def test_same_packages_in_another_key_order_are_accepted(tmp_path):
    make_run(tmp_path)
    for f in (tmp_path / "eval_check").rglob("provenance.jsonl"):
        edit(f, packages={"a": 1, "torch": "2.11.0"})
        break
    for f in tmp_path.rglob("provenance.jsonl"):
        rec = json.loads(f.read_text())
        rec["packages"] = {"torch": "2.11.0", "a": 1} if "eval_check" not in str(f) else {"a": 1, "torch": "2.11.0"}
        f.write_text(json.dumps(rec) + "\n")
    a.check_provenance(tmp_path)


def test_main_gate_exit_codes_and_report_files(tmp_path, capsys):
    make_run(tmp_path / "ok")
    assert a.main(["--root", str(tmp_path / "ok"), "--gate", "harness"]) == 0
    assert "gate harness passed" in capsys.readouterr().out
    make_run(tmp_path / "bad", harness_change={"N0": {(16, 8): (0, 1), (17, 8): (0, 1)}})
    assert a.main(["--root", str(tmp_path / "bad"), "--gate", "harness"]) == 1
    assert "gate harness FAILED: harness: 6 of 8" in capsys.readouterr().out
    out = tmp_path / "report.txt"
    assert a.main(["--root", str(tmp_path / "ok"), "--out", str(out)]) == 0
    assert out.read_text() == capsys.readouterr().out
    assert json.loads(Path(str(out) + ".json").read_text())["decision"] == a.analyse(tmp_path / "ok")["decision"]


def test_report_text(tmp_path):
    spec = {("eval_check", "N"): set(range(12)), ("eval_check", "R"): {10, 11}, ("eval_searched", "L"): {0}}
    text = a.format_report(a.analyse(make_run(tmp_path, by_index(spec))))
    lines = text.splitlines()
    assert lines[0] == "step 14: is round 2's loss on 0/8 real, and is it dilution? commit abc"
    assert lines[1] == "harness: 8 of 8 committed episodes reproduce (ok; N0 4/4, R0 4/4)"
    assert lines[2] == ("training sets: L0_0 50 episodes (as R), L0_1 50 episodes (as R), L0_2 50 episodes (as R); "
                        "the same 50 files (sha256) for all three")
    assert lines[3].startswith("0/8 (eval_check), of 96: N0 12 (6.6-20.8%), N1 12 (6.6-20.8%)")
    assert "test A, N0 against R0 on 0/8: 12 against 2 of 96 (only N 10, only R 0), one-sided p 0.000977" in lines
    assert "test A passes (p < 0.05 in at least 2 of 3 seeds)" in lines
    assert "test B does not pass (p < 0.05 in at least 2 of 3 seeds)" in lines
    assert "reported, L against R on the searched states, seed 0: 4 against 0 of 64, two-sided p 0.125" in lines
    assert lines[-2].startswith("reported, eval_check, L seeds pooled 0, eval_check, N seeds pooled 36")
    assert lines[-1] == "decision: round 2's loss on 0/8 is real; longer training was not shown to remove it"


def test_training_runs_reading_different_contents_are_refused(tmp_path):
    make_run(tmp_path)
    p = tmp_path / "runs" / "L0_1" / "provenance.jsonl"
    rec = json.loads(p.read_text())
    name = rec["settings"]["train_episodes"][0]
    rec["settings"]["train_episode_sha256"][name] = "1" * 64
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="different contents"):
        a.check_training_data(tmp_path)


def test_real_provenance_layout_is_accepted(tmp_path):
    """A record built by the code that writes it (vla_core.provenance.collect with self_filtered_bc's data_provenance
    keys merged into the settings, as write_provenance does): check_settings reads the sha256 where it really is."""
    from vla_core.provenance import collect

    eps = ["a.npz", "b.npz"]
    data = {"train_episodes": eps, "val_episodes": [], "train_episode_sha256": {e: "0" * 64 for e in eps},
            "val_episode_sha256": {}, "n_collected": 2, "n_collected_success": 2, "n_demonstrations": 0,
            "demo_datasets": []}
    settings = dict(a.TRAIN_SETTINGS, train_steps=a.L_STEPS, stage="train", seed=1, task_id=0, n_action_steps=10,
                    max_steps=520, data_dir="/tmp/step14/task_0/round2_data", val_init_states="", **data)
    rec = collect(str(ROOT), settings)
    assert "train_episode_sha256" in rec["settings"] and "train_episode_sha256" not in rec
    a.check_settings(tmp_path, tmp_path / "runs" / "L0_1" / "provenance.jsonl", rec)
    rec["settings"].pop("train_episode_sha256")
    with pytest.raises(ValueError, match="sha256"):
        a.check_settings(tmp_path, tmp_path / "runs" / "L0_1" / "provenance.jsonl", rec)
