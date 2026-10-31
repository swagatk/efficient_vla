import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import analyze_step5b as a5  # noqa: E402

LOG_FIELDS = ["epoch", "init_state", "seed", "success", "steps", "decisions"]


def eps(succeed, seeds=range(16, 32)):
    """{(task, seed, init): success}; ``succeed(seed, init)`` says which."""
    return {(0, s, i): int(succeed(s, i)) for s in seeds for i in range(10)}


SYS = a5.split(eps(lambda s, i: 1))[0]


def first_n_sys(n):
    keep = sorted(SYS)[:n]
    return lambda s, i: (0, s, i) in keep


def base_eps():
    return eps(lambda s, i: i in (0, 1) and s < 28)


def with_sys(n):
    pick = first_n_sys(n)
    b = base_eps()
    return eps(lambda s, i: pick(s, i) or b[(0, s, i)])


def test_rule_edges_of_the_systematic_part():
    prior = base_eps()
    assert a5.seed_result(base_eps(), prior, with_sys(8))["meets"] is True
    assert a5.seed_result(base_eps(), prior, with_sys(7))["meets"] is False
    assert a5.seed_result(base_eps(), prior, with_sys(8))["learned"] is True
    assert a5.seed_result(base_eps(), with_sys(8), with_sys(8))["learned"] is False


def test_rule_edges_of_the_reachable_guard():
    drop = lambda k: {key: (0 if key in sorted(x for x in base_eps() if base_eps()[x])[:k] else v)  # noqa: E731
                      for key, v in with_sys(8).items()}
    assert a5.seed_result(base_eps(), base_eps(), drop(4))["meets"] is True
    assert a5.seed_result(base_eps(), base_eps(), drop(5))["meets"] is False


def result(meets=False, learned=False, gain=False):
    return {"meets": meets, "learned": learned, "reachable": {"gain": gain, "learned": gain}}


@pytest.mark.parametrize("seeds,training,text", [
    ([result(True, True), result(True, True), result()], [0, 0, 0], "hypothesis rejected"),
    ([result(True, True), result(True), result()], [0, 0, 0], "repeated form"),
    ([result(True), result(True), result(gain=True)], [0, 0, 0], "repeated form"),
    ([result(True, True), result(gain=True), result(gain=True)], [0, 0, 0], "reachable states, not to systematic"),
    ([result(True, True), result(gain=True), result()], [0.50, 0.50, 0], "did not transfer"),
    ([result(True, True), result(gain=True), result()], [0.50, 0.49, 0], "did not learn"),
    ([result(), result(), result()], [0, 0, 0], "did not learn"),
])
def test_reading_of_the_outcome(seeds, training, text):
    assert text in a5.decide(dict(enumerate(seeds)), dict(enumerate(training)), 0.4167)


def test_a_reachable_gain_of_the_repeated_form_alone_does_not_read_as_transfer():
    gain_only = {"meets": False, "learned": False, "reachable": {"gain": True, "learned": False}}
    text = a5.decide({0: gain_only, 1: gain_only, 2: gain_only}, {0: 0.5, 1: 0.5, 2: 0.5}, 0.5)
    assert "did not learn" in text


def test_training_gain_is_measured_from_the_untrained_actor():
    r = {k: result() for k in range(3)}
    assert "did not transfer" in a5.decide(r, {0: 0.75, 1: 0.75, 2: 0}, 0.67)      # at the edge: 0.67 + 0.08
    assert "did not learn" in a5.decide(r, {0: 0.75, 1: 0.74, 2: 0}, 0.67)
    rows = [[{"epoch": "0", "success": "1"}, {"epoch": "1", "success": "0"}], [{"epoch": "0", "success": "0"}]]
    assert a5.first_epoch_success(rows) == 0.5


def reach_with(net):
    extra = sorted(k for k, v in base_eps().items() if not v and k[2] in (3, 4))[:net]
    return {k: int(v or k in extra) for k, v in base_eps().items()}


def test_reachable_gain_edges_against_base_and_prior():
    assert a5.seed_result(base_eps(), base_eps(), reach_with(8))["reachable"]["gain"] is True
    assert a5.seed_result(base_eps(), base_eps(), reach_with(7))["reachable"]["gain"] is False
    assert a5.seed_result(base_eps(), base_eps(), reach_with(8))["reachable"]["learned"] is True
    assert a5.seed_result(base_eps(), base_eps(), reach_with(7))["reachable"]["learned"] is False
    assert a5.seed_result(base_eps(), reach_with(8), reach_with(8))["reachable"]["learned"] is False
    assert a5.seed_result(base_eps(), reach_with(8), reach_with(8))["reachable"]["gain"] is True


def test_a_failed_harness_is_not_read_and_arms_are_counted_separately(tmp_path):
    table = eps(lambda s, i: i == 0)                       # 16 of 160, 21 below the reference
    run_folder(tmp_path, base=table)
    prior = eps(lambda s, i: i in (0, 1, 2))               # 16 systematic, 32 reachable
    write_arm(tmp_path / "eval" / "shared", "prior", prior)
    write_arm(tmp_path / "eval" / "seed0", "actor_mean", eps(lambda s, i: i < 3, (16,)))
    write_arm(tmp_path / "eval" / "shared", "zero", eps(lambda s, i: i < 4, (16,)))
    out = a5.analyse(tmp_path, 6)
    assert out["decision"].startswith("not read: the harness failed")
    assert out["harness"]["gap"] == -21 and out["base"] == {"systematic": 0, "reachable": 16}
    assert out["prior"] == {"systematic": 16, "reachable": 32}
    assert out["secondary_of_10"] == {"actor_mean0": 3, "actor_mean1": 0, "actor_mean2": 0, "zero": 4}


def test_report_files_and_training_percentages(tmp_path):
    run_folder(tmp_path)
    assert a5.main(["--root", str(tmp_path), "--epochs", "6", "--out", str(tmp_path / "r.txt")]) == 0
    text = (tmp_path / "r.txt").read_text()
    assert "training, last 5 epochs: 50.0%" in text and "untrained actor (epoch 0, 3 seeds): 0.0%" in text
    assert json.loads((tmp_path / "r.txt.json").read_text())["training_prior"] == 0.0


def test_training_gate_reads_training_seed_0(tmp_path, capsys):
    run_folder(tmp_path)
    log = tmp_path / "train" / "seed1" / "train_log.csv"
    log.write_text("")
    assert a5.main(["--root", str(tmp_path), "--gate", "training", "--epochs", "6"]) == 0
    assert "50.0%" in capsys.readouterr().out


def write_arm(folder, arm, table):
    keys = sorted(table)
    for k, part in enumerate((keys[: len(keys) // 2], keys[len(keys) // 2:])):
        path = folder / f"part{k}" / f"per_episode_{arm}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task_id", "seed", "init_state", "success", "steps"])
            w.writerows([*key, table[key], 520] for key in part)


def run_folder(tmp_path, base=None, last=0.5):
    table = base if base is not None else base_eps()
    shared = tmp_path / "eval" / "shared"
    write_arm(shared, "base", table)
    write_arm(shared, "prior", table)
    write_arm(shared, "zero", eps(lambda s, i: 0, (16,)))
    for t in (0, 1, 2):
        write_arm(tmp_path / "eval" / f"seed{t}", "actor", table)
        write_arm(tmp_path / "eval" / f"seed{t}", "actor_mean", eps(lambda s, i: 0, (16,)))
        folder = tmp_path / "train" / f"seed{t}"
        folder.mkdir(parents=True)
        wins = round(last * 200)
        with open(folder / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            for k in range(240):
                w.writerow({"epoch": k // 40, "init_state": 10, "seed": k, "success": int(k >= 40 and k - 40 < wins),
                            "steps": 1, "decisions": 1})
    record = {"git_commit": "c", "git_dirty": False, "git_dirty_noise_reachability": False,
              "settings": {"task_id": 0, "n_action_steps": 10, "max_steps": 520, "policy_path": "x/a24f1f3804y",
                           "train_init_states": "10-49", "updates_per_epoch": 100}}
    (tmp_path / "train" / "seed0" / "provenance.jsonl").write_text(json.dumps(record) + "\n")
    return record


def test_analysis_reads_a_whole_run_and_refuses_a_partial_one(tmp_path):
    run_folder(tmp_path)
    out = a5.analyse(tmp_path, 6)
    assert out["base"] == {"systematic": 0, "reachable": 24}
    assert out["training_prior"] == 0.0 and "learned its training states" in out["decision"]    # 0% to 50%
    assert "actor0" in a5.format_report(out)
    (tmp_path / "eval" / "seed1" / "part0" / "per_episode_actor.csv").unlink()
    with pytest.raises(ValueError):
        a5.analyse(tmp_path, 6)


def test_harness_gate_edges(tmp_path):
    for n, ok in ((21, True), (20, False), (53, True), (54, False)):
        keys = sorted(eps(lambda s, i: 1))[:n]
        folder = tmp_path / str(n)
        write_arm(folder / "eval" / "shared", "base", eps(lambda s, i: (0, s, i) in keys))
        assert a5.main(["--root", str(folder), "--gate", "harness"]) == (0 if ok else 1)


def test_training_gate_edges(tmp_path):
    for last, ok in ((0.30, True), (0.295, False)):
        folder = tmp_path / str(last)
        folder.mkdir()
        run_folder(folder, last=last)
        assert a5.main(["--root", str(folder), "--gate", "training", "--epochs", "6"]) == (0 if ok else 1)


def test_missing_epochs_and_repeated_episodes_are_refused(tmp_path):
    run_folder(tmp_path)
    with pytest.raises(ValueError):
        a5.training_rows(tmp_path, 0, 7)
    log = tmp_path / "train" / "seed0" / "train_log.csv"
    text = log.read_text()
    log.write_text(text.replace("\n0,10,1,", "\n0,10,0,"))
    with pytest.raises(ValueError):
        a5.training_rows(tmp_path, 0, 6)
    log.write_text(text)
    bad = tmp_path / "eval" / "shared" / "part1" / "per_episode_base.csv"
    bad.write_text(bad.read_text() + bad.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError):
        a5.load_arm(tmp_path / "eval" / "shared", "base")


@pytest.mark.parametrize("change", [
    {"git_dirty": True}, {"git_commit": None}, {"git_dirty_noise_reachability": True}, {"n_action_steps": 1},
    {"max_steps": 100}, {"task_id": 1}, {"train_init_states": "0-49"}, {"updates_per_epoch": 8},
])
def test_provenance_refuses_other_runs(tmp_path, change):
    record = run_folder(tmp_path)
    other = json.loads(json.dumps(record))
    for k, v in change.items():
        (other["settings"] if k in other["settings"] else other)[k] = v
    (tmp_path / "train" / "seed1" / "provenance.jsonl").write_text(json.dumps(other) + "\n")
    with pytest.raises(ValueError):
        a5.check_provenance(tmp_path)


def test_provenance_refuses_another_snapshot_and_a_wrong_final_epoch(tmp_path):
    record = run_folder(tmp_path)
    other = json.loads(json.dumps(record))
    other["settings"].update(arm="actor", epoch=29)
    path = tmp_path / "eval" / "seed1" / "part0" / "provenance.jsonl"
    path.write_text(json.dumps(other) + "\n")
    assert a5.check_provenance(tmp_path, 30)["records"] == 2
    with pytest.raises(ValueError):
        a5.check_provenance(tmp_path, 37)
    other["settings"].update(arm="prior")
    path.write_text(json.dumps(other) + "\n")
    assert a5.check_provenance(tmp_path, 37)["records"] == 2
    head = json.loads(json.dumps(record))
    head["settings"]["policy_path"] = "lerobot/smolvla_libero"
    path.write_text("")
    (tmp_path / "train" / "seed0" / "provenance.jsonl").write_text(json.dumps(head) + "\n")
    with pytest.raises(ValueError):
        a5.check_provenance(tmp_path)


def test_provenance_skips_the_pilot_and_needs_records(tmp_path):
    record = run_folder(tmp_path)
    assert a5.check_provenance(tmp_path)["records"] == 1
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "provenance.jsonl").write_text(json.dumps(dict(record, git_commit="other")) + "\n")
    assert a5.check_provenance(tmp_path)["commit"] == "c"
    (tmp_path / "train" / "seed0" / "provenance.jsonl").unlink()
    with pytest.raises(ValueError):
        a5.check_provenance(tmp_path)
