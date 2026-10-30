import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import analyze_step5a as a5  # noqa: E402
import deep_noise_search as dns  # noqa: E402


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(chunk_size=6, max_action_dim=5)
        self.received = []

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        self.received.append(noise)
        return noise


def fake_policy():
    return SimpleNamespace(model=FakeModel())


def test_noise_shapes_and_determinism():
    a, b = dns.make_noise("full", 3, 6, 5), dns.make_noise("full", 3, 6, 5)
    assert a.shape == (1, 6, 5) and a.dtype == torch.float32 and torch.equal(a, b)
    assert not torch.equal(a, dns.make_noise("full", 4, 6, 5))
    r = dns.make_noise("repeat", 3, 6, 5)
    assert r.shape == (1, 6, 5) and all(torch.equal(r[0, 0], r[0, k]) for k in range(6))
    assert not torch.equal(r[0, 0], a[0, 0])                    # the families do not share a stream
    assert not torch.equal(r, dns.make_noise("repeat", 4, 6, 5))
    assert dns.NOISE_SEED_BASE["full"] + 10 ** 5 < dns.NOISE_SEED_BASE["repeat"] < dns.JITTER_SEED_BASE
    with pytest.raises(ValueError):
        dns.make_noise("sampled", 0, 6, 5)


def test_jitter_stays_in_family_and_has_the_set_size():
    for fam in ("full", "repeat"):
        z, j = dns.make_noise(fam, 3, 50, 32), dns.make_noise(fam, 3, 50, 32, 5000)
        d = j - z
        assert 0.27 < float(d.std()) < 0.33 and torch.equal(j, dns.make_noise(fam, 3, 50, 32, 5000))
        assert not torch.equal(j, dns.make_noise(fam, 3, 50, 32, 5001))
    j = dns.make_noise("repeat", 3, 6, 5, 5000)
    assert all(torch.equal(j[0, 0], j[0, k]) for k in range(6))


def test_fixed_noise_ignores_torch_seed_and_needs_no_state_change():
    z = dns.make_noise("full", 0, 6, 5)
    z_before = z.clone()
    torch.manual_seed(0)
    before = torch.rand(1)
    p = fake_policy()
    dns.install_fixed_noise(p, z)
    state = torch.zeros(1, 8)
    torch.manual_seed(0)
    p.model.sample_actions(None, None, None, None, state)
    assert torch.equal(torch.rand(1), before)                    # no random numbers drawn
    p.model.sample_actions(None, None, None, None, state)
    assert all(torch.equal(n, z_before) for n in p.model.received) and len(p.model.received) == 2


def test_fixed_noise_refusals_and_uninstall():
    p = fake_policy()
    with pytest.raises(ValueError, match="shape"):
        dns.install_fixed_noise(p, torch.zeros(1, 5, 5))
    z = dns.make_noise("full", 0, 6, 5)
    dns.install_fixed_noise(p, z)
    with pytest.raises(RuntimeError):
        dns.install_fixed_noise(p, z)
    with pytest.raises(ValueError, match="supplies"):
        p.model.sample_actions(None, None, None, None, torch.zeros(1, 8), noise=z)
    with pytest.raises(ValueError, match="one observation"):
        p.model.sample_actions(None, None, None, None, torch.zeros(2, 8))
    dns.uninstall_fixed_noise(p)
    assert "sample_actions" not in vars(p.model)


def test_plan_jobs_seeds_and_resume():
    assert dns.plan_jobs("full", [0, 1, 2], None, {1001}) == [(0, 1000), (2, 1002)]
    assert dns.plan_jobs("sampled", [69], None, set()) == [(69, 1069)]
    assert dns.plan_jobs("full", [7], [5000, 5001], {5000}) == [(7, 5001)]
    with pytest.raises(ValueError):
        dns.plan_jobs("full", [7, 8], [5000], set())
    with pytest.raises(ValueError, match="one fixed noise"):
        dns.plan_jobs("sampled", [7], [5000], set())


def test_result_file_names(tmp_path):
    assert dns.result_file(tmp_path, "full", 2, [0], None).name == "per_episode_full_state2.csv"
    assert dns.result_file(tmp_path, "full", 2, [7], [5000]).name == "confirm_full_n7_state2.csv"


def test_constants_match_readme():
    assert a5.SYSTEMATIC_STATES == (2, 5, 6, 7, 8) and a5.CONTROL_STATE == 9 and len(a5.NOISE_IDS) == 70
    assert len(a5.CONTROL_IDS) == 35 and a5.CONFIRM_JITTER_SEEDS == (5000, 5001, 5002, 5003)
    assert (a5.MAX_CONFIRM_PER_CELL, a5.MIN_CONFIRMED, a5.MIN_CONTROL) == (3, 2, 3)
    assert (a5.HARNESS_REFERENCE, a5.HARNESS_TOLERANCE, a5.EPISODE_SEED_BASE) == (7, 6, dns.EPISODE_SEED_BASE)
    assert a5.FAMILIES == dns.FAMILIES


def test_zero_upper_known_values():
    assert a5.zero_upper(210) == pytest.approx(1 - 0.05 ** (1 / 210))
    assert 0.0141 < a5.zero_upper(210) < 0.0142 and 0.0418 < a5.zero_upper(70) < 0.0419


def write(path, seeds_wins):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for s, v in seeds_wins.items():
            w.writerow([0, s, 0, v, 520])


def make_run(tmp_path, wins=None, confirms=None, control=10, harness=7):
    """A complete run: ``wins`` {(family, state): noise ids}, ``confirms`` {(family, state, id): successes of 4}."""
    wins, confirms = wins or {}, confirms or {}
    for f in a5.FAMILIES:
        write(tmp_path / f"per_episode_{f}_state9.csv", {1000 + k: int(k < control) for k in a5.CONTROL_IDS})
        for s in a5.SYSTEMATIC_STATES:
            write(tmp_path / f"per_episode_{f}_state{s}.csv",
                  {1000 + k: int(k in wins.get((f, s), ())) for k in a5.NOISE_IDS})
    for (f, s, k), n in confirms.items():
        write(tmp_path / f"confirm_{f}_n{k}_state{s}.csv", {e: int(j < n) for j, e in enumerate(a5.CONFIRM_JITTER_SEEDS)})
    rec = {"git_commit": "abc", "git_dirty": False, "git_dirty_noise_reachability": False,
           "settings": {"task_id": 0, "n_action_steps": 10, "max_steps": 520, "policy_path": "snap"}}
    (tmp_path / "provenance.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "seed0").mkdir()
    (tmp_path / "harness" / "seed0" / "provenance.jsonl").write_text(json.dumps(rec) + "\n")
    with open(tmp_path / "harness" / "per_episode_base.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for j, (s, i) in enumerate((s, i) for s in range(3) for i in range(10)):
            w.writerow([0, s, i, int(j < harness), 520])
    return tmp_path


def test_all_unreached(tmp_path):
    r = a5.analyse(make_run(tmp_path))
    assert r["decision"].startswith("unreached") and r["states"][2]["episodes"] == 210
    assert r["states"][2]["upper_if_zero"] == pytest.approx(a5.zero_upper(210))
    assert "mixed rate 1.4%" in a5.format_report(r) and r["provenance"]["records"] == 2


def test_classes_and_confirmation_threshold(tmp_path):
    wins = {("full", 2): [5], ("repeat", 5): [6], ("sampled", 6): [1]}
    r = a5.analyse(make_run(tmp_path, wins, {("full", 2, 5): 2, ("repeat", 5, 6): 1}))
    assert r["decision"].startswith("mostly reachable")
    assert r["states"][2]["class"] == "robust" and r["states"][5]["class"] == "fragile"
    assert r["states"][6]["class"] == "sampled" and r["states"][7]["class"] == "unreached"
    assert r["decision"].startswith("mostly reachable: 3 of 5")


def test_mixed_reading_and_three_hit_cap(tmp_path):
    wins = {("full", 2): [1, 2, 3, 4]}
    r = a5.analyse(make_run(tmp_path, wins, {("full", 2, k): 0 for k in (1, 2, 3)}))
    assert r["decision"].startswith("mixed: 1 of 5") and r["states"][2]["confirmed"]["full"] == {"1": 0, "2": 0, "3": 0}
    assert a5.hits_to_confirm({9: 1, 1: 1, 4: 1, 2: 1, 3: 0}) == [1, 2, 4]


def test_missing_confirmation_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        a5.analyse(make_run(tmp_path, {("full", 2): [5]}))


def test_family_below_control_is_not_read(tmp_path):
    root = make_run(tmp_path, {("repeat", 2): [5]})
    write(root / "per_episode_repeat_state9.csv", {1000 + k: int(k < 2) for k in a5.CONTROL_IDS})
    r = a5.analyse(root)
    assert r["family_read"]["repeat"] is False and r["states"][2]["class"] == "unreached"
    assert r["states"][2]["episodes"] == 140 and r["unread_hits"] == {"repeat": {2: 1}}
    assert r["decision"].startswith("unreached") and "read family" in r["decision"]
    text = a5.format_report(r)
    assert "repeat 2 (not read)" in text and "hits of families not read" in text
    assert a5.zero_upper(140) == pytest.approx(r["states"][5]["upper_if_zero"])


def test_harness_and_sampled_control_stop_the_reading(tmp_path):
    assert a5.analyse(make_run(tmp_path, harness=14))["decision"].startswith("not read")
    (tmp_path / "b").mkdir()
    assert a5.analyse(make_run(tmp_path / "b", control=2))["decision"].startswith("not read")
    (tmp_path / "c").mkdir()
    assert a5.analyse(make_run(tmp_path / "c", harness=1))["harness"] == {"base": 1, "gap": -6, "ok": True}


def test_incomplete_or_repeated_refused(tmp_path):
    root = make_run(tmp_path)
    write(root / "per_episode_full_state2.csv", {1000 + k: 0 for k in range(69)})
    with pytest.raises(ValueError, match="finish the run"):
        a5.analyse(root)
    p = root / "x.csv"
    p.write_text("task_id,seed,init_state,success,steps\n0,1,0,1,5\n0,1,0,0,5\n")
    with pytest.raises(ValueError, match="twice"):
        a5.read_rows(p)
    bad = root / "harness" / "per_episode_base.csv"
    bad.write_text("task_id,seed,init_state,success,steps\n0,0,0,1,5\n")
    with pytest.raises(ValueError, match="harness"):
        a5.harness_check(bad)


def test_gates_and_provenance(tmp_path):
    root = make_run(tmp_path, harness=14)
    assert "harness failed" in a5.gate(root, "harness") and a5.gate(root, "control") is None
    with pytest.raises(SystemExit, match="STOP"):
        a5.main(["--root", str(root), "--gate", "harness"])
    (tmp_path / "b").mkdir()
    b = make_run(tmp_path / "b", control=2)
    assert a5.gate(b, "harness") is None and "sampled control" in a5.gate(b, "control")
    a5.main(["--root", str(b), "--gate", "harness"])
    (tmp_path / "c").mkdir()
    c = make_run(tmp_path / "c")
    good = (c / "provenance.jsonl").read_text()
    for old, new in (('"abc"', '"def"'), ('"git_dirty": false', '"git_dirty": true'),
                     ('"git_dirty_noise_reachability": false', '"git_dirty_noise_reachability": true'),
                     ('"n_action_steps": 10', '"n_action_steps": 1'), ('"task_id": 0', '"task_id": 1'),
                     ('"max_steps": 520', '"max_steps": 500'), ('"snap"', '"other"')):
        (c / "provenance.jsonl").write_text(good.replace(old, new))
        with pytest.raises(ValueError, match="provenance"):
            a5.analyse(c)


def test_success_must_be_binary(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("task_id,seed,init_state,success,steps\n0,1,0,2,5\n")
    with pytest.raises(ValueError, match="success 2"):
        a5.read_rows(p)


def test_hits_skip_unread_families(tmp_path, capsys):
    root = make_run(tmp_path, {("repeat", 2): [5], ("full", 5): [6]})
    write(root / "per_episode_repeat_state9.csv", {1000 + k: int(k < 2) for k in a5.CONTROL_IDS})
    capsys.readouterr()
    a5.main(["--root", str(root), "--hits"])
    assert capsys.readouterr().out == "full 5 6\n"


def test_choose_streams():
    r = lambda *t: {n: n / s for n, s in t}      # noqa: E731  streams, seconds per episode
    assert a5.choose_streams(r((2, 40), (4, 40), (6, 40)), cpus=12) == 6
    assert a5.choose_streams(r((2, 40), (4, 72), (6, 100)), cpus=12) == 2     # 4 gains 1.11x
    assert a5.choose_streams(r((2, 40), (4, 60), (6, 200)), cpus=12) == 4
    assert a5.choose_streams(r((2, 40), (4, 40), (6, 40)), cpus=5) == 4       # one CPU stays free
    assert a5.choose_streams(r((2, 40), (4, 69.5)), cpus=12) == 4             # gain just above 1.15x
    with pytest.raises(ValueError):
        a5.choose_streams({4: 40.0}, cpus=4)


def test_pilot_from_logs(tmp_path):
    for n, wall in ((2, 100), (4, 120)):
        d = tmp_path / f"streams_{n}"
        d.mkdir()
        (d / "s1.log").write_text("[search] x time=40.0s\n[search] y time=40.0s\nnoise\n")
        (d / "s2.log").write_text("[search] z time=9s\n")
        (d / "wall_seconds.txt").write_text(f"{wall}\n")
    out = tmp_path / "pilot.json"
    a5.main(["--pilot", str(tmp_path), "--cpus", "12", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["streams"] == 2 and r["stages"]["2"]["episodes"] == 3     # 4 streams: same episodes, more wall time
    assert r["episodes_per_hour"] == pytest.approx(3 / 100 * 3600)


def test_control_threshold_edge(tmp_path):
    r = a5.analyse(make_run(tmp_path, control=3))
    assert r["family_read"] == {"sampled": True, "full": True, "repeat": True} and "states" in r


def test_main_writes_report_and_lists_hits(tmp_path, capsys):
    root = make_run(tmp_path, {("full", 2): [5]}, {("full", 2, 5): 4})
    capsys.readouterr()
    a5.main(["--root", str(root), "--hits"])
    assert capsys.readouterr().out == "full 2 5\n"
    out = tmp_path / "report.txt"
    a5.main(["--root", str(root), "--out", str(out)])
    assert "decision: mixed: 1 of 5" in out.read_text()
    assert json.loads((tmp_path / "report.txt.json").read_text())["states"]["2"]["class"] == "robust"
