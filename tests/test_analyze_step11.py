"""Step 11 analysis: pairing with step 9's repeat arm, tests A and B with Holm, refusals, and the committed step 9 files."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_step8 as a8  # noqa: E402
import analyze_step11 as a11  # noqa: E402
from analyze_step7 import fisher_greater  # noqa: E402
from test_analyze_step8 import episodes, provenance, scene, write  # noqa: E402

IDS = list(a11.IDS)


def make_step9(root, hits=None, **prov):
    """A step-9 random arm: {(task, state): ids that succeeded with repeat noise}."""
    hits = hits or {}
    change = prov.pop("settings", {})
    for t in a8.TASKS:
        folder = root / f"task_{t}" / "test" / "random"
        for s in a11.HELD_OUT[t]:
            settings = {"family": "repeat", "noise_ids": a11.ids_text(), "init_state": s}
            settings.update(change)
            provenance(folder, t, git_commit=prov.get("git_commit", a11.STEP9_COMMIT), settings=settings,
                       **{k: v for k, v in prov.items() if k != "git_commit"})
            write(folder / f"per_episode_repeat_state{s}.csv", episodes(t, s, IDS, hits.get((t, s), ())))
    return root


def make_run(root, hits=None, harness=None, scenes=None, drop=None, settings=None):
    """A step-11 folder: {(task, state): ids that succeeded with sampled noise}."""
    hits, harness, scenes = hits or {}, harness or {}, scenes or {}
    for t in a8.TASKS:
        folder = root / f"task_{t}"
        provenance(folder, t)
        for seed in range(3):
            n = harness.get(t, a8.HARNESS_REFERENCE[t])
            rows = [{"task_id": t, "seed": seed, "init_state": i, "success": int(seed * 10 + i < n), "steps": 520}
                    for i in range(10)]
            write(folder / "harness" / f"seed{seed}" / "per_episode_base.csv", rows)
        for s in a11.HELD_OUT[t]:
            scene(root, t, s, scenes.get((t, s), True))
            ids = IDS[:-1] if drop == (t, s) else IDS
            write(folder / "test" / "sampled" / f"per_episode_sampled_state{s}.csv", episodes(t, s, ids, hits.get((t, s), ())))
            st = {"family": "sampled", "noise_ids": a11.ids_text(), "init_state": s, "jitter_seeds": None}
            st.update((settings or {}).get((t, s), {}))
            provenance(folder / "test" / "sampled", t, settings=st)
    return root


def runs(tmp_path, rep=None, smp=None, **kw):
    return make_run(tmp_path / "run", smp, **kw), make_step9(tmp_path / "s9", rep)


def test_counts_pairs_and_both_tests(tmp_path):
    rep = {(7, 38): IDS[:6], (0, 14): IDS[:4], (4, 26): IDS[10:13]}
    smp = {(7, 38): IDS[:1], (0, 14): IDS[20:21]}
    root, s9 = runs(tmp_path, rep, smp)
    r = a11.analyse(root, s9)
    row = next(v for v in r["states"] if (v["task"], v["state"]) == (0, 14))
    assert (row["repeat"], row["sampled"], row["repeat_only"], row["sampled_only"]) == (4, 1, 4, 1)
    a, b = r["test A"], r["test B"]
    assert (a["repeat"], a["sampled"], a["episodes"]) == (13, 2, 400)
    assert (b["repeat"], b["sampled"], b["episodes"]) == (7, 1, 300)
    assert a["p"] == pytest.approx(fisher_greater(13, 2, 400)) and b["p"] == pytest.approx(fisher_greater(7, 1, 300))
    assert [7, 38] not in b["states"] and len(b["states"]) == 6
    assert a["passes"] and b["passes"] and r["decision"].startswith("replicated: fixed noise")


def test_only_task_7_carries_it(tmp_path):
    root, s9 = runs(tmp_path, {(7, 38): IDS[:10], (7, 47): IDS[10:16]})
    r = a11.analyse(root, s9)
    assert r["test A"]["passes"] and not r["test B"]["passes"]
    assert r["decision"].startswith("replicated pooled, but not without task 7")


def test_nothing_replicates(tmp_path):
    root, s9 = runs(tmp_path, {(0, 14): IDS[:1]}, {(0, 27): IDS[:1]})
    r = a11.analyse(root, s9)
    assert not r["test A"]["passes"] and not r["test B"]["passes"]
    assert r["decision"].startswith("not replicated")


def test_holm_uses_the_two_tests():
    # test A at 0.03 alone would pass 0.05, but under Holm the smaller p must be under 0.025
    assert a8.holm({"test A": 0.03, "test B": 0.04}) == {"test A": False, "test B": False}
    assert a8.holm({"test A": 0.02, "test B": 0.04}) == {"test A": True, "test B": True}
    assert a8.holm({"test A": 0.02, "test B": 0.05}) == {"test A": True, "test B": False}


@pytest.mark.parametrize("a, b, want", [(True, True, "replicated: fixed"), (True, False, "replicated pooled"),
                                        (False, True, "replicated without"), (False, False, "not replicated")])
def test_reading(a, b, want):
    assert a11.reading(True, a, b).startswith(want)
    assert a11.reading(False, a, b).startswith("not read: a harness")
    assert a11.reading(True, None, None).startswith("not read: no held-out")


def test_scene_failure_drops_the_state_from_both_arms(tmp_path):
    root, s9 = runs(tmp_path, {(6, 14): IDS[:5]}, scenes={(6, 14): False})
    r = a11.analyse(root, s9)
    assert r["scene_failed"] == [[6, 14]]
    assert r["test A"]["episodes"] == 350 and r["test A"]["repeat"] == 0
    assert (6, 14) not in {(t, s) for t, s, _ in a11.jobs(root)}


def test_all_scenes_fail_is_not_read(tmp_path):
    scenes = {(t, s): False for t, v in a11.HELD_OUT.items() for s in v}
    root, s9 = runs(tmp_path, scenes=scenes)
    r = a11.analyse(root, s9)
    assert r["decision"].startswith("not read: no held-out") and "test A" not in r


def test_harness(tmp_path):
    root, s9 = runs(tmp_path, harness={6: 13})
    assert a11.analyse(root, s9)["decision"].startswith("not read: a harness")
    assert a11.gate(root).startswith("harness failed: task 6")
    root2, _ = runs(tmp_path / "b", harness={6: 14, 0: 13})
    assert a11.gate(root2) is None


def test_refuses_missing_episode(tmp_path):
    root, s9 = runs(tmp_path, drop=(4, 26))
    with pytest.raises(ValueError, match="finish the run"):
        a11.analyse(root, s9)


def test_refuses_wrong_sampled_settings(tmp_path):
    for bad in ({"family": "repeat"}, {"noise_ids": "6000,6001"}, {"jitter_seeds": "5000"}, {"init_state": 3}):
        root, s9 = runs(tmp_path / str(len(list(tmp_path.iterdir()))), settings={(0, 14): bad})
        with pytest.raises(ValueError, match="sampled arm"):
            a11.analyse(root, s9)
        with pytest.raises(ValueError, match="sampled arm"):
            a11.gate(root)


def test_refuses_a_step9_folder_from_another_commit_or_setting(tmp_path):
    root = make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="step 9 provenance"):
        a11.analyse(root, make_step9(tmp_path / "a", git_commit="d" * 40))
    with pytest.raises(ValueError, match="step 9 provenance"):
        a11.analyse(root, make_step9(tmp_path / "b", settings={"family": "sampled"}))


def test_refuses_dirty_run(tmp_path):
    root, s9 = runs(tmp_path)
    provenance(root / "task_0" / "test" / "sampled", 0, git_dirty=True,
               settings={"family": "sampled", "noise_ids": a11.ids_text(), "init_state": 14})
    with pytest.raises(ValueError, match="dirty"):
        a11.analyse(root, s9)


def test_committed_step9_folder_is_accepted_and_its_counts_are_step9s():
    info = a11.check_step9(a11.STEP9)
    assert info["records"] >= 8 and len(info["sha256"]) == 8
    r9 = json.loads((a11.STEP9 / "step9_report.txt.json").read_text())
    from analyze_step5a import load_family
    for t in a8.TASKS:
        got = sum(sum(load_family(a11.STEP9 / f"task_{t}/test/random", "repeat", s, a11.IDS).values())
                  for s in a11.HELD_OUT[t])
        assert got == r9["tasks"][str(t)]["random"]


def test_report_and_cli(tmp_path, capsys):
    root, s9 = runs(tmp_path, {(4, 18): IDS[:2]})
    assert a11.main(["--root", str(root), "--step9", str(s9), "--jobs"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 8 and lines[0] == "0 14 " + a11.ids_text()
    out = tmp_path / "report.txt"
    assert a11.main(["--root", str(root), "--step9", str(s9), "--out", str(out)]) == 0
    text = out.read_text()
    assert "4/18: repeat 2/50 (0.5-13.7%), sampled 0/50 (0.0-7.1%), pairs repeat only 2, sampled only 0" in text
    assert "test A (8 states): repeat 2/400" in text and "test B (6 states): repeat 2/300" in text
    assert text.endswith("decision: not replicated: step 8's test 2 does not extend to these states\n")
    assert json.loads(Path(str(out) + ".json").read_text())["test A"]["repeat"] == 2
    assert a11.main(["--root", str(root), "--gate", "harness"]) == 0


def test_report_lines_exactly(tmp_path):
    root, s9 = runs(tmp_path, {(7, 38): IDS[:3], (0, 27): IDS[:1]}, {(7, 38): IDS[2:4]})
    r = a11.analyse(root, s9)
    text = a11.format_report(r)
    assert text.splitlines()[1] == ("repeat arm: step 9 at f2a067e, 8 provenance records, files NOT the committed folder; "
                                    "package versions the same as step 9's")
    assert "7/38: repeat 3/50 (1.3-16.5%), sampled 2/50 (0.5-13.7%), pairs repeat only 2, sampled only 1" in text
    a, b = r["test A"], r["test B"]
    assert (f"test A (8 states): repeat 4/400 (0.3-2.5%), sampled 2/400 (0.1-1.8%), one-sided Fisher p {a['p']:.3g}, "
            f"does not pass (Holm at 0.05); paired McNemar cross-check {a['mcnemar_cross_check']:.3g}") in text
    assert a["mcnemar_cross_check"] == pytest.approx(a11.mcnemar_greater(3, 1))
    assert f"test B (6 states): repeat 1/300 (0.0-1.8%), sampled 0/300 (0.0-1.2%), one-sided Fisher p {b['p']:.3g}" in text
    assert "held-out states that failed the scene check: []" in text and "harness task 6: base 20/30 against 20 (gap +0, ok)" in text


def test_test_b_without_states_when_only_task_7_is_read(tmp_path):
    scenes = {(t, s): False for t, v in a11.HELD_OUT.items() for s in v if t != 7}
    root, s9 = runs(tmp_path, {(7, 38): IDS[:8]}, scenes=scenes)
    r = a11.analyse(root, s9)
    assert r["test B"]["p"] is None and r["test B"]["passes"] is None and r["test A"]["passes"]
    assert r["decision"] == "replicated pooled; test B had no state to read"
    assert "test B: no states" in a11.format_report(r)


def test_cli_gate_fails_with_exit_1(tmp_path, capsys):
    root, _ = runs(tmp_path, harness={0: 0})
    assert a11.main(["--root", str(root), "--gate", "harness"]) == 1
    assert capsys.readouterr().out.startswith("harness failed: task 0")
    assert "FAILED" in a11.format_report(a11.analyse(root, tmp_path / "s9"))


def test_step9_settings_and_pinned_files_are_checked(tmp_path):
    root = make_run(tmp_path / "run")
    for i, bad in enumerate(({"max_steps": 300}, {"policy_path": "/s/other"}, {"init_state": 3})):
        with pytest.raises(ValueError, match="step 9 provenance"):
            a11.analyse(root, make_step9(tmp_path / f"s{i}", settings=bad))
    with pytest.raises(ValueError, match="step 9 provenance"):
        a11.analyse(root, make_step9(tmp_path / "d", git_dirty_noise_reachability=True))
    s9 = make_step9(tmp_path / "ok")
    info = a11.check_step9(s9)
    with pytest.raises(ValueError, match="differ from the committed"):
        a11.check_step9(s9, dict(info["sha256"], **{"task_7/test/random/per_episode_repeat_state38.csv": "0" * 64}))
    assert a11.check_step9(s9, info["sha256"])["pinned"]


def test_committed_step9_files_match_the_pins():
    info = a11.check_step9(a11.STEP9, a11.STEP9_SHA256)
    assert info["pinned"] and info["sha256"] == a11.STEP9_SHA256 and info["records"] == 40


def test_different_packages_are_reported_not_refused(tmp_path):
    root = make_run(tmp_path / "run")
    s9 = make_step9(tmp_path / "s9", packages={"torch": {"version": "2.10.0"}})
    r = a11.analyse(root, s9)
    assert r["packages_match_step9"] is False
    assert "package versions DIFFERENT from step 9's" in a11.format_report(r)
