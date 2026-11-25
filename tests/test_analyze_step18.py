"""Tests for noise_reachability/analyze_step18.py: step 18's (M1) rule on synthetic pooled counts."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))
import analyze_step18 as a  # noqa: E402
from analyze_step7 import fisher_greater  # noqa: E402

N = 800


def counts(ref, s, r, **reported):
    return {"repeat": ref, "scaled_sigma0.5": s, "ar_repeat_rho0.9": r, **reported}


def test_ratio_interval_is_the_share_interval_mapped_to_a_ratio():
    lo, hi = a.ratio_interval(20, 20)
    assert lo < 1 < hi and lo == pytest.approx(1 / hi, rel=1e-6)          # symmetric share 0.5 -> reciprocal ends
    lo0, hi0 = a.ratio_interval(0, 30)
    assert lo0 == 0 and 0 < hi0 < 0.2
    assert a.ratio_interval(0, 0) == (0.0, float("inf"))
    assert a.ratio_interval(5, 0)[1] == float("inf")
    # more evidence -> narrower interval around the same ratio
    lo1, hi1 = a.ratio_interval(10, 20)
    lo2, hi2 = a.ratio_interval(40, 80)
    assert lo1 < lo2 < 0.5 < hi2 < hi1


def test_gate_needs_enough_constant_noise_successes():
    assert not a.rule(counts(a.MIN_REFERENCE - 1, 0, 0), N)["read"]
    assert a.rule(counts(a.MIN_REFERENCE, 0, 0), N)["read"]


def test_tests_are_fisher_with_holm_over_the_two():
    out = a.rule(counts(30, 5, 17), N)
    assert out["p"]["scaled_sigma0.5"] == fisher_greater(30, 5, N)
    assert out["p"]["ar_repeat_rho0.9"] == fisher_greater(30, 17, N)
    p_s, p_r = out["p"]["scaled_sigma0.5"], out["p"]["ar_repeat_rho0.9"]
    assert p_s < 0.025 and 0.025 < p_r < 0.05                    # Holm: smaller under 0.025, the other under 0.05
    assert out["conditions"]["scaled_sigma0.5"]["below"] and out["conditions"]["ar_repeat_rho0.9"]["below"]


def test_holm_stops_when_the_smaller_p_misses_its_level():
    x = next(k for k in range(30, 0, -1) if 0.025 <= fisher_greater(30, k, N) < 0.05)
    out = a.rule(counts(30, x, x), N)
    assert not out["conditions"]["scaled_sigma0.5"]["below"] and not out["conditions"]["ar_repeat_rho0.9"]["below"]


def test_keeps_half_needs_the_lower_end_strictly_above_half():
    out = a.rule(counts(40, 38, 4), N)
    s = out["conditions"]["scaled_sigma0.5"]
    assert s["ratio_interval"][0] > 0.5 and s["keeps_half"] and not s["below"]
    assert out["reading"].startswith("lower diversity")
    out = a.rule(counts(40, 20, 4), N)
    assert not out["conditions"]["scaled_sigma0.5"]["keeps_half"]


def test_overall_readings():
    assert a.rule(counts(40, 2, 3), N)["reading"].startswith("constancy")
    assert a.rule(counts(40, 2, 38), N)["reading"].startswith("persistence")
    assert a.rule(counts(40, 38, 2), N)["reading"].startswith("lower diversity")
    assert a.rule(counts(40, 2, 28), N)["reading"].startswith("not lower diversity")       # drift neither below nor half
    assert a.rule(counts(40, 30, 2), N)["reading"].startswith("not resolved")
    # below and keeping half at once needs many successes
    out = a.rule(counts(400, 300, 10), N)
    assert out["conditions"]["scaled_sigma0.5"]["below"] and out["conditions"]["scaled_sigma0.5"]["keeps_half"]
    assert out["reading"].startswith("partly lower diversity")


def test_reported_conditions_get_intervals_but_no_test():
    out = a.rule(counts(40, 2, 3, ar_repeat_rho0=0, scaled_sigma1=1, zero=0), N)
    assert set(out["p"]) == set(a.TESTED)
    assert out["conditions"]["ar_repeat_rho0"]["below"] is None and "reading" not in out["conditions"]["zero"]
    assert out["conditions"]["scaled_sigma1"]["ratio_interval"] == a.ratio_interval(1, 40)


def test_condition_readings():
    assert a.condition_reading(True, False) == "below constant noise"
    assert a.condition_reading(True, True).startswith("below constant noise, but keeps")
    assert a.condition_reading(False, True).startswith("matches")
    assert a.condition_reading(False, False) == "not resolved"


# ---- the run's files: a synthetic run folder ------------------------------------------------------------------------

import csv  # noqa: E402
import json  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d"


def write_csv(path, task, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in sorted(rows.items()):
            w.writerow([task, seed, state, succ, steps])


def add_prov(folder, settings, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    st = dict({"n_action_steps": 10, "max_steps": 520, "policy_path": POLICY, "jitter_seeds": None, "record_dir": None,
               "lora_dir": None, "feature_dir": None}, **settings)
    rec = dict({"git_commit": "abc", "git_dirty": False, "git_dirty_noise_reachability": False,
                "packages": {"torch": "2.11.0"}, "gpus": ["Tesla T4", "Tesla T4"], "settings": st}, **extra)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def add_harness(root, session, change=None):
    for (t, s), rows in a.committed_harness().items():
        rows = {**rows, **(change or {}).get((t, s), {})}
        write_csv(a.harness_file(root, session, t, s), t, {(1000 + k, s): v for k, v in rows.items()})
        add_prov(a.harness_dir(root, session, t, s), {"task_id": t, "init_state": s, "family": "ar_repeat", "rho": 1.0,
                                                      "sigma": None, "noise_ids": a.ids_spec(a.harness_ids(t, s))})


def make_run(root, success=None, harness_change=None, skip=None, settings_change=None):
    """success(label, task, state, id) -> bool; harness_change: {(t, s): {id: row}}; skip: (label, t, s, block)."""
    success = success or (lambda c, t, s, k: False)
    add_harness(root, 0, harness_change)
    for t, s in a.STATES:
        for label, (fam, rho, sigma, _) in a.CONDITIONS.items():
            for b, ids in enumerate(a.blocks(t, s, label)):
                if (label, t, s, b) == skip:
                    continue
                rows = {(1000 + k, s): (int(success(label, t, s, k)), 300 if success(label, t, s, k) else 520)
                        for k in ids}
                write_csv(a.block_file(root, t, s, label, b), t, rows)
                st = {"task_id": t, "init_state": s, "family": fam, "rho": rho, "sigma": sigma,
                      "noise_ids": a.ids_spec(ids)}
                st.update((settings_change or {}).get((label, t, s, b), {}))
                add_prov(a.block_dir(root, t, s, label, b), st)
    return root


def test_protocol_constants():
    ids = [k for t, s in a.STATES for c in a.CONDITIONS for k in a.condition_ids(t, s, c)]
    assert min(ids) == 20000 and max(ids) == 20000 + 200 * 7 + 99 and min(ids) > 10049
    per_state = {(t, s): set(a.condition_ids(t, s, "repeat")) for t, s in a.STATES}
    assert all(not (per_state[x] & per_state[y]) for x in per_state for y in per_state if x != y)   # own ids
    for t, s in a.STATES:
        assert set(a.condition_ids(t, s, "scaled_sigma1")) < set(a.condition_ids(t, s, "repeat"))   # same first ids
        assert len(a.condition_ids(t, s, "zero")) == 2
    assert [a.CONDITIONS[c][3] for c in a.CONDITIONS] == [100, 100, 100, 20, 20, 20, 2]
    from noise_processes import family_label
    assert all(family_label(*v[:3]) == c for c, v in a.CONDITIONS.items())
    assert family_label("ar_repeat", 1.0) == a.HARNESS_LABEL
    import step18_states  # noqa: F401  (calibration/ on the path below)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability" / "calibration"))


def test_states_are_the_selection_rules():
    import step18_states as st
    rows = st.select(st.candidates(st.tally(st.ROOT / "week6", st.REPEAT), st.tally(st.ROOT / "week6", st.SAMPLED)))
    assert tuple(k for k, *_ in rows) == a.STATES


def test_committed_harness_rows():
    h = a.committed_harness()
    assert h[(0, 8)] == {10000: (1, 424), 10001: (0, 520)} and h[(7, 33)] == {1459: (1, 327), 1450: (0, 520)}
    assert h[(7, 47)] == {6007: (1, 251), 6000: (0, 520)}


def test_harness_keeps_states_that_reproduce_and_needs_seven(tmp_path):
    root = make_run(tmp_path / "ok")
    h = a.gate(root)
    assert h["passes"] and len(h["kept"]) == 8
    one = make_run(tmp_path / "one", harness_change={(0, 8): {10000: (1, 425)}})
    h = a.gate(one)
    assert h["passes"] and [0, 8] not in h["kept"] and len(h["kept"]) == 7
    two = make_run(tmp_path / "two", harness_change={(0, 8): {10000: (1, 425)}, (7, 47): {6000: (1, 300)}})
    h = a.gate(two)
    assert not h["passes"] and len(h["kept"]) == 6
    r, eps = a.analyse(two)
    assert eps is None and r["reading"].startswith("not read")


def test_refuses_incomplete_or_foreign_files(tmp_path):
    with pytest.raises(ValueError, match="finish the run"):
        a.analyse(make_run(tmp_path / "skip", skip=("scaled_sigma0.5", 6, 8, 9)))
    bad = make_run(tmp_path / "rho", settings_change={("ar_repeat_rho0.9", 0, 6, 3): {"rho": 0.5}})
    with pytest.raises(ValueError, match="provenance: family"):
        a.analyse(bad)
    ids = make_run(tmp_path / "ids", settings_change={("repeat", 4, 11, 2): {"noise_ids": "1-10"}})
    with pytest.raises(ValueError, match="block 2 ids"):
        a.analyse(ids)
    extra = make_run(tmp_path / "extra")
    add_prov(extra / "task_0" / "other" / "block0", {"task_id": 0, "init_state": 8, "family": "repeat"})
    with pytest.raises(ValueError, match="unexpected folder"):
        a.analyse(extra)
    dirty = make_run(tmp_path / "dirty")
    add_prov(a.block_dir(dirty, 7, 33, "zero", 0), {"task_id": 7, "init_state": 33, "family": "zero", "rho": None,
                                                 "sigma": None, "noise_ids": a.ids_spec(a.blocks(7, 33, "zero")[0])},
             git_dirty=True)
    with pytest.raises(ValueError, match="dirty"):
        a.analyse(dirty)


def test_duplicated_rows_are_refused(tmp_path):
    root = make_run(tmp_path / "dup")
    f = a.block_file(root, 0, 8, "repeat", 0)
    write_csv(f, 0, {(1000 + a.blocks(0, 8, "repeat")[0][0], 8): (0, 520)})
    with pytest.raises(ValueError, match="twice"):
        a.analyse(root)


def test_known_answer_on_a_synthetic_run(tmp_path):
    # repeat: ids j < 6 succeed on every state (48 of 800); sigma 0.5: j == 0 (8); rho 0.9: j 0, 6, 7 (24; one shared
    # with repeat per state); rho 0.5: j 0
    # on the first ids; zero: succeeds on 7/33 only
    def success(c, t, s, k):
        j = (k - 20000) % 200
        return {"repeat": j < 6, "scaled_sigma0.5": j == 0, "ar_repeat_rho0.9": j in (0, 6, 7), "ar_repeat_rho0.5": j == 0,
                "ar_repeat_rho0": False, "scaled_sigma1": False, "zero": (t, s) == (7, 33)}[c]
    r, eps = a.analyse(make_run(tmp_path / "k", success=success))
    assert r["pooled"]["repeat"] == [48, 800] and r["pooled"]["scaled_sigma0.5"] == [8, 800]
    assert r["pooled"]["ar_repeat_rho0.9"] == [24, 800] and r["pooled"]["ar_repeat_rho0.5"] == [8, 160]
    assert r["pooled"]["zero"] == [2, 16] and len(eps) == 8 * (300 + 60 + 2)
    res = r["rule"]
    assert res["p"]["scaled_sigma0.5"] == fisher_greater(48, 8, 800)
    assert res["p"]["ar_repeat_rho0.9"] == fisher_greater(48, 24, 800)
    assert r["reading"].startswith("constancy")
    rep = r["reported"]
    assert rep["ar_repeat_rho0.5"]["repeat_same_ids"] == [48, 160]          # repeat on the same 20 ids per state
    assert rep["mcnemar_repeat_vs_rho0.9"] == {"repeat_only": 40, "rho0.9_only": 16,
                                               "p_one_sided": a.mcnemar_greater(40, 16)}
    assert rep["ar_repeat_rho0.5"]["successes"] == 8 and rep["ar_repeat_rho0.5"]["episodes"] == 160
    assert rep["ar_repeat_rho0.5"]["ratio_interval"] == list(a.ratio_interval(8, 48))
    assert rep["scaled_sigma0.5"]["repeat_same_ids"] == [48, 800]
    assert rep["scaled_sigma0.5"]["ratio_interval"] == list(a.ratio_interval(8, 48))
    assert rep["rho_trend"]["repeat"] == [48, 160] and rep["rho_trend"]["ar_repeat_rho0"] == [0, 160]
    assert rep["zero_outcomes"]["7/33"] == {"outcomes": [[1, 300], [1, 300]], "agree": True}
    assert r["per_state"]["0/8"]["repeat"]["successes"] == 6
    text = a.format_report(r)
    assert "constancy" in text and "repeat 48/800 (6.0%)" in text and "rho 0.9 only 16" in text


def test_main_writes_report_and_table(tmp_path):
    root = make_run(tmp_path / "m")
    assert a.main(["--root", str(root), "--gate", "harness", "--out", str(tmp_path / "h.json")]) == 0
    assert json.loads((tmp_path / "h.json").read_text())["passes"]
    assert a.main(["--root", str(root), "--out", str(tmp_path / "r.txt"), "--table", str(tmp_path / "t.csv")]) == 0
    assert json.loads((tmp_path / "r.txt.json").read_text())["reading"].startswith("not read")   # repeat 0 < 10
    assert len((tmp_path / "t.csv").read_text().splitlines()) == 1 + 8 * 362


def test_ratio_interval_of_one_success():
    lo, hi = a.ratio_interval(1, 0)
    assert lo > 0 and hi == float("inf")


def test_keeps_half_is_strict(monkeypatch):
    lo = a.ratio_interval(38, 40)[0]
    monkeypatch.setattr(a, "KEEP", lo)
    out = a.rule(counts(40, 38, 4), N)
    assert not out["conditions"]["scaled_sigma0.5"]["keeps_half"]
    assert out["conditions"]["scaled_sigma0.5"]["reading"] == "not resolved"


def test_committed_sources_are_sha_checked(monkeypatch):
    monkeypatch.setitem(a.SHA256, a.STEP8_733, "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        a.committed_harness()


def test_a_feature_or_adapter_run_is_refused(tmp_path):
    bad = make_run(tmp_path / "f", settings_change={("repeat", 0, 8, 0): {"feature_dir": "/x"}})
    with pytest.raises(ValueError, match="not a plain search"):
        a.analyse(bad)


def test_report_of_a_stopped_run_names_the_dropped_states(tmp_path):
    two = make_run(tmp_path / "two", harness_change={(0, 8): {10000: (1, 425)}, (7, 47): {6000: (1, 300)}})
    r, _ = a.analyse(two)
    text = a.format_report(r)
    assert "6 of 8 states" in text and "STOPPED" in text and "'0/8'" in text and "'7/47'" in text
    assert text.endswith("\n") and "reading: not read" in text and "pooled" not in text
    assert a.main(["--root", str(two), "--gate", "harness"]) == 1


def test_every_later_session_must_reproduce_the_kept_states(tmp_path):
    root = make_run(tmp_path / "s", harness_change={(0, 8): {10000: (1, 425)}})
    add_harness(root, 1, {(0, 8): {10000: (1, 425)}})              # a dropped state may differ again
    h = a.gate(root)
    assert h["passes"] and len(h["sessions"]) == 2 and [0, 8] not in h["kept"]
    bad = make_run(tmp_path / "b")
    add_harness(bad, 1, {(6, 12): {10018: (1, 232)}})
    h = a.gate(bad)
    assert not h["passes"] and h["later_differ"] == ["session 1: 6/12"]
    r, eps = a.analyse(bad)
    assert eps is None and "later session" in r["reading"] and "stack changed" in a.format_report(r)
    assert a.main(["--root", str(bad), "--gate", "harness"]) == 1


def test_harness_sessions_must_be_complete_and_without_gaps(tmp_path):
    root = make_run(tmp_path / "g")
    add_harness(root, 2)
    with pytest.raises(ValueError, match="sessions"):
        a.gate(root)
    part = make_run(tmp_path / "p")
    (a.harness_file(part, 0, 7, 47)).unlink()
    with pytest.raises(ValueError, match="finish the run"):
        a.gate(part)


def test_a_record_of_another_state_or_task_in_a_block_folder_is_refused(tmp_path):
    bad = make_run(tmp_path / "st", settings_change={("repeat", 0, 8, 1): {"init_state": 6}})
    with pytest.raises(ValueError, match="state 6"):
        a.analyse(bad)
    bad = make_run(tmp_path / "tk", settings_change={("zero", 4, 26, 0): {"task_id": 6}})
    with pytest.raises(ValueError, match="task 6"):
        a.analyse(bad)
    harness_rho = make_run(tmp_path / "h")
    add_prov(a.harness_dir(harness_rho, 0, 0, 8), {"task_id": 0, "init_state": 8, "family": "ar_repeat", "rho": 0.9,
                                                    "sigma": None, "noise_ids": "10000-10001"})
    with pytest.raises(ValueError, match="rho 0.9"):
        a.gate(harness_rho)


def test_settings_from_the_real_parser_pass_their_own_folder_and_fail_others(tmp_path):
    import deep_noise_search as d

    def rec(argv):
        st = vars(d.parse_args(argv + ["--output_dir", "x"]))
        st.update(policy_path=POLICY)
        return {"settings": st}
    for c, (fam, rho, sigma, _) in a.CONDITIONS.items():
        flags = ["--family", fam] + (["--rho", f"{rho:g}"] if rho is not None else []) + (
            ["--sigma", f"{sigma:g}"] if sigma is not None else [])
        ids = a.ids_spec(a.blocks(0, 8, c)[0])
        r = rec(flags + ["--task_id", "0", "--init_state", "8", "--noise_ids", ids])
        a.check_settings(tmp_path, a.block_dir(tmp_path, 0, 8, c, 0) / "provenance.jsonl", r)
        other = "scaled_sigma1" if c != "scaled_sigma1" else "scaled_sigma0.5"
        with pytest.raises(ValueError):
            a.check_settings(tmp_path, a.block_dir(tmp_path, 0, 8, other, 0) / "provenance.jsonl", r)
    h = rec(["--family", "ar_repeat", "--rho", "1", "--task_id", "0", "--init_state", "8", "--noise_ids", "10000-10001"])
    a.check_settings(tmp_path, a.harness_dir(tmp_path, 0, 0, 8) / "provenance.jsonl", h)
    with pytest.raises(ValueError):
        a.check_settings(tmp_path, a.block_dir(tmp_path, 0, 8, "repeat", 0) / "provenance.jsonl", h)


def test_a_drifting_noise_below_but_keeping_half_reads_as_persistence():
    out = a.rule(counts(400, 10, 300), N)
    r = out["conditions"]["ar_repeat_rho0.9"]
    assert r["below"] and r["keeps_half"] and out["reading"].startswith("persistence")


def test_a_later_session_with_other_packages_is_refused_at_the_gate(tmp_path):
    root = make_run(tmp_path / "pk")
    add_harness(root, 1)
    f = a.harness_dir(root, 1, 0, 8) / "provenance.jsonl"
    rec = json.loads(f.read_text().splitlines()[0])
    rec["packages"] = {"torch": "2.11.0+cu128"}
    f.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="packages changed"):
        a.gate(root)
