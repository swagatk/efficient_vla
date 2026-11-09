"""paper_tables.py: the committed tables are what the script makes from the committed result files, with known values."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
import paper_tables as pt  # noqa: E402


def test_committed_tables_regenerate_exactly():
    for name, text in pt.build(ROOT).items():
        assert (ROOT / "noise_reachability" / "paper_tables" / name).read_text() == text, name


def test_known_values_from_the_result_files():
    t = pt.build(ROOT)
    h = t["4.1_headroom.md"]
    assert "| 0, 4, 6, 7, 8 | 0.471 | 0.523 | 0.840 | 8 |" in h and "| 0, 4, 6, 7 | 0.480 | 0.467 | 0.825 | 7 |" in h
    assert "| 8 | 0.438 | 0.750 | 0.900 | 1 |" in h and "| 7 | 0.562 | 0.418 | 1.000 | 0 |" in h
    assert "tasks 0, 4, 6, 7, 8 0.697 in hindsight, 0.098 held out (p 0.22); tasks 0, 4, 6, 7 0.664 and -0.025 (p 0.55)." in h
    assert "| `repeat` | 20/350 (3.5-8.7%) |" in t["4.2_step5a_families.md"]
    assert "| `sampled` | 0/350 (0.0-1.0%) |" in t["4.2_step5a_families.md"]
    assert "| 2: four states pooled | 1/600 (0.0-0.9%) | 38/600 (4.5-8.6%) | 4.1e-11 | passes |" in t["4.3_step8_tests.md"]
    assert "| 7 | 7/38, 7/47 | 0/50 | 13/100 |" in t["4.4_step9_carry_over.md"]
    assert "Permutation p 0.408." in t["4.4_step9_carry_over.md"]


def test_table_shape():
    assert pt.table(["a", "b"], [[1, 2]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |\n"


def test_tex_table():
    md = pt.table(["Family", "Rate"], [["`repeat`", "20/350 (3.5-8.7%)"]]) + "\nPermutation p 0.408.\n"
    tex = pt.to_tex(md)
    assert tex.splitlines() == ["\\begin{tabular}{ll}", "\\toprule", "Family & Rate \\\\", "\\midrule",
                                "\\texttt{repeat} & 20/350 (3.5-8.7\\%) \\\\", "\\bottomrule", "\\end{tabular}",
                                "\\par\\smallskip{\\footnotesize Permutation p 0.408.}"]


def test_every_table_converts(tmp_path):
    for name, text in pt.build(ROOT).items():
        tex = pt.to_tex(text)
        md_rows = [l for l in text.splitlines() if l.startswith("|") and not l.startswith("| ---")]
        assert tex.count(" \\\\") == len(md_rows)                 # header and every body row, nothing dropped
        assert tex.startswith("\\begin{tabular}") and "\\bottomrule" in tex and "|" not in tex


def test_search_states_table_matches_steps_5a_and_7():
    t = pt.build(ROOT)["4.2b_search_states.md"]
    assert "| 0/8 | 5a | 0/70 | 2/70 | 8/70 | robust |" in t and "| 0/2 | 5a | 0/70 | 0/70 | 1/70 | fragile |" in t
    assert "| 6/8 | 7 | 0/70 | 0/70 | 3/70 | robust |" in t and "| 8/4 | 7 | 4/70 | 7/70 | 10/70 | robust |" in t
    assert t.count("\n| ") == 9                       # header and eight states


def test_harness_and_control_tables():
    t = pt.build(ROOT)
    h, c = t["3_harness.md"], t["3_controls.md"]
    assert "| 0 | 7 | 5 |  | 6 | 6 |" in h and "| 8 | 15 |  | 13 |  |  |" in h and "| 6 | 20 |  | 16 | 16 | 16 |" in h
    assert "| 5a | 0/9 | 8 | 12 | 11 |" in c and "| 7 | 6/0 | 14 | 13 | 5 |" in c and "| 8 | 7/4 | 15 |  | 12 |" in c
    assert c.count("\n| 8 |") == 4 and c.count("\n| 7 |") == 3


def test_step10_table_matches_the_report():
    t = pt.build(ROOT)["4.6_step10_learned.md"]
    assert "| Trained | 0/2, 0/5, 0/6, 0/7, 0/8 | 30 | 80 | 0 | 31, 31, 37 | 1, 2, 3 |" in t
    assert "| Trained | 7/33 | 33 | 32 | 0 | 29, 30, 26 | -- |" in t
    assert "| Easy | 0/0 | 0 | 16 | 0 | 16, 16, 16 | 0, 0, 0 |" in t
    assert "| Held out | 6/14, 6/20 | 0 | 64 | 0 | 0, 0, 0 | -- |" in t
    assert "N0, N1, N2 77, 71, 83 against base 1" in t and "N 99 against A 6" in t
    trained = [l for l in t.splitlines() if l.startswith("| Trained")]
    n = [sum(int(l.split("|")[6].split(",")[i]) for l in trained) for i in range(3)]
    assert n == [77, 71, 83]                       # the trained rows add up to test 1's counts


def test_train_counts_reads_both_name_forms():
    line = {"settings": {"train_episodes": ["full_n25_j5002_init08_seed5002.npz", "seed0_init10.npz", "x_init08_y.npz"]}}
    assert pt.train_counts(line) == {8: 2, 10: 1}


def test_step11_table_matches_the_report():
    t = pt.build(ROOT)["4.5_step11_replication.md"]
    assert "| A: all eight | 23/400 (3.7-8.5%) | 3/400 (0.2-2.2%) | 3.4e-05 | passes |" in t
    assert "| B: tasks 0, 4, 6 | 10/300 (1.6-6.0%) | 2/300 (0.1-2.4%) | 0.018 | passes |" in t
    assert "0/14 4-2" in t and "7/47 6-0" in t and "6/20 0-0" in t


# ---- step 12 (no result yet: a synthetic run through the real analysis gives the report's exact shape) -------------

def _step12_report(tmp_path):
    import json
    sys.path.insert(0, str(ROOT / "tests"))
    from test_analyze_step12 import make_run
    import analyze_step12 as s12

    def success(t, p, seed, s):          # R wins on searched states, N on held-out ones; distinct counts per seed
        i = int(p[1])
        if s in s12.SEARCHED[t]:
            return int(p[0] == "R" and seed % 4 != i or p[0] == "N" and seed % 8 == i)
        if s in s12.HELD_OUT[t]:
            return int(p[0] == "N" and seed % 2 == 0)
        return int(seed % 4 == 0)
    make_run(tmp_path / "run", tmp_path / "s10", success=success)
    return json.loads(json.dumps(s12.analyse(tmp_path / "run", tmp_path / "s10"))), s12


def test_step12_table_counts_match_the_analysis(tmp_path):
    r, s12 = _step12_report(tmp_path)
    t = pt.step12_table(r)
    rows = [l for l in t.splitlines() if l.startswith("| ") and "---" not in l][1:]
    assert len(rows) == 7                                   # 3 searched, 2 held-out and 2 check groups
    assert rows[0] == "| Searched | 0/2, 0/5, 0/6, 0/11 | 20/200 | 64 | 8, 8, 8 | 48, 48, 48 |"
    assert rows[2] == "| Searched | 7/38 | 5/50 | 32 | 4, 4, 4 | 24, 24, 24 |"
    assert rows[3] == "| Held out | 6/20 | -- | 32 | 16, 16, 16 | 0, 0, 0 |"
    assert rows[5] == "| Check | 0/8 | -- | 16 | 4, 4, 4 | 4, 4, 4 |"
    seeds = r["tests"]["test 1"]["per_seed"]
    for i in range(3):                                      # searched rows add up to test 1's counts, per seed
        cells = [r_.split(" | ") for r_ in rows[:3]]
        assert sum(int(c[5].strip(" |").split(", ")[i]) for c in cells) == seeds[str(i)]["first"]
        assert sum(int(c[4].split(", ")[i]) for c in cells) == seeds[str(i)]["second"]
    assert ("Test 1 (searched states, 192 paired episodes per seed): R0, R1, R2 144, 144, 144 against N0, N1, N2 "
            "24, 24, 24") in t and "; passes (p < 0.05 in at least 2 of 3 seeds)." in t
    v = r["reported"]["held out, R against N, seeds pooled"]
    assert f"Held out, seeds pooled: R 0 against N {v['second']} of {v['n']}, two-sided p" in t and v["second"] == 96
    assert "Check, seeds pooled: R 36 against N 36 of 144, two-sided p 1." in t   # 0/8 4 x 3, 7/33 8 x 3


def test_step12_table_refuses_a_state_with_another_role(tmp_path):
    import pytest
    r, _ = _step12_report(tmp_path)
    next(s for s in r["states"] if (s["task"], s["state"]) == (6, 20))["role"] = "check"
    with pytest.raises(AssertionError):
        pt.step12_table(r)
    for s in r["states"]:
        s["role"] = {(6, 20): "held out", (0, 5): "check"}.get((s["task"], s["state"]), s["role"])
    with pytest.raises(AssertionError):                     # one of a group's four states mislabelled
        pt.step12_table(r)
    r, _ = _step12_report(tmp_path / "b")
    r["states"].pop()
    with pytest.raises(AssertionError, match="states in the report"):
        pt.step12_table(r)


def test_step12_table_is_built_only_when_the_report_exists(tmp_path, monkeypatch):
    import pytest
    missing = tmp_path / "none.json"
    monkeypatch.setattr(pt, "OPTIONAL_INPUTS", {"step12": str(missing)})
    assert "4.7_step12_second_round.md" not in pt.build(ROOT) and "step12" not in pt.read_inputs(ROOT)
    import json
    r, _ = _step12_report(tmp_path)
    missing.write_text(json.dumps(r))
    with pytest.raises(AssertionError, match="E12"):        # a step-12 report but no committed E12 episode files
        pt.build(ROOT)
    monkeypatch.setattr(pt, "OVERVIEW", tuple(x for x in pt.OVERVIEW if x[0] != "E12"))
    t = pt.build(ROOT)
    assert t["4.7_step12_second_round.md"] == pt.step12_table(r) and pt.read_inputs(ROOT)["step12"] == str(missing)
    tex = pt.to_tex(t["4.7_step12_second_round.md"])
    assert tex.count(" \\\\") == 8 and "|" not in tex


# ---- Table I (overview) ---------------------------------------------------------------------------------------

def test_overview_counts_every_experiment_from_its_files():
    t = pt.build(ROOT)["1_overview.md"]
    want = {"E1": ("700", "8 of 50 states at 0"), "E2": ("900", "negative"), "E3": ("4,120", "not read"),
            "E4": ("1,430", "negative"), "E5a": ("1,249", "all 5 reached"), "E5b": ("5,280", "did not learn"),
            "E6": ("1,280", "yes, 1 of 48"), "E7": ("1,095", "failed; labels too coarse"),
            "E8": ("3,172", "both tests pass"), "E9": ("662", "not shown"), "E10": ("2,940", "both tests pass"),
            "E11": ("520", "both tests pass"), "E12": ("--", "running")}
    rows = {c[0]: (c[2], c[3]) for c in ([x.strip() for x in l.strip("|").split("|")] for l in t.splitlines()[2:])}
    assert rows == want


def test_overview_leaves_out_merged_copies_pilots_and_summaries():
    files = {e: {str(f.relative_to(ROOT)) for f in pt.episode_files(ROOT, pats)} for e, _, pats in pt.OVERVIEW}
    s = "week6/noise_reachability_step"
    assert f"{s}4_kaggle/run/eval/per_episode_A0.csv" not in files["E4"]               # merged copy of A0's parts
    assert f"{s}4_kaggle/run/eval/A0/part0/per_episode_lora.csv" in files["E4"]
    assert not any("pilot" in f for f in files["E5a"] | files["E5b"])
    assert not any("wrong_setting" in f or "_timing" in f or "consolidated" in f for f in files["E1"])
    assert f"{s}10_kaggle/run/task_0/collect_data/collection.csv" not in files["E10"]
    assert f"{s}8_kaggle/run/task_0/harness/seed0/per_episode_base.csv" in files["E8"]


def test_count_rows_and_readings(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("h\n1\n2\n\n")
    assert pt.count_rows([f, f]) == 4
    yes, no = {"passes": True}, {"passes": False}
    assert pt.passes(yes, yes) == "both tests pass" and pt.passes(yes, no) == "one of two tests passes"
    assert pt.passes(no, yes) == "one of two tests passes" and pt.passes(no, no) == "neither test passes"


def test_overview_outcomes_follow_the_reports():
    import copy
    data = {k: json.loads((ROOT / v).read_text()) for k, v in pt.INPUTS.items() if v.endswith(".json")}
    d = copy.deepcopy(data)
    d["step6"]["labels_hold"] = False
    d["step10"]["tests"]["test 2"]["passes"] = False
    first = next(s for s in d["step5a"]["states"] if int(s) != pt.CONTROL_5A)
    d["step5a"]["states"][first]["hits"] = 0
    d["step12"] = {"tests": {"test 1": {"passes": False}}}
    o = pt.outcomes(d)
    assert o["E6"] == "no, 1 of 48" and o["E10"] == "one of two tests passes" and o["E5a"] == "4 of 5 reached"
    assert o["E12"] == "test 1 does not pass" and pt.outcomes(data)["E12"] == "running"
    d["step12"]["tests"]["test 1"]["passes"] = True
    assert pt.outcomes(d)["E12"] == "test 1 passes"


def test_overview_tex_keeps_the_papers_column_layout():
    tex = pt.to_tex(pt.build(ROOT)["1_overview.md"], pt.COLSPECS["1_overview.md"])
    assert tex.startswith("\\begin{tabular}{@{}lp{3.3cm}rp{2.2cm}@{}}\n\\toprule\nExp. & Question & Episodes & Outcome")
    assert "E4 & LoRA on the policy's own successes & 1,430 & negative \\\\" in tex


def test_overview_inputs_hash_paths_and_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "OVERVIEW", (("E1", "q", ("d/*.csv",)),))
    (tmp_path / "d").mkdir()
    (tmp_path / "d/a.csv").write_text("h\n1\n")
    (tmp_path / "d/b.csv").write_text("h\n2\n")
    first = pt.overview_inputs(tmp_path)["E1"]
    import hashlib
    h = hashlib.sha256()
    for name in ("a.csv", "b.csv"):
        h.update(f"d/{name}".encode() + b"\0" + hashlib.sha256((tmp_path / "d" / name).read_bytes()).digest())
    assert first == {"files": 2, "sha256": h.hexdigest()}
    (tmp_path / "d/b.csv").write_text("h\n3\n")
    assert pt.overview_inputs(tmp_path)["E1"]["sha256"] != first["sha256"]
    monkeypatch.undo()
    assert pt.overview_inputs(ROOT)["E8"]["files"] == 108
