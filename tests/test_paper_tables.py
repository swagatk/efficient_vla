"""paper_tables.py: the committed tables are what the script makes from the committed result files, with known values."""
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
