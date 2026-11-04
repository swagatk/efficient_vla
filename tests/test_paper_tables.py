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
