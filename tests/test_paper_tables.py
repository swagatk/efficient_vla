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
    assert "| pooled, tasks 0, 4, 6, 7, 8 | 0.471 | 0.840 | 0.697 |  | 8 |" in t["4.1_headroom.md"]
    assert "| `repeat` | 20/350 (3.5-8.7%) |" in t["4.2_step5a_families.md"]
    assert "| `sampled` | 0/350 (0.0-1.0%) |" in t["4.2_step5a_families.md"]
    assert "| 2: four states pooled | 1/600 (0.0-0.9%) | 38/600 (4.5-8.6%) | 4.1e-11 | passes |" in t["4.3_step8_tests.md"]
    assert "| 7 | 7/38, 7/47 | 0/50 | 13/100 |" in t["4.4_step9_carry_over.md"]
    assert "Permutation p 0.408." in t["4.4_step9_carry_over.md"]


def test_table_shape():
    assert pt.table(["a", "b"], [[1, 2]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
