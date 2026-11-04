"""paper_figures.py: the plotted values are the committed results (no plotting is run here)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "noise_reachability"))
import paper_figures as pf  # noqa: E402


def test_step8_points_are_the_report_rates_and_intervals():
    rows = pf.points(json.loads(pf.STEP8.read_text()))
    assert [r["label"] for r in rows] == ["6/8\nn=300", "0/11\nn=150", "4/11\nn=150", "6/12\nn=150", "7/33\nn=150"]
    assert rows[0]["repeat"][0] == pytest.approx(10 / 300) and rows[0]["sampled"][0] == pytest.approx(2 / 300)
    assert rows[4]["repeat"][0] == pytest.approx(25 / 150) and rows[4]["sampled"][0] == 0.0
    for r in rows:
        for fam in ("sampled", "repeat"):
            rate, lo, hi = r[fam]
            assert lo <= rate <= hi


def test_step1_grid_is_the_reachability_counts():
    tasks, grid = pf.state_grid(json.loads(pf.STEP1.read_text()))
    assert tasks == [0, 4, 6, 7, 8]
    zeros = sorted((t, s) for t, row in zip(tasks, grid) for s, c in enumerate(row) if c == 0)
    assert zeros == [(0, 2), (0, 5), (0, 6), (0, 7), (0, 8), (4, 2), (6, 8), (8, 4)]
    assert [sum(row) for row in grid] == [41, 79, 97, 90, 70]
