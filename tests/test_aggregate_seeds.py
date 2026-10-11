import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hybrid_diffusion"))

from aggregate_seeds import load_successes, mcnemar_exact_p, paired_counts, summarise  # noqa: E402


def test_mcnemar_matches_known_values():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(12, 11) == 1.0
    assert mcnemar_exact_p(1, 6) == pytest.approx(0.125)
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact_p(6, 1) == mcnemar_exact_p(1, 6)


def test_paired_counts_and_summary():
    base = {0: True, 1: False, 2: False, 3: True}
    res = {0: True, 1: True, 2: False, 3: False}
    c = paired_counts(base, res)
    assert c == {"n": 4, "base_success": 2, "residual_success": 2, "base_only": 1, "residual_only": 1}

    better = paired_counts(base, {0: True, 1: True, 2: True, 3: True})
    s = summarise({1: c, 2: better})
    assert s["pooled_episodes"] == 8
    assert s["pooled_base_success"] == 4 and s["pooled_residual_success"] == 6
    assert s["base_rate_mean"] == pytest.approx(0.5) and s["base_rate_std"] == pytest.approx(0.0)
    assert s["residual_rate_mean"] == pytest.approx(0.75)
    assert s["pooled_base_only"] == 1 and s["pooled_residual_only"] == 3


def test_paired_counts_rejects_mismatched_init_states():
    with pytest.raises(ValueError):
        paired_counts({0: True}, {1: True})


def test_load_successes_rejects_incomplete_or_wrong_seed(tmp_path):
    p = tmp_path / "eval_results.json"
    p.write_text(json.dumps({"eval_noise_seed": 2, "episodes": [{"seed": 0, "success": True}, {"seed": 1, "success": False}]}))
    assert load_successes(str(p), 2, 2) == {0: True, 1: False}
    with pytest.raises(ValueError):
        load_successes(str(p), 1, 2)
    with pytest.raises(ValueError):
        load_successes(str(p), 2, 50)
