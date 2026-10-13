import json
import subprocess
import sys
from pathlib import Path

import pytest

HYBRID_DIR = Path(__file__).resolve().parent.parent / "hybrid_diffusion"
sys.path.insert(0, str(HYBRID_DIR))

from failure_overlap import failure_counts, independent_histogram, overlap_summary, permutation_p  # noqa: E402


def outcomes(n, failed):
    return {i: i not in failed for i in range(n)}


def test_failure_counts_and_mismatched_states():
    per_seed = {1: outcomes(4, {0, 1}), 2: outcomes(4, {1, 2})}
    assert failure_counts(per_seed) == {0: 1, 1: 2, 2: 1, 3: 0}
    with pytest.raises(ValueError):
        failure_counts({1: {0: True}, 2: {1: True}})


def test_independent_histogram_matches_binomial():
    assert independent_histogram([0.5, 0.5]) == pytest.approx([0.25, 0.5, 0.25])
    hist = independent_histogram([0.2, 0.5, 0.9])
    assert sum(hist) == pytest.approx(1.0)
    assert hist[0] == pytest.approx(0.8 * 0.5 * 0.1)
    assert hist[3] == pytest.approx(0.2 * 0.5 * 0.9)


def test_same_states_failing_every_seed_is_systematic():
    per_seed = {s: outcomes(50, set(range(30))) for s in (1, 2, 3)}
    summary = overlap_summary(per_seed, 0.6, 0.3)
    assert summary["oracle_rate"] == pytest.approx(0.4)
    assert summary["randomness_index"] == pytest.approx(0.0)
    assert summary["always_failing_states"] == list(range(30))
    assert summary["verdict"] == "mostly_systematic"
    assert permutation_p(per_seed, 500, 0) < 0.01


def test_disjoint_failures_are_random():
    per_seed = {1: outcomes(50, set(range(0, 15))), 2: outcomes(50, set(range(15, 30))), 3: outcomes(50, set(range(30, 45)))}
    summary = overlap_summary(per_seed, 0.6, 0.3)
    assert summary["oracle_rate"] == pytest.approx(1.0)
    assert summary["states_failing_in_k_seeds"] == [5, 45, 0, 0]
    assert sum(summary["expected_if_independent"]) == pytest.approx(50.0)
    assert summary["randomness_index"] > 1.0
    assert summary["verdict"] == "mostly_random"
    assert permutation_p(per_seed, 500, 0) > 0.9


def test_no_failures_is_undetermined():
    summary = overlap_summary({1: outcomes(10, set()), 2: outcomes(10, set())}, 0.6, 0.3)
    assert summary["randomness_index"] is None
    assert summary["verdict"] == "undetermined"


def test_permutation_p_is_reproducible():
    per_seed = {1: outcomes(20, {0, 1, 2, 3}), 2: outcomes(20, {2, 3, 4, 5})}
    assert permutation_p(per_seed, 300, 7) == permutation_p(per_seed, 300, 7)


def test_cli_runs_without_site_packages(tmp_path):
    for seed, failed in ((1, {0, 1}), (2, {1, 2})):
        d = tmp_path / f"eval_base_s{seed}"
        d.mkdir()
        episodes = [{"seed": i, "success": i not in failed} for i in range(6)]
        (d / "eval_results.json").write_text(json.dumps({"eval_noise_seed": seed, "episodes": episodes}))
    cmd = [sys.executable, "-S", str(HYBRID_DIR / "failure_overlap.py"), "--out_root", str(tmp_path),
           "--seeds", "1", "2", "--episodes", "6", "--n_perm", "200"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads((tmp_path / "failure_overlap_s1-2.json").read_text())
    assert saved["states_failing_in_k_seeds"] == [3, 2, 1]
    assert saved["always_failing_states"] == [1]
