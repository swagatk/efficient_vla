"""check_target_positions: the single-position mask and the frame-weighted bin means."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noise_reachability"))
from check_target_positions import BINS, bin_means, only_position  # noqa: E402


def test_only_position_keeps_one_position_and_existing_padding():
    pad = torch.zeros(3, 50, dtype=torch.bool)
    pad[2, 40:] = True                                   # frame 2 runs past the episode end at position 40
    for k in (0, 9, 45):
        m = only_position(pad, k)
        assert m.shape == pad.shape and m.dtype == torch.bool
        assert (~m).sum(dim=1).tolist() == ([1, 1, 1] if k < 40 else [1, 1, 0])
        assert not m[0, k] and m[0, [j for j in range(50) if j != k]].all()
    assert not pad.any(dim=1)[:2].any()                  # the input is not modified


def test_bin_means_weight_by_frames():
    sums, counts = np.zeros(50), np.zeros(50)
    sums[0], counts[0] = 2.0, 4                          # mean 0.5 at position 0
    sums[1], counts[1] = 1.0, 1                          # positions 1-9: (1 + 9) / (1 + 3) = 2.5
    sums[2], counts[2] = 9.0, 3
    out = bin_means(sums, counts)
    assert list(out) == [f"{lo}-{hi - 1}" for lo, hi in BINS]
    assert out["0-0"] == 0.5 and out["1-9"] == 2.5 and np.isnan(out["10-19"])
