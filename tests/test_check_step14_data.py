"""check_step14_data: the linked folder must give exactly R0_s's committed training episodes for every seed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_step14_data as c  # noqa: E402
from test_self_filtered_bc import save_fake_episodes  # noqa: E402


def step12(root, names):
    for s in (0, 1, 2):
        d = root / "runs" / f"R0_{s}"
        d.mkdir(parents=True)
        (d / "provenance.jsonl").write_text(json.dumps({"settings": {"train_episodes": names[s]}}) + "\n")
    return root


def test_same_episodes_pass_and_a_difference_fails(tmp_path):
    save_fake_episodes(tmp_path / "data", [(10, True), (11, False), (12, True)])
    good = ["seed0_init10.npz", "seed0_init12.npz"]
    assert c.check(tmp_path / "data", step12(tmp_path / "a", {0: good, 1: good, 2: good})) == {0: True, 1: True, 2: True}
    bad = {0: good, 1: good[:1], 2: good + ["seed0_init11.npz"]}
    assert c.check(tmp_path / "data", step12(tmp_path / "b", bad)) == {0: True, 1: False, 2: False}


def test_order_matters(tmp_path):
    save_fake_episodes(tmp_path / "data", [(10, True), (12, True)])
    names = ["seed0_init12.npz", "seed0_init10.npz"]
    assert c.check(tmp_path / "data", step12(tmp_path / "a", {0: names, 1: names, 2: names}))[0] is False


def test_main_exit_codes(tmp_path):
    assert c.main([]) == 2


def test_main_exit_codes_follow_every_seed(monkeypatch):
    monkeypatch.setattr(c, "check", lambda d: {0: True, 1: True, 2: True})
    assert c.main(["/data"]) == 0
    monkeypatch.setattr(c, "check", lambda d: {0: True, 1: False, 2: True})
    assert c.main(["/data"]) == 1
    monkeypatch.setattr(sys, "argv", ["check_step14_data.py", "/data"])
    assert c.main() == 1
    monkeypatch.setattr(sys, "argv", ["check_step14_data.py"])
    assert c.main() == 2 and c.main(["/a", "/b"]) == 2
