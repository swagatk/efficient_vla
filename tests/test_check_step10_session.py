"""check_step10_session.py: the resume check and the evaluation counts refuse what they should, on synthetic folders."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "check_step10_session", ROOT / "week6/noise_reachability_step10_kaggle/check_step10_session.py")
cs = importlib.util.module_from_spec(spec)
sys.modules["check_step10_session"] = cs
spec.loader.exec_module(cs)


def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_kept_from_accepts_appends_and_flags_rewrites(tmp_path):
    prev, now = tmp_path / "s1", tmp_path / "s2"
    _write(prev / "a.csv", "h\n1\n")
    _write(now / "a.csv", "h\n1\n2\n")          # appended: kept
    _write(prev / "b.csv", "h\n1\n")
    _write(now / "b.csv", "h\n0\n")             # rewritten
    _write(prev / "c.csv", "h\n")               # missing now
    kept, changed = cs.kept_from(prev, now)
    assert kept == 1 and changed == ["b.csv", "c.csv"]


def _eval(root, policy, rows):
    name = "base" if policy == "base" else "lora"
    _write(root / "task_0/eval" / policy / "part0" / f"per_episode_{name}.csv",
           "task_id,seed,init_state,success\n" + "".join(f"0,{s},{i},0\n" for s, i in rows))


def test_eval_counts_counts_and_refuses_repeats_and_strays(tmp_path):
    _eval(tmp_path, "base", [(16, 0), (16, 1)])
    assert cs.eval_counts(tmp_path) == {(0, "base"): (2, 208)}
    _eval(tmp_path, "A0", [(16, 0), (16, 0)])
    with pytest.raises(ValueError, match="repeated"):
        cs.eval_counts(tmp_path)
    _eval(tmp_path, "A0", [(15, 0)])            # seed 15 is not an evaluation seed
    with pytest.raises(ValueError, match="unplanned"):
        cs.eval_counts(tmp_path)


spec4 = importlib.util.spec_from_file_location(
    "compare_base_with_step4", ROOT / "week6/noise_reachability_step10_kaggle/compare_base_with_step4.py")
cb = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(cb)


def test_compare_base_counts_identical_episodes_on_shared_keys_only():
    old = {(16, 0): (1, 300), (16, 1): (0, 520), (17, 0): (1, 250)}
    new = {(16, 0): (1, 300), (16, 1): (1, 400), (99, 9): (1, 1)}   # (17, 0) and (99, 9) are not shared
    assert cb.compare(old, new) == {0: [1, 1, 1, 1], 1: [0, 1, 0, 1]}


def test_compare_base_episodes_refuses_repeats(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("task_id,seed,init_state,success,steps\n0,16,0,1,300\n0,16,0,1,300\n6,16,0,1,300\n")
    with pytest.raises(ValueError, match="twice"):
        cb.episodes([p])


def test_train_states_counts_episodes_per_init_state(tmp_path):
    names = ["full_n25_j5002_init08_seed5002.npz", "full_n30_init08_seed1030.npz", "seed0_init11.npz"]
    _write(tmp_path / "runs/N0_0/provenance.jsonl", json.dumps({"settings": {"train_episodes": names}}) + "\n")
    assert cs.train_states(tmp_path) == {"N0_0": {8: 2, 11: 1}}
