"""tools/check_archive.py: a downloaded notebook output passes only if it is the committed run plus its weights and
recordings."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_archive", ROOT / "tools/check_archive.py")
ca = importlib.util.module_from_spec(spec)
sys.modules["check_archive"] = ca
spec.loader.exec_module(ca)


def make(tmp_path):
    """A committed run and an archive of it: two training runs, one listing a recording kept in another archive."""
    run = tmp_path / "repo/run"
    for p, text in {"repo_commit.txt": "abc\n", "task_0/eval/N0/part0/per_episode_lora.csv": "seed,success\n1,0\n",
                    "runs/N0_0/provenance.jsonl": json.dumps({"settings": {"train_episodes": ["a.npz", "b.npz"]}}) + "\n",
                    "runs/N0_0/adapter/adapter_config.json": "{}",
                    "runs/R0_0/provenance.jsonl": json.dumps({"settings": {"train_episodes": ["c.npz"]}}) + "\n"}.items():
        (run / p).parent.mkdir(parents=True, exist_ok=True)
        (run / p).write_text(text)
    out = tmp_path / "arch/output/step10"
    shutil.copytree(run, out)
    (out / "logs").mkdir()
    (out / "logs/x.log").write_text("not committed")
    for name in ("N0_0", "R0_0"):
        (out / f"runs/{name}/adapter").mkdir(parents=True, exist_ok=True)
        (out / f"runs/{name}/adapter/adapter_model.safetensors").write_bytes(b"w" * 10)
    (out / "task_0/record").mkdir(parents=True)
    (out / "task_0/record/a.npz").write_bytes(b"1")
    (out / "task_0/record/b.npz").write_bytes(b"1")
    other = tmp_path / "other"
    other.mkdir()
    (other / "c.npz").write_bytes(b"1")
    return tmp_path / "arch/output", run, other


def test_a_complete_archive_passes(tmp_path):
    arch, run, other = make(tmp_path)
    lines, bad = ca.check(arch, run, [other])
    assert bad == []
    assert "committed files identical in the archive: 5/5" in lines
    assert "adapter weights: 2 runs, 0.0 MB" in lines and "training recordings found: 3/3" in lines
    assert ca.main(["--archive", str(arch), "--run", str(run), "--recordings", str(other)]) == 0


def test_missing_or_changed_files_weights_and_recordings_fail(tmp_path):
    arch, run, other = make(tmp_path)
    root = arch / "step10"
    _, bad = ca.check(arch, run)                                          # c.npz only in the other archive
    assert bad == ["no recording c.npz"]
    (root / "task_0/eval/N0/part0/per_episode_lora.csv").write_text("seed,success\n1,1\n")
    (root / "runs/R0_0/adapter/adapter_model.safetensors").write_bytes(b"")
    (root / "runs/N0_0/adapter/adapter_config.json").unlink()
    (root / "task_0/record/a.npz").unlink()
    _, bad = ca.check(arch, run, [other])
    assert bad == ["missing runs/N0_0/adapter/adapter_config.json", "differs task_0/eval/N0/part0/per_episode_lora.csv",
                   "no adapter weights for R0_0", "no recording a.npz"]
    assert ca.main(["--archive", str(arch), "--run", str(run), "--recordings", str(other)]) == 1


def test_the_run_root_must_match_the_committed_commit_once(tmp_path):
    arch, run, _ = make(tmp_path)
    (arch / "step10/repo_commit.txt").write_text("other\n")
    with pytest.raises(ValueError, match="0 folders"):
        ca.check(arch, run)
    (arch / "step10/repo_commit.txt").write_text("abc\n")
    shutil.copytree(arch / "step10", arch / "copy")
    with pytest.raises(ValueError, match="2 folders"):
        ca.check(arch, run)


def test_summary_and_report_file_on_failure(tmp_path):
    arch, run, other = make(tmp_path)
    root = arch / "step10"
    (root / "task_0/eval/N0/part0/per_episode_lora.csv").write_text("changed\n")
    (root / "task_0/record/a.npz").unlink()
    lines, bad = ca.check(arch, run, [other])
    assert "committed files identical in the archive: 4/5" in lines and "training recordings found: 2/3" in lines
    out = tmp_path / "check.txt"
    assert ca.main(["--archive", str(arch), "--run", str(run), "--recordings", str(other), "--out", str(out)]) == 1
    assert out.read_text().endswith("differs task_0/eval/N0/part0/per_episode_lora.csv\nno recording a.npz\n"
                                    "archive check: FAILED (2 problems)\n")
    (root / "task_0/eval/N0/part0/per_episode_lora.csv").write_text("seed,success\n1,0\n")
    (root / "task_0/record/a.npz").write_bytes(b"1")
    assert ca.main(["--archive", str(arch), "--run", str(run), "--recordings", str(other), "--out", str(out)]) == 0
    assert out.read_text().endswith("archive check: passed\n")
    for args in (["--run", str(run)], ["--archive", str(arch)]):
        with pytest.raises(SystemExit):
            ca.main(args)
