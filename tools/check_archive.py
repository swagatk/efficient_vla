#!/usr/bin/env python3
"""Check a downloaded notebook output (tools/archive_kaggle_output.sh) against the committed run it belongs to.

    python -I -S tools/check_archive.py --archive <archive>/<slug>/output --run week6/<step folder>/run \
        [--recordings <other archive output> ...] [--out <txt>]

The archive passes when:
  1. it is the same run: every committed file under --run (CSV, provenance, reports, adapter configs; logs are not
     committed) is in the archive at the same relative path with identical bytes;
  2. every committed training run (runs/<name>/provenance.jsonl) has its adapter weights,
     runs/<name>/adapter/adapter_model.safetensors, non-empty;
  3. every recording a training run lists in settings.train_episodes is in the archive or in a --recordings folder
     (found by file name; an earlier step's archive holds the recordings a later step reuses).
The run root inside the archive is the folder that holds the committed run's repo_commit.txt. Standard library.
"""
import argparse
import json
import sys
from pathlib import Path


def run_root(archive: Path, run: Path) -> Path:
    """The folder in ``archive`` whose repo_commit.txt equals the committed run's."""
    want = (run / "repo_commit.txt").read_bytes()
    found = [p.parent for p in sorted(archive.rglob("repo_commit.txt")) if p.read_bytes() == want]
    if len(found) != 1:
        raise ValueError(f"{len(found)} folders in {archive} hold the committed run's repo_commit.txt")
    return found[0]


def same_files(root: Path, run: Path) -> list[str]:
    """Committed files that are missing from the archive or differ from it."""
    bad = []
    for f in sorted(p for p in run.rglob("*") if p.is_file()):
        rel = f.relative_to(run)
        g = root / rel
        if not g.is_file():
            bad.append(f"missing {rel}")
        elif g.read_bytes() != f.read_bytes():
            bad.append(f"differs {rel}")
    return bad


def adapters(root: Path, run: Path) -> tuple[dict[str, int], list[str]]:
    """{training run: adapter bytes} and the runs whose weights are missing or empty."""
    sizes, bad = {}, []
    for prov in sorted(run.glob("runs/*/provenance.jsonl")):
        name = prov.parent.name
        w = root / "runs" / name / "adapter" / "adapter_model.safetensors"
        if not w.is_file() or w.stat().st_size == 0:
            bad.append(f"no adapter weights for {name}")
        else:
            sizes[name] = w.stat().st_size
    return sizes, bad


def recordings(run: Path, roots) -> tuple[int, list[str]]:
    """How many distinct recordings the training runs list, and those found in none of ``roots``."""
    have = {p.name for r in roots for p in Path(r).rglob("*.npz")}
    names = set()
    for prov in sorted(run.glob("runs/*/provenance.jsonl")):
        for line in prov.read_text().splitlines():
            names.update(json.loads(line)["settings"].get("train_episodes") or [])
    return len(names), sorted(f"no recording {n}" for n in names - have)


def check(archive, run, extra=()) -> tuple[list[str], list[str]]:
    archive, run = Path(archive), Path(run)
    root = run_root(archive, run)
    lines = [f"archive run root: {root}"]
    bad = same_files(root, run)
    n = sum(1 for p in run.rglob("*") if p.is_file())
    lines.append(f"committed files identical in the archive: {n - len(bad)}/{n}")
    sizes, b = adapters(root, run)
    bad += b
    lines.append(f"adapter weights: {len(sizes)} runs, {sum(sizes.values()) / 1e6:.1f} MB")
    k, b = recordings(run, [archive, *extra])
    bad += b
    lines.append(f"training recordings found: {k - len(b)}/{k}")
    return lines, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--recordings", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    lines, bad = check(a.archive, a.run, a.recordings)
    text = "\n".join(lines + bad + ["archive check: " + ("passed" if not bad else f"FAILED ({len(bad)} problems)")])
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
