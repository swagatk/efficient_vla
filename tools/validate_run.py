"""Check a results folder before any number from it is read or reported.

Works on evaluate_all.py output (``per_episode.csv`` + ``provenance.jsonl`` at the top) and on per-task layouts
(``task_<t>/per_episode_<policy>.csv`` + ``task_<t>/provenance.jsonl``). Checks:

- provenance: every record has a git commit, the same commit, and a clean tree (``git_dirty`` and any
  ``git_dirty_*`` false); each ``--expect key=value`` matches the record's ``settings`` (lists and range strings
  such as "0-2" or "0,1,2" compare as sets of integers);
- episodes: each per-episode file group (same file name across task folders) holds every
  (task, seed, init_state) of ``--tasks x --seeds x --init_states`` exactly once, and nothing else;
- folders whose name starts with ``_`` (the notebook's ``_timing/`` check episode) are not part of the run and
  are skipped;
- logs: each ``--log_grep PATTERN`` occurs in every ``eval.log`` / ``eval.log.gz`` under the folder (e.g. the
  applied setting), and no log contains a Python traceback.

    python -S tools/validate_run.py <folder> --tasks 0 4 6 7 8 --seeds 3-15 --init_states 0-9 \
        --expect n_action_steps=10 --log_grep "n_action_steps': 10"

Exit code 0 if every check passes, 1 otherwise. Reads files only: run with ``python -S``.
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_range(spec):
    """'0-2', '0,1,2', '0 1 2', [0, 1, 2] or 7 -> sorted list of ints; None if not integer-like."""
    if isinstance(spec, bool):
        return None
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return sorted(int(x) for x in spec) if all(isinstance(x, int) and not isinstance(x, bool) for x in spec) else None
    out = set()
    try:
        for part in str(spec).replace(" ", ",").split(","):
            if part:
                lo, _, hi = part.partition("-")
                out.update(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        return None
    return sorted(out)


def values_match(actual, expected: str) -> bool:
    a, e = parse_range(actual), parse_range(expected)
    if a is not None and e is not None:
        return a == e
    return str(actual) == expected


def check_provenance(records, expect):
    errors = []
    if not records:
        return ["no provenance.jsonl records found"]
    commits = {r.get("git_commit") for r in records}
    if None in commits or len(commits) != 1:
        errors.append(f"provenance: expected one git commit, found {sorted(map(str, commits))}")
    for k in sorted({k for r in records for k in r if k == "git_dirty" or k.startswith("git_dirty_")}):
        dirty = sum(1 for r in records if r.get(k) is not False)
        if dirty:
            errors.append(f"provenance: {k} is not false in {dirty} of {len(records)} records")
    for key, value in expect.items():
        bad = [r.get("settings", {}).get(key, "<missing>") for r in records
               if not values_match(r.get("settings", {}).get(key, "<missing>"), value)]
        if bad:
            errors.append(f"provenance: settings.{key} should be {value}, found {sorted(set(map(str, bad)))} "
                          f"in {len(bad)} of {len(records)} records")
    return errors


def check_episodes(groups, tasks, seeds, states):
    errors = []
    if not groups:
        return ["no per_episode*.csv files found"]
    expected = {(t, s, i) for t in tasks for s in seeds for i in states}
    for name, keys in sorted(groups.items()):
        counts = Counter(keys)
        dup = [k for k, c in counts.items() if c > 1]
        missing, extra = expected - set(counts), set(counts) - expected
        if dup or missing or extra:
            errors.append(f"episodes {name}: {len(counts)} distinct of {len(expected)} expected; "
                          f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), {len(extra)} unexpected "
                          f"(e.g. {sorted(extra)[:3]}), {len(dup)} duplicated (e.g. {sorted(dup)[:3]})")
    return errors


def read_log(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        return f.read()


def check_logs(logs, patterns):
    errors = []
    if patterns and not logs:
        errors.append("no eval.log files found to check --log_grep against")
    for path in logs:
        text = read_log(path)
        for pat in patterns:
            if pat not in text:
                errors.append(f"log {path}: pattern {pat!r} not found")
        if "Traceback (most recent call last)" in text:
            errors.append(f"log {path}: contains a Python traceback")
    return errors


def _in_run(path, folder):
    return not any(part.startswith("_") for part in path.relative_to(folder).parts[:-1])


def collect(folder):
    folder = Path(folder)
    files = lambda pattern: [p for p in sorted(folder.rglob(pattern)) if _in_run(p, folder)]  # noqa: E731
    records = []
    for p in files("provenance.jsonl"):
        records += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for p in files("per_episode*.csv"):
        with open(p) as f:
            groups[p.name] += [(int(r["task_id"]), int(r["seed"]), int(r["init_state"])) for r in csv.DictReader(f)]
    logs = files("eval.log") + files("eval.log.gz")
    return records, dict(groups), logs


def validate(folder, tasks, seeds, states, expect, patterns):
    records, groups, logs = collect(folder)
    return check_provenance(records, expect) + check_episodes(groups, tasks, seeds, states) + check_logs(logs, patterns)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--tasks", required=True, help="e.g. '0 4 6 7 8' or 0-9")
    p.add_argument("--seeds", required=True)
    p.add_argument("--init_states", default="0-9")
    p.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                   help="a run setting the provenance must show; repeatable")
    p.add_argument("--log_grep", action="append", default=[], metavar="PATTERN",
                   help="text every eval.log must contain; repeatable")
    p.add_argument("--exclude", action="append", default=[], metavar="FILE_NAME",
                   help="per-episode file names to ignore (e.g. a subset copy); repeatable")
    a = p.parse_args(argv)
    expect = dict(e.split("=", 1) for e in a.expect)
    records, groups, logs = collect(a.folder)
    groups = {k: v for k, v in groups.items() if k not in a.exclude}
    errors = (check_provenance(records, expect)
              + check_episodes(groups, parse_range(a.tasks), parse_range(a.seeds), parse_range(a.init_states))
              + check_logs(logs, a.log_grep))
    print(f"{a.folder}: {len(records)} provenance records, {len(groups)} episode file groups, {len(logs)} logs")
    for e in errors:
        print("FAIL", e)
    print("OK: every check passed" if not errors else f"{len(errors)} check(s) failed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
