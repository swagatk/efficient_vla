"""Step 17 analysis: the feature check against step 16's committed rows, the duplicate-id identity, provenance refusals
and the features file step17_g1b.py reads."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
sys.path.insert(0, str(ROOT / "tests"))
import analyze_step17 as a  # noqa: E402
from test_analyze_step16 import add_prov, record, search, write_csv  # noqa: E402

DIM = 5


@pytest.fixture(autouse=True)
def small_dim(monkeypatch):
    """The fake records have DIM values; the real check expects 968."""
    monkeypatch.setattr(a, "EXPECTED_DIM", DIM)


def feats_of(t, s):
    return [float(t), float(s), 0.5, -1.25, float(t * 100 + s)]


def make_run(root, change=None, dup_feature_change=False, dup_row_change=False, feature_dir=None, skip_json=None,
             meta_change=None):
    """A complete run that reproduces step 16; ``change``: {(t, s): (success, steps)} for FEATURE_ID's episode."""
    old = a.committed_rows()
    change = change or {}
    for t in a.TASKS:
        d = a.out_dir(root, t)
        for s in a.MAP_STATES:
            rows = {}
            for k in a.ids_of(t, s):
                row = old[(t, s, k)]
                if k == a.FEATURE_ID and (t, s) in change:
                    row = change[(t, s)]
                if k == a.DUPLICATE_ID and dup_row_change:
                    row = (1 - row[0], row[1])
                rows[(1000 + k, s)] = row
                f = feats_of(t, s)
                if k == a.DUPLICATE_ID and dup_feature_change and (t, s) == (6, 0):
                    f = f[:-1] + [f[-1] + 1e-6]
                rec = {"task_id": t, "init_state": s, "noise_id": k, "seed": 1000 + k, "success": row[0],
                       "steps": row[1], "dim": DIM, "dtype": "float32", "inferences": 3, "features": f}
                rec.update((meta_change or {}).get((t, s, k), {}))
                if (t, s, k) != skip_json:
                    p = a.feature_path(root, t, s, k)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(rec))
            write_csv(d / f"per_episode_repeat_state{s}.csv", t, rows)
            st = search(t, s, a.ids_spec(a.ids_of(t, s)))
            st["feature_dir"] = feature_dir or f"/kaggle/working/step17/task_{t}/features/json"
            add_prov(d, record(st))
    return root


def test_protocol_constants():
    assert a.TASKS == (0, 4, 6, 7) and a.MAP_STATES == tuple(range(30))
    assert a.FEATURE_ID == 10000 and a.DUPLICATE_ID == 10001 and a.MIN_SAME == 114
    assert a.DUPLICATE_STATES == ((0, 0), (4, 0), (6, 0), (7, 0))
    assert a.ids_spec(a.ids_of(4, 0)) == "10000-10001" and a.ids_spec(a.ids_of(4, 1)) == "10000"


def test_a_run_that_reproduces_step16_passes_and_writes_every_state(tmp_path):
    r, f = a.analyse(make_run(tmp_path))
    c = r["feature_check"]
    assert c["passes"] and c["same"] == 120 and c["differ"] == [] and c["duplicates_bad"] == [] and c["dim"] == DIM
    assert len(f["rows"]) == 120 and f["dim"] == DIM and f["commit"] == "abc"
    row = next(x for x in f["rows"] if (x["task_id"], x["init_state"]) == (6, 7))
    assert row["features"] == feats_of(6, 7)
    assert a.gate(tmp_path) is None


def _changed(n):
    old = a.committed_rows()
    states = [(t, s) for t in a.TASKS for s in a.MAP_STATES][:n]
    return {ts: (1 - old[(*ts, a.FEATURE_ID)][0], old[(*ts, a.FEATURE_ID)][1]) for ts in states}


def test_feature_check_threshold_edges_and_dropped_states(tmp_path):
    r, f = a.analyse(make_run(tmp_path / "six", change=_changed(6)))
    assert r["feature_check"]["same"] == 114 and r["feature_check"]["passes"] and len(f["rows"]) == 114
    dropped = {tuple(x) for x in r["feature_check"]["differ"]}
    assert dropped == set(_changed(6)) and not dropped & {(x["task_id"], x["init_state"]) for x in f["rows"]}
    r, f = a.analyse(make_run(tmp_path / "seven", change=_changed(7)))
    assert r["feature_check"]["same"] == 113 and not r["feature_check"]["passes"] and f is None
    assert "113 of 120" in a.gate(tmp_path / "seven")


def test_a_change_of_steps_alone_counts_as_different(tmp_path):
    old = a.committed_rows()
    row = old[(0, 3, a.FEATURE_ID)]
    r, _ = a.analyse(make_run(tmp_path, change={(0, 3): (row[0], row[1] - 1)}))
    assert r["feature_check"]["differ"] == [[0, 3]]


def test_duplicate_id_must_match_bit_for_bit_and_reproduce_its_row(tmp_path):
    r, f = a.analyse(make_run(tmp_path / "f", dup_feature_change=True))
    assert r["feature_check"]["duplicates_bad"] == [[6, 0]] and not r["feature_check"]["passes"] and f is None
    r, _ = a.analyse(make_run(tmp_path / "r", dup_row_change=True))
    assert len(r["feature_check"]["duplicates_bad"]) == 4


@pytest.mark.parametrize("meta", [{"success": 1}, {"steps": 1}, {"seed": 7}, {"noise_id": 3}, {"dtype": "bfloat16"},
                                  {"inferences": 0}, {"dim": 6}, {"init_state": 3}, {"task_id": 0}])
def test_a_feature_record_that_does_not_match_its_episode_is_refused(tmp_path, meta):
    old = a.committed_rows()
    if "success" in meta:
        meta = {"success": 1 - old[(4, 2, a.FEATURE_ID)][0]}
    if "steps" in meta:
        meta = {"steps": old[(4, 2, a.FEATURE_ID)][1] + 1}
    with pytest.raises(ValueError):
        a.analyse(make_run(tmp_path, meta_change={(4, 2, a.FEATURE_ID): meta}))


def test_missing_features_unfinished_states_and_mixed_dimensions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        a.analyse(make_run(tmp_path / "j", skip_json=(7, 9, a.FEATURE_ID)))
    root = make_run(tmp_path / "u")
    (a.out_dir(root, 4) / "per_episode_repeat_state0.csv").write_text("task_id,seed,init_state,success,steps\n4,11000,0,0,520\n")
    with pytest.raises(ValueError, match="finish the run"):
        a.analyse(root)
    root = make_run(tmp_path / "d")
    p = a.feature_path(root, 0, 5, a.FEATURE_ID)
    rec = json.loads(p.read_text())
    rec.update(dim=DIM + 1, features=rec["features"] + [0.0])
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="dim"):                      # refused per record (EXPECTED_DIM)
        a.analyse(root)


def test_provenance_refuses_other_folders_ids_and_feature_dirs(tmp_path):
    with pytest.raises(ValueError, match="feature_dir"):
        a.check_provenance(make_run(tmp_path / "fd", feature_dir="/kaggle/working/step17/task_0/features/json"))
    root = make_run(tmp_path / "ids")
    add_prov(a.out_dir(root, 6), record({**search(6, 3, "10000-10001"),
                                        "feature_dir": "/x/task_6/features/json"}))
    with pytest.raises(ValueError, match="ids"):
        a.check_provenance(root)
    root = make_run(tmp_path / "folder")
    add_prov(root / "task_0" / "map", record({**search(0, 1, "10000"), "feature_dir": "/x/task_0/features/json"}))
    with pytest.raises(ValueError, match="unexpected folder"):
        a.check_provenance(root)
    root = make_run(tmp_path / "gpu")
    add_prov(a.out_dir(root, 0), {**record({**search(0, 1, "10000"), "feature_dir": "/x/task_0/features/json"}),
                                  "gpus": ["NVIDIA GeForce RTX 4090"]})
    with pytest.raises(ValueError, match="GPUs"):
        a.check_provenance(root)


def test_main_writes_the_report_and_the_features_file(tmp_path):
    root = make_run(tmp_path / "run")
    out, feats = tmp_path / "r.txt", tmp_path / "f.json"
    assert a.main(["--root", str(root), "--out", str(out), "--features", str(feats)]) == 0
    text = out.read_text()
    assert "120 of 120 states reproduce" in text and text.splitlines()[-1].startswith("decision: feature check passes")
    assert len(json.loads(feats.read_text())["rows"]) == 120
    assert a.main(["--root", str(root), "--gate", "features"]) == 0
    bad = make_run(tmp_path / "bad", change=_changed(7))
    assert a.main(["--root", str(bad), "--gate", "features"]) == 1


def test_features_file_is_what_step17_g1b_reads(tmp_path):
    import step17_g1b as g

    feats = tmp_path / "f.json"
    a.main(["--root", str(make_run(tmp_path / "run")), "--features", str(feats)])
    tasks, states = g.load_inputs(g.STEP16_TABLE, feats)
    assert {t: len(v) for t, v in states.items()} == {0: 30, 4: 30, 6: 30, 7: 30}
    Y, X = tasks[6]
    assert Y.shape == (30, 50) and X.shape == (30, DIM) and list(X[7]) == feats_of(6, 7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", 3])
def test_non_finite_or_non_float_features_are_refused(tmp_path, bad):
    root = make_run(tmp_path)
    p = a.feature_path(root, 4, 8, a.FEATURE_ID)
    rec = json.loads(p.read_text())
    rec["features"][2] = bad
    p.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="finite float"):
        a.analyse(root)


def test_the_real_feature_dimension_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "EXPECTED_DIM", 968)
    with pytest.raises(ValueError, match="expected 968"):
        a.analyse(make_run(tmp_path))


def test_a_row_of_another_state_is_refused(tmp_path):
    root = make_run(tmp_path)
    with open(a.out_dir(root, 6) / "per_episode_repeat_state3.csv", "a") as f:
        f.write("6,11005,4,0,520\n")                                     # one stray row of state 4 among state 3's
    with pytest.raises(ValueError, match="rows of another state"):
        a.analyse(root)


def test_one_inference_is_enough(tmp_path):
    r, _ = a.analyse(make_run(tmp_path, meta_change={(0, 1, a.FEATURE_ID): {"inferences": 1}}))
    assert r["feature_check"]["passes"]


def test_report_names_failing_duplicates_and_writes_no_features(tmp_path):
    root = make_run(tmp_path / "run", dup_feature_change=True)
    out, feats = tmp_path / "r.txt", tmp_path / "f.json"
    assert a.main(["--root", str(root), "--out", str(out), "--features", str(feats)]) == 0
    text = out.read_text()
    assert "FAILED on [[6, 0]]" in text and "all identical" not in text and "(FAILED)" in text
    assert text.splitlines()[-1] == "decision: not read: the feature check failed"
    assert not feats.exists()                                            # nothing for step17_g1b to read
    good = a.format_report(a.analyse(make_run(tmp_path / "good"))[0])
    assert "all identical" in good and "FAILED" not in good
