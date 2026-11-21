"""Step 16 analysis: the protocol's constants, provenance refusals, the harness threshold, complete parts, the part-1
gate at its edge and the table."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step16 as a  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d"


def write_csv(path, task, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in sorted(rows.items()):
            w.writerow([task, seed, state, succ, steps])


def record(settings, **extra):
    st = dict({"n_action_steps": 10, "max_steps": 520, "policy_path": POLICY}, **settings)
    return dict({"git_commit": "abc", "git_dirty": False, "git_dirty_noise_reachability": False,
                 "packages": {"torch": "2.11.0"}, "gpus": ["Tesla T4", "Tesla T4"], "settings": st}, **extra)


def add_prov(folder, rec):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "provenance.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def search(task, state, ids):
    return {"task_id": task, "family": "repeat", "init_state": state, "noise_ids": ids, "jitter_seeds": None,
            "record_dir": None, "lora_dir": None}


def make_run(root, success=None, parts=1, scene=None, harness_change=None, skip=None):
    """success(task, state, id) -> bool; skip: (task, state, part) whose file is left out."""
    success = success or (lambda t, s, k: False)
    scene = scene or {}
    for t in a.TASKS:
        d = root / f"task_{t}" / "scene"
        d.mkdir(parents=True, exist_ok=True)
        for s in a.MAP_STATES:
            (d / f"scene_state{s}.json").write_text(json.dumps({"task_id": t, "init_state": s,
                                                                 "same": scene.get((t, s), True)}))
            add_prov(d, record({"task_id": t, "init_state": s, "output_dir": str(d)}))
    rows = dict(a.committed_harness())
    rows.update(harness_change or {})
    d = root / "harness" / "base"
    write_csv(d / f"per_episode_repeat_state{a.HARNESS_STATE}.csv", 7, rows)
    add_prov(d, record(search(7, a.HARNESS_STATE, a.ids_spec(a.HARNESS_IDS))))
    for t in a.TASKS:
        for j in range(parts):
            d = a.map_dir(root, t, j)
            for s in a.MAP_STATES:
                if not scene.get((t, s), True) or (t, s, j) == skip:
                    continue
                rows = {(1000 + k, s): (int(success(t, s, k)), 300 if success(t, s, k) else 520)
                        for k in a.parse_ids(a.PARTS[j])}
                write_csv(a.map_file(root, t, j, s), t, rows)
                add_prov(d, record(search(t, s, a.PARTS[j])))
    return root


# ---- constants ---------------------------------------------------------------------------------------------------

def test_protocol_constants():
    committed = (set(range(0, 70)) | set(range(100, 400)) | set(range(1000, 1600)) | set(range(6000, 6050))
                 | set(range(7000, 7050)) | set(range(9000, 9400)))
    ids = a.parse_ids(",".join(a.PARTS))
    assert ids == list(range(10000, 10050)) and not committed & set(ids)
    assert [len(a.parse_ids(p)) for p in a.PARTS] == [a.IDS_PER_PROCESS] * 5 and a.IDS_PER_PROCESS == 10
    assert a.TASKS == (0, 4, 6, 7) and a.MAP_STATES == tuple(range(30))
    for t in a.TASKS:
        assert not set(a.HELD_OUT_STATES[t]) & set(a.MAP_STATES)
        assert set(a.HELD_OUT_STATES[t]) <= set(range(30, 50))
    assert a.HELD_OUT_STATES[7] == tuple(s for s in range(30, 50) if s not in (33, 38, 47))
    assert len(a.HELD_OUT_STATES[0]) == 20
    assert a.HARNESS_MIN_SAME == 11 and len(a.HARNESS_IDS) == 12


def test_committed_harness_has_six_successes_and_is_pinned(tmp_path):
    old = a.committed_harness()
    assert sum(v[0] for v in old.values()) == 6 and len(old) == 12
    bad = tmp_path / "step15"
    (bad / a.STEP15_FILE).parent.mkdir(parents=True)
    (bad / a.STEP15_FILE).write_text((a.STEP15 / a.STEP15_FILE).read_text())
    assert a.committed_harness(bad) == old          # another folder is not sha-checked but must hold the rows


# ---- harness -----------------------------------------------------------------------------------------------------

def _flip(n):
    old = a.committed_harness()
    keys = sorted(old)[:n]
    return {k: (1 - old[k][0], old[k][1]) for k in keys}


def test_harness_threshold_edges(tmp_path):
    assert a.harness(make_run(tmp_path / "r0"))["passes"]
    assert a.harness(make_run(tmp_path / "r1", harness_change=_flip(1)))["passes"]          # 11 of 12
    h = a.harness(make_run(tmp_path / "r2", harness_change=_flip(2)))                       # 10 of 12
    assert not h["passes"] and h["same"] == 10 and len(h["differ"]) == 2
    assert a.gate(tmp_path / "r2").startswith("harness: 10 of 12")
    assert a.gate(tmp_path / "r1") is None


def test_harness_counts_a_change_of_steps(tmp_path):
    old = a.committed_harness()
    k = sorted(old)[0]
    ch = {k: (old[k][0], old[k][1] - 1)}
    k2 = sorted(old)[1]
    ch[k2] = (old[k2][0], old[k2][1] - 1)
    assert not a.harness(make_run(tmp_path, harness_change=ch))["passes"]


def test_harness_refuses_an_unfinished_run(tmp_path):
    root = make_run(tmp_path)
    p = root / "harness" / "base" / f"per_episode_repeat_state{a.HARNESS_STATE}.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="finish the run"):
        a.harness(root)


# ---- provenance --------------------------------------------------------------------------------------------------

def test_provenance_accepts_a_clean_run(tmp_path):
    assert a.check_provenance(make_run(tmp_path))["commit"] == "abc"


@pytest.mark.parametrize("change", [
    {"git_dirty": True}, {"git_dirty_noise_reachability": True}, {"gpus": ["Tesla P100"]}, {"git_commit": "def"},
    {"packages": {"torch": "2.0"}}])
def test_provenance_refuses(tmp_path, change):
    root = make_run(tmp_path)
    add_prov(a.map_dir(root, 0, 0), record(search(0, 0, a.PARTS[0]), **change))
    with pytest.raises(ValueError):
        a.check_provenance(root)


@pytest.mark.parametrize("settings", [
    {"n_action_steps": 1}, {"family": "full"}, {"lora_dir": "/x/adapter"}, {"init_state": 30}, {"noise_ids": "10000-10010"},
    {"task_id": 8}, {"policy_path": "lerobot/smolvla_base"}])
def test_provenance_refuses_wrong_settings(tmp_path, settings):
    root = make_run(tmp_path)
    add_prov(a.map_dir(root, 0, 0), record(dict(search(0, 0, a.PARTS[0]), **settings)))
    with pytest.raises(ValueError):
        a.check_provenance(root)


def test_provenance_refuses_an_unknown_folder_and_wrong_part_ids(tmp_path):
    root = make_run(tmp_path)
    add_prov(root / "task_0" / "map" / "part9", record(search(0, 0, a.PARTS[0])))
    with pytest.raises(ValueError, match="unexpected folder"):
        a.check_provenance(root)
    root = make_run(tmp_path / "b")
    add_prov(a.map_dir(root, 0, 1), record(search(0, 0, a.PARTS[0])))
    with pytest.raises(ValueError, match="ids"):
        a.check_provenance(root)


def test_provenance_refuses_harness_with_other_ids(tmp_path):
    root = make_run(tmp_path)
    add_prov(root / "harness" / "base", record(search(7, 38, "9000-9011")))
    with pytest.raises(ValueError, match="harness ids"):
        a.check_provenance(root)


# ---- the map -----------------------------------------------------------------------------------------------------

def test_map_table_reads_every_part_and_drops_failed_scenes(tmp_path):
    root = make_run(tmp_path, success=lambda t, s, k: k % 2 == 0, parts=2, scene={(4, 3): False})
    table = a.map_table(root, 2)
    assert len(table) == (4 * 30 - 1) * 20
    assert (4, 3, 10000) not in table
    assert table[(0, 0, 10000)] == (1, 300) and table[(0, 0, 10001)] == (0, 520)
    with pytest.raises(ValueError, match="finish the run"):
        a.map_table(root, 3)
    with pytest.raises(ValueError):
        a.map_table(root, 0)


def test_map_table_refuses_a_missing_state_file_and_extra_rows(tmp_path):
    root = make_run(tmp_path, skip=(6, 7, 0))
    with pytest.raises(ValueError, match="6/7 part 0"):
        a.map_table(root, 1)
    root = make_run(tmp_path / "b")
    write_csv(a.map_file(root, 0, 0, 0), 0, {(1000 + 10010, 0): (0, 520)})
    with pytest.raises(ValueError, match="others"):
        a.map_table(root, 1)


def test_map_table_refuses_a_duplicated_episode_and_a_missing_scene(tmp_path):
    root = make_run(tmp_path)
    write_csv(a.map_file(root, 0, 0, 0), 0, {(1000 + 10000, 0): (0, 520)})
    with pytest.raises(ValueError, match="twice"):
        a.map_table(root, 1)
    root = make_run(tmp_path / "b")
    (root / "task_7" / "scene" / "scene_state29.json").unlink()
    with pytest.raises(ValueError, match="scene check of 7/29"):
        a.map_table(root, 1)


def _gate_run(tmp_path, n_inf):
    """n_inf states with both outcomes (one success at id 10000), the rest all failures."""
    states = [(t, s) for t in a.TASKS for s in a.MAP_STATES][:n_inf]
    return make_run(tmp_path, success=lambda t, s, k: (t, s) in states and k == 10000)


def test_part1_gate_edges(tmp_path):
    assert a.MIN_INFORMATIVE_FRACTION * 120 == 12
    _, t11 = a.analyse(_gate_run(tmp_path / "a", 11), 1)
    r12, _ = a.analyse(_gate_run(tmp_path / "b", 12), 1)
    assert not a.part1_gate(t11)["passes"] and a.part1_gate(t11)["informative_states"] == 11
    assert r12["part1_gate"]["passes"] and r12["decision"].startswith("part 1's gate passes")


def test_part1_gate_counts_dropped_states_as_not_informative(tmp_path):
    root = make_run(tmp_path, success=lambda t, s, k: k == 10000, scene={(0, s): False for s in range(30)})
    g = a.part1_gate(a.map_table(root, 1))
    assert g["informative_states"] == 90 and g["map_states"] == 120


def test_always_successful_state_is_not_informative(tmp_path):
    root = make_run(tmp_path, success=lambda t, s, k: True)
    assert a.part1_gate(a.map_table(root, 1))["informative_states"] == 0


def test_analyse_stops_at_a_failed_harness(tmp_path):
    r, table = a.analyse(make_run(tmp_path, harness_change=_flip(2)), 1)
    assert table is None and r["decision"] == "not read: the harness failed"
    assert "FAILED" in a.format_report(r)


def test_table_round_trip_and_report(tmp_path):
    root = make_run(tmp_path, success=lambda t, s, k: (s + k) % 3 == 0)
    r, table = a.analyse(root, 1)
    out = tmp_path / "t.csv"
    a.write_table(table, out)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 120 * 10 and rows[0] == {"task_id": "0", "init_state": "0", "noise_id": "10000",
                                                "success": str(int(10000 % 3 == 0)), "steps": "520"}
    assert r["successes"] == sum(int(x["success"]) for x in rows)
    text = a.format_report(r)
    assert "harness: 12 of 12" in text and "part 1 gate" in text


def test_main_gate_exit_codes(tmp_path, capsys):
    assert a.main(["--root", str(make_run(tmp_path / "ok")), "--gate", "harness"]) == 0
    assert a.main(["--root", str(make_run(tmp_path / "bad", harness_change=_flip(2))), "--gate", "harness"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_provenance_refuses_a_mixed_gpu_list(tmp_path):
    root = make_run(tmp_path)
    add_prov(a.map_dir(root, 0, 0), record(search(0, 0, a.PARTS[0]), gpus=["Tesla T4", "Tesla P100"]))
    with pytest.raises(ValueError, match="GPUs"):
        a.check_provenance(root)


def test_all_five_parts_and_their_report(tmp_path):
    root = make_run(tmp_path, success=lambda t, s, k: t == 4 and k == 10049, parts=5, scene={(6, 2): False})
    r, table = a.analyse(root, 5)
    assert len(table) == 119 * 50 and r["episodes"] == 119 * 50
    assert r["scene_failed"] == [[6, 2]]
    assert r["per_task"] == {"0": [0, 1500], "4": [30, 1500], "6": [0, 1450], "7": [0, 1500]}
    assert r["decision"] == "parts 1-5 complete: run step16_g1.py on the table" and "part1_gate" not in r
    r3, _ = a.analyse(root, 3)
    assert r3["decision"].endswith("only after every part is done")
    with pytest.raises(ValueError):
        a.map_table(root, 6)


def test_main_writes_report_json_and_table(tmp_path, capsys):
    root = make_run(tmp_path / "run", success=lambda t, s, k: s == 0)
    out, tab = tmp_path / "r.txt", tmp_path / "t.csv"
    assert a.main(["--root", str(root), "--out", str(out), "--table", str(tab)]) == 0
    assert out.read_text() == capsys.readouterr().out
    r = json.loads((tmp_path / "r.txt.json").read_text())
    assert r["parts"] == 1 and r["part1_gate"]["informative_states"] == 0 and r["successes"] == 40
    assert len(list(csv.DictReader(open(tab)))) == 1200
    bad = make_run(tmp_path / "bad", harness_change=_flip(2))
    assert a.main(["--root", str(bad), "--out", str(tmp_path / "b.txt"), "--table", str(tmp_path / "b.csv")]) == 0
    assert not (tmp_path / "b.csv").exists() and "FAILED" in (tmp_path / "b.txt").read_text()
