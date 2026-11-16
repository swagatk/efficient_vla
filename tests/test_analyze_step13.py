"""Step 13 analysis: known answers, the two-sided 2-of-3 rule with one sign, R7 = N7, harness per session, refusals."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_step13 as a  # noqa: E402
from analyze_colab_t4_check import step10_rows  # noqa: E402

POLICY = "/root/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/a24f1f380461b665158d52acd140152c79e8b36d"


def write_csv(path, task, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in sorted(rows.items()):
            w.writerow([task, seed, state, succ, steps])


def record(settings, **extra):
    st = dict({"n_action_steps": 10, "max_steps": 520, "policy_path": POLICY, "stage": "eval"}, **settings)
    return dict({"git_commit": "abc", "git_dirty": False, "git_dirty_sfbc": False, "packages": {"torch": "2.11.0"},
                 "gpus": ["Tesla T4"], "settings": st}, **extra)


def write_prov(folder, rec):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.jsonl").write_text(json.dumps(rec) + "\n")


def make_run(root, success=None, sessions=1, harness_change=None):
    """success(policy, task, seed, state) -> 0/1 (default 0)."""
    success = success or (lambda p, t, s, i: 0)
    for k in range(1, sessions + 1):
        for t in a.TASKS:
            rows = dict(step10_rows(t, "N0"))
            rows.update((harness_change or {}).get((k, t), {}))
            d = root / "harness" / f"session{k}" / f"task_{t}"
            write_csv(d / "per_episode_lora.csv", t, rows)
            write_prov(d, record({"task_id": t, "eval_policies": "lora", "eval_init_states": str(a.HARNESS_STATE[t]),
                                  "eval_seeds": "16-19", "lora_dir": f"/content/drive/x/step10/runs/N{t}_0/adapter"}))
    for t in a.TASKS:
        for p in a.run_policies(t):
            for j, part in enumerate(a.PARTS):
                lo, hi = map(int, part.split("-"))
                rows = {(s, i): (success(p, t, s, i), 300) for s in range(lo, hi + 1) for i in a.STATES[t]}
                d = root / "eval" / p / f"task_{t}" / f"part{j}"
                write_csv(d / f"per_episode_{a.suffix(p)}.csv", t, rows)
                st = {"task_id": t, "eval_policies": "base" if p == "base" else "lora",
                      "eval_init_states": a.spec(a.STATES[t]), "eval_seeds": part}
                if p != "base":
                    st["lora_dir"] = f"/content/drive/x/{a.adapter_path(t, p)}"
                write_prov(d, record(st))
    return root


def episodes(n_first):
    """success for the first n_first (task, seed, state) episodes in a fixed order, else 0."""
    order = [(t, s, i) for t in a.TASKS for s in a.SEEDS for i in a.STATES[t]]
    return set(order[:n_first])


def test_constants_follow_the_committed_training_sets():
    assert a.trained_states() == a.TRAINED
    assert sum(len(v) for v in a.STATES.values()) == 59 and a.STATES[7] == tuple(s for s in range(30, 50) if s != 33)
    assert a.shared_weights_identical() and a.run_policies(7) == ["base", "N0", "N1", "N2"]
    assert len(a.SEEDS) * 59 == 472 and [len(range(int(p[:3]), int(p[4:]) + 1)) for p in a.PARTS] == [4, 4]
    run = sum(len(a.STATES[t]) * len(a.SEEDS) * len(a.run_policies(t)) for t in a.TASKS)
    assert run == 2848                                     # the README's "2,848 run"


def test_known_answer_and_r7_takes_n7s_episodes(tmp_path):
    good = episodes(40)                                     # N succeeds on 40 episodes (all on task 0), base on none
    r = a.analyse(make_run(tmp_path, lambda p, t, s, i: int(p[0] == "N" and (t, s, i) in good)))
    v = r["tests"]["N against base"]["per_seed"]["0"]
    assert (v["n"], v["first"], v["second"], v["first_only"], v["second_only"]) == (472, 40, 0, 40, 0)
    assert v["p_two_sided"] == pytest.approx(2 / 2 ** 40) and v["sign"] == 1
    assert r["tests"]["N against base"]["direction"] == "helps"
    assert r["tests"]["R against base"]["changes"] is False       # R ran nothing successful on tasks 0, 6
    assert r["per_task"]["7"]["R1"] == r["per_task"]["7"]["N1"]


def test_r_on_task_7_uses_n_episodes(tmp_path):
    seven = {(7, s, i) for s in a.SEEDS for i in a.STATES[7][:5]}           # 40 task-7 episodes
    r = a.analyse(make_run(tmp_path, lambda p, t, s, i: int(p[0] == "N" and (t, s, i) in seven)))
    assert r["tests"]["R against base"]["direction"] == "helps" and r["per_task"]["7"]["R2"][:2] == [40, 152]
    v = r["reported"]["R against N"]["1"]
    assert v["first_only"] == v["second_only"] == 0


def test_hurts_and_mixed_signs(tmp_path):
    bad = episodes(30)
    r = a.analyse(make_run(tmp_path / "a", lambda p, t, s, i: int(p == "base" and (t, s, i) in bad)))
    assert r["tests"]["N against base"]["direction"] == "hurts" and r["tests"]["R against base"]["direction"] == "hurts"
    assert r["decision"] == ("N (round 1) hurts task success on unseen initial states; R (round 2) hurts task success "
                             "on unseen initial states")
    # base succeeds on 30 episodes; N0 on those and 30 more (helps), N1 on none (hurts), N2 on the same 30: no change
    def f(p, t, s, i):
        base = episodes(60) - bad
        if p == "N0":
            return int((t, s, i) in episodes(60))
        if p == "N1":
            return 0
        return int((t, s, i) in base)
    r = a.analyse(make_run(tmp_path / "b", f))
    per = r["tests"]["N against base"]["per_seed"]
    assert [per[s]["sign"] for s in "012"] == [1, -1, 0] and not r["tests"]["N against base"]["changes"]


def test_two_of_three_seeds_needed(tmp_path):
    def f(k1, k2):
        def g(p, t, s, i):
            n = {"N0": 20, "N1": k1, "N2": k2}.get(p, 0)
            return int((t, s, i) in episodes(n))
        return g
    r = a.analyse(make_run(tmp_path / "a", f(6, 5)))          # p: 2^-19, 0.03125, 0.0625
    per = r["tests"]["N against base"]["per_seed"]
    assert [per[s]["significant"] for s in "012"] == [True, True, False] and r["tests"]["N against base"]["changes"]
    r = a.analyse(make_run(tmp_path / "b", f(5, 5)))          # p 0.0625 twice
    assert not r["tests"]["N against base"]["changes"]
    assert r["decision"].startswith("N (round 1): no change of 10 points or more shown")


def test_harness_every_session(tmp_path):
    one = {(2, 6): {(16, 8): (1, 1)}}
    h = a.harness(make_run(tmp_path / "a", sessions=2, harness_change=one))
    assert h["sessions"]["session2"]["same"] == 11 and h["passes"]
    two = {(2, 6): {(16, 8): (1, 1)}, (2, 7): {(17, 33): (0, 520)}}
    make_run(tmp_path / "b", sessions=3, harness_change=two)
    h = a.harness(tmp_path / "b")
    assert not h["passes"] and h["sessions"]["session1"]["passes"] and h["sessions"]["session3"]["passes"]
    assert a.gate(tmp_path / "b", "harness/session2").startswith("harness harness/session2: 10 of 12")
    assert a.gate(tmp_path / "b", "harness/session1") is None
    assert a.analyse(tmp_path / "b")["decision"] == "not read: a session's harness failed"


def test_refuses_r_run_on_task_7_and_unequal_weights(tmp_path, monkeypatch):
    make_run(tmp_path)
    write_csv(tmp_path / "eval" / "R0" / "task_7" / "part0" / "per_episode_lora.csv", 7, {(200, 30): (0, 520)})
    with pytest.raises(ValueError, match="R was run on task 7"):
        a.load_all(tmp_path)
    make_run(tmp_path / "b")
    monkeypatch.setitem(a.WEIGHTS_SHA256, "R7_1", "0" * 64)
    with pytest.raises(ValueError, match="weights differ"):
        a.load_all(tmp_path / "b")


def edit(path, **changes):
    rec = json.loads(path.read_text())
    for k, v in changes.items():
        (rec["settings"] if k in rec["settings"] or k in ("lora_dir",) else rec)[k] = v
    path.write_text(json.dumps(rec) + "\n")


@pytest.mark.parametrize("where,changes", [
    ("eval/N1/task_6/part0", {"lora_dir": "/x/step12/runs/R6_1/adapter"}),
    ("eval/R2/task_0/part1", {"lora_dir": "/x/step12/runs/R0_1/adapter"}),
    ("eval/base/task_7/part0", {"eval_policies": "lora"}), ("eval/N0/task_0/part1", {"eval_seeds": "200-203"}),
    ("eval/N0/task_0/part0", {"eval_init_states": "30,31"}), ("eval/N0/task_0/part0", {"task_id": 6}),
    ("eval/N0/task_0/part0", {"gpus": ["NVIDIA L4"]}), ("eval/N0/task_0/part0", {"gpus": ["Tesla T4", "NVIDIA A100"]}),
    ("eval/N0/task_0/part0", {"git_dirty_sfbc": True}), ("eval/N0/task_0/part0", {"git_commit": "def"}),
    ("eval/N0/task_0/part0", {"max_steps": 600}), ("eval/N0/task_0/part0", {"policy_path": "/x/other"}),
    ("eval/N0/task_0/part0", {"packages": {"torch": "2.12"}}), ("eval/N0/task_0/part0", {"stage": "train"}),
    ("harness/session1/task_6", {"lora_dir": "/x/step10/runs/N6_1/adapter"}),
    ("harness/session1/task_7", {"eval_seeds": "16-20"})])
def test_refuses_wrong_provenance(tmp_path, where, changes):
    make_run(tmp_path)
    edit(tmp_path / where / "provenance.jsonl", **changes)
    with pytest.raises(ValueError):
        a.check_provenance(tmp_path)


def test_refuses_missing_duplicate_and_foreign_episodes(tmp_path):
    make_run(tmp_path)
    f = tmp_path / "eval" / "R1" / "task_6" / "part1" / "per_episode_lora.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="finish the run"):
        a.load_eval(tmp_path, "R1", 6)
    f.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="twice"):
        a.load_eval(tmp_path, "R1", 6)
    f.write_text("\n".join(lines + ["6,204,29,1,300"]) + "\n")
    with pytest.raises(ValueError, match="others"):
        a.load_eval(tmp_path, "R1", 6)
    f.write_text("\n".join(lines[:-1] + ["0" + lines[-1][1:]]) + "\n")
    with pytest.raises(ValueError, match="task 0"):
        a.load_eval(tmp_path, "R1", 6)


def test_clean_run_and_report(tmp_path, capsys):
    good = episodes(40)
    make_run(tmp_path, lambda p, t, s, i: int(p[0] == "N" and (t, s, i) in good), sessions=2)
    p = a.check_provenance(tmp_path)
    assert p == {"records": 2 * 3 + 2 * sum(len(a.run_policies(t)) for t in a.TASKS), "commit": "abc"}
    assert a.main(["--root", str(tmp_path), "--gate", "harness/session2"]) == 0
    out = tmp_path / "r.txt"
    assert a.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert text in capsys.readouterr().out
    lines = text.splitlines()
    assert lines[1] == ("harness: session1 12/12, session2 12/12 (ok; at least 11 each); sessions stopped by a failed "
                        "harness, without episodes: none")
    assert lines[2].startswith("task 0: base 0/160, N0 40/160, N1 40/160, N2 40/160, R0 0/160")
    assert lines[4] == "task 7: base 0/152, N0 0/152, N1 0/152, N2 0/152, R0 0/152, R1 0/152, R2 0/152"
    assert "N against base: changes (helps) (p < 0.05 in at least 2 of 3 seeds, one sign)" in lines
    assert "reported, R0 against N0: 0 against 40 of 472, two-sided p 1.82e-12" in lines
    assert lines[-2] == "reported, pooled counts: N 120, R 0, base x 3 0"
    assert json.loads(Path(str(out) + ".json").read_text())["tests"]["N against base"]["direction"] == "helps"
    row = next(s for s in a.analyse(tmp_path)["states"] if (s["task"], s["state"]) == (0, 30))
    assert row["N0"][0] == 2 and row["base"][0] == 0            # episodes(40) = seeds 200-201 on task 0's 20 states


def test_adapter_paths_are_the_protocols():
    assert a.adapter_path(6, "N2") == "step10/runs/N6_2/adapter" and a.adapter_path(0, "R1") == "step12/runs/R0_1/adapter"


def test_exactly_two_seeds_hurting_is_a_change(tmp_path):
    def f(p, t, s, i):
        if p == "base":
            return int((t, s, i) in episodes(20))
        return int(p == "N2" and (t, s, i) in episodes(20))         # N0, N1 lose all 20 (p 2^-19); N2 equal
    r = a.analyse(make_run(tmp_path, f))
    per = r["tests"]["N against base"]["per_seed"]
    assert [per[s]["significant"] for s in "012"] == [True, True, False]
    assert r["tests"]["N against base"]["direction"] == "hurts"


def test_one_dirty_flag_among_clean_ones_is_refused(tmp_path):
    make_run(tmp_path)
    p = tmp_path / "eval" / "N1" / "task_0" / "part0" / "provenance.jsonl"
    rec = json.loads(p.read_text())
    rec.update(git_dirty_noise_reachability=False, git_dirty_sfbc=True)
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        a.check_provenance(tmp_path)


def test_packages_in_another_key_order_are_one_set(tmp_path):
    make_run(tmp_path)
    for k, f in enumerate(sorted(tmp_path.rglob("provenance.jsonl"))):
        rec = json.loads(f.read_text())
        rec["packages"] = {"a": 1, "torch": "2.11.0"} if k % 2 else {"torch": "2.11.0", "a": 1}
        f.write_text(json.dumps(rec) + "\n")
    a.check_provenance(tmp_path)


def test_pooled_counts_with_a_base_success(tmp_path):
    r = a.analyse(make_run(tmp_path, lambda p, t, s, i: int((t, s, i) in episodes(3))))
    pooled = r["reported"]["pooled counts (no pooled p: the seeds share their episodes)"]
    assert pooled == {"N": 9, "R": 9, "base x 3": 9}


def test_gate_failure_exit_code(tmp_path, capsys):
    make_run(tmp_path, sessions=1, harness_change={(1, 0): {(16, 8): (0, 520), (17, 8): (0, 520)}})
    assert a.main(["--root", str(tmp_path), "--gate", "harness/session1"]) == 1
    assert "gate harness/session1 FAILED: harness harness/session1: 10 of 12" in capsys.readouterr().out


def test_pending_and_failed_harness_folders_are_not_read(tmp_path):
    make_run(tmp_path, sessions=2)
    import shutil
    shutil.copytree(tmp_path / "harness" / "session2", tmp_path / "harness_pending" / "session3")
    (tmp_path / "harness_pending" / "session3" / "task_7" / "per_episode_lora.csv").unlink()   # a session cut short
    shutil.copytree(tmp_path / "harness" / "session2", tmp_path / "harness_failed" / "session4")
    edit(tmp_path / "harness_failed" / "session4" / "task_0" / "provenance.jsonl", git_dirty=True)
    r = a.analyse(tmp_path)
    assert r["harness"]["passes"] and list(r["harness"]["sessions"]) == ["session1", "session2"]
    assert r["harness"]["failed_sessions"] == ["session4"]
    with pytest.raises(ValueError, match="harness has"):
        a.gate(tmp_path, "harness_pending/session3")


def test_gate_refuses_other_package_versions_and_reports_unreadable_harness(tmp_path, capsys):
    import shutil
    make_run(tmp_path, sessions=1)
    shutil.copytree(tmp_path / "harness" / "session1", tmp_path / "harness_pending" / "session2")
    assert a.gate(tmp_path, "harness_pending/session2") is None
    for f in (tmp_path / "harness_pending" / "session2").rglob("provenance.jsonl"):
        edit(f, packages={"torch": "2.11.0+cu128"})
    assert "package versions" in a.gate(tmp_path, "harness_pending/session2")
    assert a.main(["--root", str(tmp_path), "--gate", "harness_pending/session2"]) == 1
    (tmp_path / "harness_pending" / "session2" / "task_6" / "per_episode_lora.csv").unlink()
    assert a.main(["--root", str(tmp_path), "--gate", "harness_pending/session2"]) == 2
    assert "could not be read" in capsys.readouterr().out
    first = tmp_path / "first"
    make_run(first, sessions=1)
    shutil.move(str(first / "harness" / "session1"), str(first / "harness_pending" / "session1"))
    assert a.gate(first, "harness_pending/session1") is None            # the first session sets the versions


def test_per_task_counts_have_exact_intervals(tmp_path):
    r = a.analyse(make_run(tmp_path, lambda p, t, s, i: int((t, s, i) in episodes(16))))
    k, n, ci = r["per_task"]["0"]["base"]
    assert (k, n) == (16, 160) and ci[0] < 0.1 < ci[1] and list(ci) == list(a.clopper_pearson(16, 160))
