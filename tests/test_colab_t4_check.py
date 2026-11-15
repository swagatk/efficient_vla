"""Colab T4 reproduction check (noise_reachability/analyze_colab_t4_check.py): known answers, the 11-of-12 rule's
edges, refusals, and the real step-10 files."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "noise_reachability"))
import analyze_colab_t4_check as c  # noqa: E402

T4 = ["Tesla T4"]


def write_rows(path, task, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success", "steps"])
        for (seed, state), (succ, steps) in rows.items():
            w.writerow([task, seed, state, succ, steps])


def outcome(t, seed):
    return (1, 300 + seed + t) if (seed + t) % 2 else (0, 520)


def make_step10(root):
    for t in c.TASKS:
        for pol in ("base", "N0"):
            rows = {(s, c.HARNESS_STATE[t]): outcome(t, s) for s in c.HARNESS_SEEDS}
            rows[(99, 1)] = (0, 520)                       # other episodes of the same part are ignored
            write_rows(root / f"task_{t}" / "eval" / pol / "part0" / f"per_episode_{'base' if pol == 'base' else 'lora'}.csv",
                       t, rows)
    prov = {"packages": {"torch": {"version": "2.11.0"}}}
    (root / "task_0" / "eval" / "base" / "part0" / "provenance.jsonl").write_text(json.dumps(prov) + "\n")


def make_run(root, change=None, gpus=T4, settings=None, packages=None):
    """Colab run equal to step 10 except the (policy, task, seed) -> outcome in ``change``."""
    change = change or {}
    for t in c.TASKS:
        for pol, (flag, suffix) in c.POLICIES.items():
            rows = {(s, c.HARNESS_STATE[t]): change.get((pol, t, s), outcome(t, s)) for s in c.HARNESS_SEEDS}
            d = root / f"task_{t}" / pol
            write_rows(d / f"per_episode_{suffix}.csv", t, rows)
            st = {"stage": "eval", "task_id": t, "eval_policies": flag, "n_action_steps": 10, "max_steps": 520,
                  "eval_init_states": str(c.HARNESS_STATE[t]), "eval_seeds": "16-19",
                  "policy_path": "/x/snapshots/a24f1f380461b665158d52acd140152c79e8b36d",
                  "lora_dir": f"/content/drive/in/N{t}_0/adapter"}
            st.update((settings or {}).get((pol, t), {}))
            rec = {"git_commit": "abc", "git_dirty": False, "git_dirty_sfbc": False, "gpus": gpus, "settings": st,
                   "packages": packages or {"torch": {"version": "2.11.0"}}}
            (d / "provenance.jsonl").write_text(json.dumps(rec) + "\n")


@pytest.fixture
def step10(tmp_path):
    make_step10(tmp_path / "step10")
    return tmp_path / "step10"


def report(run, step10):
    return {"compare": c.compare(run, step10), "provenance": c.provenance(run, step10)}


def test_identical_run_passes_12_of_12(tmp_path, step10):
    make_run(tmp_path / "run")
    r = report(tmp_path / "run", step10)
    assert all(v["same"] == 12 and v["passes"] for v in r["compare"].values())
    assert r["provenance"]["same_packages_as_step10"] and r["provenance"]["gpus"] == T4


def test_rule_edges_11_passes_10_fails(tmp_path, step10):
    one = {("base", 0, 16): (1, 1)}                                       # steps differ: not the same episode
    make_run(tmp_path / "a", one)
    cmp_ = c.compare(tmp_path / "a", step10)
    assert cmp_["base"]["same"] == 11 and cmp_["base"]["passes"] and cmp_["N0"]["same"] == 12
    assert cmp_["base"]["per_task"][0]["differ"] == [[[16, 8], [1, 1], list(outcome(0, 16))]]
    two = {("N0", 6, 17): outcome(6, 17)[::-1], ("N0", 7, 18): (outcome(7, 18)[0], outcome(7, 18)[1] + 1)}
    make_run(tmp_path / "b", two)
    cmp_ = c.compare(tmp_path / "b", step10)
    assert cmp_["N0"]["same"] == 10 and not cmp_["N0"]["passes"] and cmp_["base"]["passes"]


def test_success_alone_differing_counts(tmp_path, step10):
    s = outcome(7, 19)
    make_run(tmp_path / "run", {("N0", 7, 19): (1 - s[0], s[1])})
    assert c.compare(tmp_path / "run", step10)["N0"]["same"] == 11


def test_refuses_missing_extra_and_duplicate_episodes(tmp_path, step10):
    make_run(tmp_path / "run")
    path = tmp_path / "run" / "task_6" / "base" / "per_episode_base.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="finish the run"):
        c.compare(tmp_path / "run", step10)
    path.write_text("\n".join(lines + ["6,20,8,0,520"]) + "\n")
    with pytest.raises(ValueError, match="finish the run"):
        c.compare(tmp_path / "run", step10)
    path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="twice"):
        c.compare(tmp_path / "run", step10)


@pytest.mark.parametrize("bad", [
    {("base", 0): {"n_action_steps": 1}}, {("N0", 7): {"eval_seeds": "16-20"}}, {("N0", 6): {"eval_init_states": "9"}},
    {("N0", 0): {"lora_dir": "/content/drive/in/N0_1/adapter"}}, {("base", 6): {"eval_policies": "lora"}},
    {("base", 7): {"policy_path": "/x/snapshots/other"}}, {("N0", 0): {"max_steps": 600}}])
def test_refuses_wrong_settings(tmp_path, step10, bad):
    make_run(tmp_path / "run", settings=bad)
    with pytest.raises(ValueError):
        c.provenance(tmp_path / "run", step10)


def test_refuses_other_gpus_dirty_trees_and_two_commits(tmp_path, step10):
    make_run(tmp_path / "a", gpus=["NVIDIA L4"])
    with pytest.raises(ValueError, match="T4"):
        c.provenance(tmp_path / "a", step10)
    make_run(tmp_path / "b")
    p = tmp_path / "b" / "task_7" / "N0" / "provenance.jsonl"
    rec = json.loads(p.read_text())
    p.write_text(json.dumps(dict(rec, git_dirty=True)) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        c.provenance(tmp_path / "b", step10)
    p.write_text(json.dumps(dict(rec, git_commit="def")) + "\n")
    with pytest.raises(ValueError, match="commits"):
        c.provenance(tmp_path / "b", step10)


def test_reports_different_packages(tmp_path, step10):
    make_run(tmp_path / "run", packages={"torch": {"version": "2.12.0"}})
    assert not c.provenance(tmp_path / "run", step10)["same_packages_as_step10"]


def test_step10_conflicting_duplicates_refused(tmp_path, step10):
    extra = step10 / "task_0" / "eval" / "N0" / "part1" / "per_episode_lora.csv"
    write_rows(extra, 0, {(16, 8): (1 - outcome(0, 16)[0], 1)})
    with pytest.raises(ValueError, match="different outcomes"):
        c.step10_rows(0, "N0", step10)


def test_real_step10_files_have_every_harness_episode():
    """The committed Kaggle T4 episodes the Colab run is compared with: 4 per task and policy, known outcomes."""
    for t in c.TASKS:
        for pol in c.POLICIES:
            assert len(c.step10_rows(t, pol)) == 4
    # step 12's harness reproduced these N0 rows exactly; their successes (step12 session1_checks.txt): 4, 1, 4
    assert [sum(v[0] for v in c.step10_rows(t, "N0").values()) for t in c.TASKS] == [4, 1, 4]


def test_real_step10_rows_pass_against_themselves(tmp_path):
    """Known answer on real data: a run equal to step 10's own rows gives 12 of 12 for both policies."""
    for t in c.TASKS:
        for pol, (_, suffix) in c.POLICIES.items():
            write_rows(tmp_path / f"task_{t}" / pol / f"per_episode_{suffix}.csv", t, c.step10_rows(t, pol))
    assert all(v["same"] == 12 for v in c.compare(tmp_path).values())


def test_counts_and_successes_reported(tmp_path, step10):
    s = outcome(6, 16)
    make_run(tmp_path / "run", {("base", 6, 16): (1 - s[0], s[1])})
    cmp_ = c.compare(tmp_path / "run", step10)
    assert cmp_["base"]["episodes"] == 12 and cmp_["N0"]["episodes"] == 12
    x = cmp_["base"]["per_task"][6]
    old = sum(outcome(6, k)[0] for k in c.HARNESS_SEEDS)
    assert x["successes_step10"] == old and x["successes_now"] == old + (1 if s[0] == 0 else -1)


def test_one_failing_policy_fails_the_check(tmp_path, step10):
    change = {("N0", 0, 16): (9, 9), ("N0", 0, 17): (9, 9)}
    make_run(tmp_path / "run", change)
    r = c.report(tmp_path / "run", step10)
    assert r["compare"]["base"]["passes"] and not r["compare"]["N0"]["passes"] and not r["passes"]
    make_run(tmp_path / "ok")
    assert c.report(tmp_path / "ok", step10)["passes"]


def test_refuses_a_mix_of_t4_and_other_gpus(tmp_path, step10):
    make_run(tmp_path / "run", gpus=["Tesla T4", "NVIDIA L4"])
    with pytest.raises(ValueError, match="T4"):
        c.provenance(tmp_path / "run", step10)


def test_same_packages_in_another_key_order_are_one_set(tmp_path, step10):
    make_run(tmp_path / "run", packages={"a": 1, "torch": {"version": "2.11.0"}})
    p = tmp_path / "run" / "task_7" / "N0" / "provenance.jsonl"
    rec = json.loads(p.read_text())
    p.write_text(json.dumps(dict(rec, packages={"torch": {"version": "2.11.0"}, "a": 1})) + "\n")
    c.provenance(tmp_path / "run", step10)


@pytest.mark.parametrize("change", [{"git_dirty_sfbc": True}, {"git_dirty": None}, {"git_dirty_sfbc": None}])
def test_refuses_missing_or_dirty_flags(tmp_path, step10, change):
    make_run(tmp_path / "run")
    p = tmp_path / "run" / "task_6" / "base" / "provenance.jsonl"
    rec = dict(json.loads(p.read_text()), **change)
    p.write_text(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
    with pytest.raises(ValueError, match="dirty"):
        c.provenance(tmp_path / "run", step10)
