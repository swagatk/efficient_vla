import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import analyze_step4 as a4  # noqa: E402

SEEDS = list(range(16, 32))
SYS = [2, 5, 6, 7, 8]
REACH = [0, 1, 3, 4, 9]


def eps(wins, seeds=SEEDS):
    """Task-0 episodes; ``wins`` is the set of (seed, init_state) that succeed."""
    return {(0, s, i): int((s, i) in wins) for s in seeds for i in range(10)}


def first(n, states):
    """The first ``n`` (seed, init_state) pairs over ``states``, seed-major."""
    return set(sorted((s, i) for s in SEEDS for i in states)[:n])


BASE_REACH = first(40, REACH)                       # base: 0/80 systematic, 40/80 reachable


def arm(n_sys, reach=BASE_REACH):
    return eps(first(n_sys, SYS) | reach)


def arms(a=(0, 0, 0), c=(0, 0, 0), b=0):
    return {"A": {k: arm(n) for k, n in enumerate(a)}, "C": {k: arm(n) for k, n in enumerate(c)}, "B": arm(b)}


BASE = eps(BASE_REACH)


def test_constants_match_readme():
    assert a4.SYSTEMATIC_STATES == (2, 5, 6, 7, 8) and a4.EVAL_SEEDS == tuple(range(16, 32))
    assert (a4.MIN_NET_SYSTEMATIC, a4.MAX_NET_LOSS_REACHABLE, a4.MIN_SEEDS, a4.ALPHA) == (8, 4, 2, 0.05)
    assert (a4.HARNESS_REFERENCE, a4.HARNESS_TOLERANCE, a4.HARNESS_SEEDS) == (7, 6, (0, 1, 2))


def test_split_counts():
    s, r = a4.split(BASE)
    assert len(s) == 80 and len(r) == 80 and sum(s.values()) == 0 and sum(r.values()) == 40
    assert all(k[2] in SYS for k in s) and all(k[2] in REACH for k in r)


def test_paired_known_answer():
    ref = {(0, 16, i): int(i < 3) for i in range(10)}          # wins 0, 1, 2
    run = {(0, 16, i): int(1 <= i < 7) for i in range(10)}     # wins 1-6: run-only 4, ref-only 1
    r = a4.paired(ref, run)
    assert (r["ref"], r["run"], r["ref_only"], r["run_only"], r["net"], r["n"]) == (3, 6, 1, 4, 3, 10)
    assert r["p"] == pytest.approx(2 * (1 + 5) / 2 ** 5)
    with pytest.raises(ValueError):
        a4.paired(ref, {(0, 17, 0): 1})


def test_systematic_threshold_edges():
    assert not a4.arm_result(BASE, arm(7))["systematic"]["ok"]           # 7 net: below 10 points
    r = a4.arm_result(BASE, arm(8))
    assert r["systematic"]["ok"] and r["systematic"]["p"] == pytest.approx(2 / 2 ** 8) and r["meets"]
    # 8 net but not significant: base wins 4 systematic episodes the arm loses, arm wins 12
    base = eps(BASE_REACH | first(4, SYS))
    run = eps(BASE_REACH | (first(16, SYS) - first(4, SYS)))
    r = a4.arm_result(base, run)["systematic"]
    assert r["net"] == 8 and r["p"] >= 0.05 and not r["ok"]


def test_reachable_guard_edges():
    lose4 = BASE_REACH - set(sorted(BASE_REACH)[:4])
    lose5 = BASE_REACH - set(sorted(BASE_REACH)[:5])
    assert a4.arm_result(BASE, arm(8, lose4))["reachable"]["guard_ok"]        # -5 points exactly
    r = a4.arm_result(BASE, arm(8, lose5))
    assert not r["reachable"]["guard_ok"] and not r["meets"]
    # small net loss but significant: loses 6, wins 4 elsewhere -> net -2, p large -> ok
    swap = (BASE_REACH - set(sorted(BASE_REACH)[:6])) | set(sorted(set(first(80, REACH)) - BASE_REACH)[:4])
    assert a4.arm_result(BASE, arm(8, swap))["reachable"]["guard_ok"]


def test_guard_allows_a_one_way_loss_of_four():
    # all 4 discordant episodes go to the base: p = 0.125, so the loss is within both parts of the guard
    full = set(first(80, REACH))
    r = a4.arm_result(eps(full), eps(full - set(sorted(full)[:4])))["reachable"]
    assert r["net"] == -4 and r["p"] == pytest.approx(0.125) and r["guard_ok"]


def test_decision_main_claim_and_alternatives():
    out = a4.analyse(BASE, arms(c=(12, 12, 0)))
    assert out["decision"].startswith("main claim supported")
    out = a4.analyse(BASE, arms(a=(8, 8, 0), c=(12, 12, 12)))
    assert out["decision"].startswith("A meets")
    out = a4.analyse(BASE, arms(a=(4, 4, 4), c=(9, 9, 0)))            # C meets, C vs A net +5: p 0.0625
    assert out["decision"].startswith("consistent with the claim")
    out = a4.analyse(BASE, arms(c=(12, 0, 0), b=10))                   # C meets in only 1 of 3 seeds
    assert out["decision"].startswith("no LoRA arm") and out["B"]["meets"]
    assert a4.analyse(BASE, arms())["decision"].startswith("negative")


def test_c_ahead_of_a_needs_two_seeds():
    out = a4.analyse(BASE, arms(a=(0, 4, 4), c=(12, 9, 9)))
    assert [out["C_vs_A"][k]["ahead"] for k in range(3)] == [True, False, False]
    assert out["decision"].startswith("consistent with the claim")


def test_pooled_repeats_base_per_seed():
    out = a4.analyse(BASE, arms(c=(12, 6, 0)))
    s = out["pooled"]["C"]["systematic"]
    assert (s["n"], s["ref"], s["run"], s["net"]) == (240, 0, 18, 18)


def test_harness_check_tolerance():
    def h(n):
        wins = sorted((s, i) for s in (0, 1, 2) for i in range(10))[:n]
        return eps(set(wins), seeds=[0, 1, 2])
    assert a4.harness_check(h(1))["ok"] and a4.harness_check(h(13))["ok"]
    assert not a4.harness_check(h(0))["ok"] and not a4.harness_check(h(14))["ok"]
    assert a4.harness_check(h(3))["gap"] == -4 and a4.harness_check(h(10))["gap"] == 3
    out = a4.analyse(BASE, arms(c=(12, 12, 0)), harness=h(20))
    assert out["decision"].startswith("not read: the harness check failed")
    with pytest.raises(ValueError):
        a4.harness_check(eps(set()))                                  # seeds 16-31, not 0-2


def test_refuses_incomplete_or_mismatched():
    short = dict(BASE)
    short.pop((0, 31, 9))
    with pytest.raises(ValueError, match="finish the run"):
        a4.analyse(short, arms())
    bad = arms()
    bad["C"].pop(2)
    with pytest.raises(ValueError, match="same 3 training seeds"):
        a4.analyse(BASE, bad)
    bad = arms()
    bad["B"] = eps(set(), seeds=SEEDS[:-1])
    with pytest.raises(ValueError):
        a4.analyse(BASE, bad)


def write_csv(path, episodes):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "seed", "init_state", "success"])
        for (t, s, i), v in sorted(episodes.items()):
            w.writerow([t, s, i, v])
    return str(path)


def test_load_refuses_other_task_and_repeats(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("task_id,seed,init_state,success\n1,16,0,1\n")
    with pytest.raises(ValueError, match="episode of task 1, expected 0"):
        a4.load(p)
    p.write_text("task_id,seed,init_state,success\n0,16,0,1\n0,16,0,0\n")
    with pytest.raises(ValueError, match="twice"):
        a4.load(p)


def test_main_end_to_end(tmp_path):
    specs = []
    for name, e in [("A0", arm(0)), ("A1", arm(0)), ("A2", arm(0)), ("C0", arm(12)), ("C1", arm(12)),
                    ("C2", arm(0)), ("B", arm(3))]:
        specs.append(f"{name}={write_csv(tmp_path / f'{name}.csv', e)}")
    harness = eps(set(sorted((s, i) for s in (0, 1, 2) for i in range(10))[:7]), seeds=[0, 1, 2])
    out = tmp_path / "report.txt"
    a4.main(["--base", write_csv(tmp_path / "base.csv", BASE), "--arm", *specs,
             "--harness", write_csv(tmp_path / "h.csv", harness), "--out", str(out)])
    text = out.read_text()
    assert "decision: main claim supported" in text and "harness check: base 7/30" in text
    j = json.loads((tmp_path / "report.txt.json").read_text())
    assert j["decision"].startswith("main claim supported") and j["C"]["0"]["meets"] is True
    import hashlib
    base_csv = str(tmp_path / "base.csv")
    assert j["provenance"]["inputs"][base_csv] == hashlib.sha256(open(base_csv, "rb").read()).hexdigest()
    assert len(j["provenance"]["inputs"]) == 9 and j["provenance"]["git_commit"]
    with pytest.raises(ValueError, match="unknown arm"):
        a4.parse_arms([f"D0={tmp_path / 'B.csv'}"])


def test_report_counts_and_interval_text(tmp_path):
    out = a4.analyse(BASE, arms(c=(12, 12, 0)))
    assert out["base"] == {"systematic": 0, "reachable": 40}
    text = a4.format_report(out)
    assert "base 0/80 systematic, 40/80 reachable" in text
    lo, hi = out["C"][0]["systematic"]["diff_ci95"]
    assert f"diff {100 * lo:+.1f} to {100 * hi:+.1f}" in text and lo > 0
    path = tmp_path / "run=1.csv"                                     # '=' inside the path
    assert a4.parse_arms([f"B={write_csv(path, arm(0))}"])["B"] == arm(0)
