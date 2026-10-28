import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import analyze_noise_signal as ans  # noqa: E402
from noise_sac import ChunkReplay  # noqa: E402


def brute_auc(s, l):
    pos, neg = [x for x, y in zip(s, l) if y], [x for x, y in zip(s, l) if not y]
    return np.mean([(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)])


def test_auc_matches_pairwise_count_with_ties_and_edges():
    rng = np.random.default_rng(0)
    s, l = rng.normal(size=200).round(1), rng.random(200) < 0.4
    assert ans.auc(s, l) == pytest.approx(brute_auc(s, l), abs=1e-12)
    assert ans.auc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5
    assert ans.auc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0
    assert ans.auc([3, 2, 1, 0], [0, 0, 1, 1]) == 0.0
    with pytest.raises(ValueError):
        ans.auc([1, 2], [1, 1])


def test_mc_returns_count_decisions_left_and_zero_for_failures():
    r = ans.mc_returns([3, 2, 1], [1, 0, 1], 0.5)
    assert r.shape == (6, 1)
    assert r[:, 0].tolist() == [0.25, 0.5, 1.0, 0.0, 0.0, 1.0]


def test_held_out_is_every_fifth_episode_from_index_four():
    assert ans.is_held_out(np.arange(11)).tolist() == [False] * 4 + [True] + [False] * 4 + [True, False]


def test_bootstrap_resamples_whole_groups_and_is_deterministic():
    # Group g has 5 transitions, all with the same score and label: a resample by transition would mix them,
    # a resample by group keeps each group's 5 rows together.
    groups = np.repeat(np.arange(40), 5)
    rng = np.random.default_rng(3)
    lab = np.repeat(rng.random(40) < 0.5, 5)
    score = np.repeat(rng.normal(size=40), 5) + lab * 0.8
    lo, hi = ans.bootstrap_auc(score, lab, groups, 500, seed=1)
    point = ans.auc(score, lab)
    assert lo < point < hi
    assert (lo, hi) == ans.bootstrap_auc(score, lab, groups, 500, seed=1)
    # Identical groups: every resample gives the same AUC, so the interval is a point.
    same = ans.bootstrap_auc(np.tile([0.0, 1.0], 20), np.tile([0, 1], 20), np.repeat(np.arange(20), 2), 200, seed=0)
    assert same == (1.0, 1.0)


def test_bootstrap_refuses_when_most_resamples_have_one_class():
    with pytest.raises(ValueError):
        ans.bootstrap_auc([0.0, 1.0] + [0.5] * 98, [1] + [0] * 99, np.arange(100), 200, seed=0)


@pytest.mark.parametrize("a, ci, ctrl, want", [
    (0.55, (0.5001, 0.6), (0.45, 0.55), "supported"),            # at the threshold, lower end just above
    (0.5499, (0.51, 0.6), (0.45, 0.55), "not supported"),        # just below the threshold
    (0.60, (0.5, 0.7), (0.45, 0.55), "not supported"),           # lower end exactly at chance (strict)
    (0.60, (0.52, 0.7), (0.5, 0.55), "supported"),               # control interval touching 0.5 includes it
    (0.60, (0.52, 0.7), (0.4, 0.5), "supported"),
    (0.60, (0.52, 0.7), (0.5001, 0.6), "not read (control biased)"),
    (0.60, (0.52, 0.7), (0.4, 0.4999), "not read (control biased)"),
])
def test_decision_rule_edges(a, ci, ctrl, want):
    assert ans.decide(a, ci, ctrl) == want


# --- collection folders -------------------------------------------------------------------------------------

SETTINGS = dict(task_id=7, policy_path="p", n_action_steps=10, rename_map={}, max_steps=520, train_init_states="10-49",
                num_episodes=6, warmup_episodes=6, updates_per_decision=0, noise_scale=5.0, gamma=0.5,
                train_hard_reset=0)


def write_stream(root, seed, decisions, success, settings=None, parts=((0, 3), (3, 6)), fd=3, ad=2,
                 commit="c0ffee", episodes_done=None, noise_scale=None):
    """A collection stream as train_noise_sac.py writes it: replay parts, checkpoint.pt listing them,
    train_log.csv, provenance.jsonl."""
    import torch

    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rb = ChunkReplay(sum(decisions), fd, ad)
    rb.feats[:] = rng.normal(size=rb.feats.shape)
    # noise_scale: noise drawn like the collection's (N(0, 1) / noise_scale, clipped); else uniform
    rb.u[:] = (rng.uniform(-0.5, 0.5, size=rb.u.shape) if noise_scale is None
               else np.clip(rng.normal(size=rb.u.shape) / noise_scale, -0.999, 0.999))
    ends = np.cumsum(decisions) - 1
    rb.terminal[ends, 0] = success
    rb.reward[ends, 0] = success
    rb.size = sum(decisions)
    edge = np.r_[0, np.cumsum(decisions)]
    for a, b in parts:
        size = rb.size
        rb.size = int(edge[b])
        rb.save_part(root / f"replay_part_{a:05d}-{b - 1:05d}.npz", int(edge[a]))
        rb.size = size
    torch.save({"episodes_done": len(decisions) if episodes_done is None else episodes_done,
                "replay_parts": [f"replay_part_{a:05d}-{b - 1:05d}.npz" for a, b in parts]}, root / "checkpoint.pt")
    with open(root / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "success", "decisions"])
        w.writeheader()
        w.writerows({"episode": e, "success": s, "decisions": d} for e, (s, d) in enumerate(zip(success, decisions)))
    rec = {"settings": dict(SETTINGS if settings is None else settings, seed=seed), "git_commit": commit,
           "git_dirty": False}
    (root / "provenance.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    return rb


DEC, SUC = [2, 3, 1, 2, 2, 1], [1, 0, 1, 0, 1, 1]


def test_load_streams_joins_streams_with_returns_groups_and_split(tmp_path):
    rb0 = write_stream(tmp_path / "s0", 0, DEC, SUC)
    write_stream(tmp_path / "s1", 1, DEC[::-1], SUC[::-1])
    d = ans.load_streams([tmp_path / "s0", tmp_path / "s1"])
    assert len(d["u"]) == 2 * sum(DEC)
    assert np.array_equal(d["u"][: sum(DEC)], rb0.u[: sum(DEC)])
    assert np.array_equal(d["feats"][: sum(DEC)], rb0.feats[: sum(DEC)].astype(np.float32))
    assert d["ret"][:3, 0].tolist() == [0.5, 1.0, 0.0]           # episode 0 succeeded after 2 decisions
    assert d["success"][:6].tolist() == [1, 1, 0, 0, 0, 1]
    assert d["group"][:3].tolist() == [0, 0, 1] and d["group"][sum(DEC)] == 100_000
    assert d["held"][: sum(DEC)].tolist() == [False] * 8 + [True, True] + [False]   # episode 4 of stream 0
    assert d["streams"] == {0: (6, 4), 1: (6, 4)}


def test_load_refuses_mismatched_or_incomplete_streams(tmp_path):
    write_stream(tmp_path / "a", 0, DEC, SUC)
    write_stream(tmp_path / "same_seed", 0, DEC, SUC)
    with pytest.raises(ValueError, match="share a seed"):
        ans.load_streams([tmp_path / "a", tmp_path / "same_seed"])
    write_stream(tmp_path / "other", 1, DEC, SUC, settings=dict(SETTINGS, noise_scale=2.5))
    with pytest.raises(ValueError, match="differ"):
        ans.load_streams([tmp_path / "a", tmp_path / "other"])
    write_stream(tmp_path / "trained", 2, DEC, SUC, settings=dict(SETTINGS, updates_per_decision=8))
    with pytest.raises(ValueError, match="not a collection run"):
        ans.load_stream(tmp_path / "trained")
    write_stream(tmp_path / "warm", 3, DEC, SUC, settings=dict(SETTINGS, warmup_episodes=5))
    with pytest.raises(ValueError, match="not a collection run"):
        ans.load_stream(tmp_path / "warm")
    write_stream(tmp_path / "short", 4, DEC, SUC, parts=((0, 3),))
    with pytest.raises(ValueError, match="replay holds"):
        ans.load_stream(tmp_path / "short")
    write_stream(tmp_path / "gap", 5, DEC, SUC, parts=((0, 2), (3, 6)))
    with pytest.raises(ValueError):
        ans.load_stream(tmp_path / "gap")
    write_stream(tmp_path / "few", 6, DEC, SUC, settings=dict(SETTINGS, num_episodes=7, warmup_episodes=7))
    with pytest.raises(ValueError, match="train_log.csv has"):
        ans.load_stream(tmp_path / "few")


def test_load_refuses_terminals_that_disagree_with_the_log(tmp_path):
    write_stream(tmp_path / "s", 0, DEC, [0, 0, 1, 0, 1, 1])
    log = (tmp_path / "s" / "train_log.csv").read_text().replace("0,0,2", "0,1,2")
    (tmp_path / "s" / "train_log.csv").write_text(log)
    with pytest.raises(ValueError, match="terminal"):
        ans.load_stream(tmp_path / "s")


def test_load_refuses_settings_that_change_within_a_stream(tmp_path):
    write_stream(tmp_path / "s", 0, DEC, SUC)
    p = tmp_path / "s" / "provenance.jsonl"
    p.write_text(p.read_text() + json.dumps({"settings": dict(SETTINGS, seed=0, max_steps=300)}) + "\n")
    with pytest.raises(ValueError, match="disagree"):
        ans.load_stream(tmp_path / "s")


# --- end to end on synthetic data ---------------------------------------------------------------------------


def synthetic(noise_effect, n_episodes=400, n_dec=6, fd=4, ad=3, seed=0):
    """Success depends on the state (first feature) and, if noise_effect, on the noise's first entry."""
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_episodes * n_dec, fd)).astype(np.float32)
    u = np.clip(rng.normal(size=(n_episodes * n_dec, ad)) / 5.0, -0.999, 0.999).astype(np.float32)
    ep = np.repeat(np.arange(n_episodes), n_dec)
    logit = feats[:, 0] + noise_effect * 25 * u[:, 0]
    succ_ep = np.array([rng.random() < 1 / (1 + np.exp(-logit[ep == e].mean() * 3)) for e in range(n_episodes)])
    return {"feats": feats, "u": u, "success": succ_ep[ep].astype(int),
            "ret": ans.mc_returns([n_dec] * n_episodes, succ_ep, 0.9), "group": ep, "held": ans.is_held_out(ep),
            "cfg": {"noise_scale": 5.0}}


@pytest.mark.slow
def test_end_to_end_finds_a_noise_effect_and_not_one_that_is_absent():
    import torch

    torch.set_num_threads(4)
    data = synthetic(1.0)
    # The best any score can do here: the true noise entry that drives success (0.667 on this data).
    oracle = ans.auc(data["u"][data["held"], 0], data["success"][data["held"]])
    with_effect = ans.analyse(data, n_critics=2, n_boot=300, hidden=32, n_updates=1500)
    assert with_effect["critic"]["action_auc"] > max(0.55, oracle - 0.12)
    assert with_effect["decision"] == "supported"
    h = data["held"]
    best_q = ans.auc(data["feats"][h, 0] + 25 * data["u"][h, 0], data["success"][h])   # the true logit
    assert with_effect["critic"]["success_auc"] > best_q - 0.12
    nul = synthetic(0.0, seed=1)
    without = ans.analyse(nul, n_critics=2, n_boot=300, hidden=32, n_updates=1500)
    assert without["decision"] == "not supported"
    assert abs(without["critic"]["action_auc"] - 0.5) < 0.08
    # The state value is still learned: close to the true state's own AUC (0.618 on this data).
    assert without["critic"]["success_auc"] > ans.auc(nul["feats"][nul["held"], 0], nul["success"][nul["held"]]) - 0.08
    # The shuffled control learns no noise effect even where there is one.
    assert abs(with_effect["control"]["action_auc"] - 0.5) < 0.08


def test_scores_noise_part_is_zero_for_a_critic_that_ignores_the_noise():
    import torch

    class StateOnly(torch.nn.Module):
        def forward(self, f, a):
            v = f.sum(1, keepdim=True)
            return v, v + 1

    f = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    u = np.zeros((10, 2), dtype=np.float32)
    q, part = ans.scores([StateOnly()], f, u, 5.0)
    assert np.allclose(q, f.sum(1)) and np.allclose(part, 0.0)


def test_scores_compare_against_noises_drawn_at_the_collected_scale():
    import torch

    class NoiseOnly(torch.nn.Module):
        def forward(self, f, a):
            v = (a ** 2).sum(1, keepdim=True)
            return v, v

    f = np.zeros((20000, 1), dtype=np.float32)
    u = np.zeros((20000, 2), dtype=np.float32)
    _, part = ans.scores([NoiseOnly()], f, u, 5.0, n_alt=1, chunk=50000)
    assert -part.mean() == pytest.approx(2 / 25, rel=0.05)           # E|a_k|^2 = 2 x (1/5)^2


def test_auc_with_a_single_positive_or_negative():
    assert ans.auc([0.1, 0.5, 0.9], [0, 1, 0]) == 0.5
    assert ans.auc([0.1, 0.5, 0.9], [1, 1, 0]) == 0.0


def test_bootstrap_matches_a_direct_implementation():
    # Same draws as bootstrap_auc: groups resampled with rng.integers over all groups, 2.5% and 97.5% quantiles.
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 3)
    lab = np.repeat(rng.random(30) < 0.5, 3)
    score = rng.normal(size=90) + lab
    ref, draw = [], np.random.default_rng(7)
    for _ in range(400):
        g = draw.integers(0, 30, 30)
        pick = np.concatenate([np.flatnonzero(groups == k) for k in g])
        if 0 < lab[pick].sum() < len(pick):
            ref.append(brute_auc(score[pick], lab[pick]))
    want = (np.quantile(ref, 0.025), np.quantile(ref, 0.975))
    assert ans.bootstrap_auc(score, lab, groups, 400, seed=7) == pytest.approx(want, abs=1e-12)


def test_load_refuses_when_only_the_first_or_second_provenance_record_differs(tmp_path):
    for name, which in (("first", 0), ("second", 1)):
        write_stream(tmp_path / name, 0, DEC, SUC)
        p = tmp_path / name / "provenance.jsonl"
        recs = p.read_text().splitlines()
        recs[which] = json.dumps({"settings": dict(SETTINGS, seed=0, max_steps=300)})
        p.write_text("\n".join(recs) + "\n")
        with pytest.raises(ValueError, match="disagree"):
            ans.load_stream(tmp_path / name)


def test_scores_clip_the_comparison_noises_like_the_collection():
    import torch

    class NoiseOnly(torch.nn.Module):
        def forward(self, f, a):
            v = (a ** 2).sum(1, keepdim=True)
            return v, v

    f = np.zeros((2000, 1), dtype=np.float32)
    u = np.zeros((2000, 2), dtype=np.float32)
    _, part = ans.scores([NoiseOnly()], f, u, 0.001, n_alt=1)       # N(0, 1) / 0.001 clipped to +-0.999
    assert -part.mean() == pytest.approx(2 * 0.999 ** 2, rel=1e-3)


def test_load_uses_the_parts_the_checkpoint_lists_and_refuses_an_unfinished_checkpoint(tmp_path):
    write_stream(tmp_path / "s", 0, DEC, SUC)
    (tmp_path / "s" / "replay_part_00003-00005.npz.tmp.npz").write_bytes(b"leftover of an interrupted save")
    assert len(ans.load_stream(tmp_path / "s")["u"]) == sum(DEC)
    write_stream(tmp_path / "unfinished", 1, DEC, SUC, episodes_done=5)
    with pytest.raises(ValueError, match="saved 5 of 6"):
        ans.load_stream(tmp_path / "unfinished")


def test_load_streams_collects_commits_and_dirty_flags(tmp_path):
    write_stream(tmp_path / "a", 0, DEC, SUC, commit="c1")
    write_stream(tmp_path / "b", 1, DEC, SUC, commit="c2")
    d = ans.load_streams([tmp_path / "a", tmp_path / "b"])
    assert d["git_commits"] == ["c1", "c2"] and d["git_dirty"] is False and d["seeds"] == [0, 1]
    p = tmp_path / "b" / "provenance.jsonl"
    p.write_text(p.read_text().replace('"git_dirty": false', '"git_dirty": true', 1))
    assert ans.load_streams([tmp_path / "a", tmp_path / "b"])["git_dirty"] is True
    q = tmp_path / "a" / "provenance.jsonl"
    rec = json.loads(q.read_text().splitlines()[0])
    q.write_text(json.dumps(dict(rec, git_dirty_noise_reachability=True)) + "\n")
    assert ans.load_stream(tmp_path / "a")["cfg"]["git_dirty"] is True


def protocol_data(**change):
    cfg = dict(ans.PROTOCOL, policy_path=f"/hf/snapshots/{ans.PROTOCOL_REVISION}")
    d = {"cfg": cfg, "seeds": [0, 1, 2, 3], "git_commits": ["abc"], "git_dirty": False}
    for k, v in change.items():
        (d["cfg"] if k in cfg else d)[k] = v
    return d


def test_check_protocol_accepts_the_readme_run_and_refuses_any_difference():
    assert ans.PROTOCOL == {"task_id": 7, "noise_scale": 5.0, "num_episodes": 250, "n_action_steps": 10,
                            "train_init_states": "10-49", "max_steps": 520, "train_hard_reset": 0}
    ans.check_protocol(protocol_data())
    for change in ({"task_id": 0}, {"noise_scale": 2.5}, {"num_episodes": 249}, {"n_action_steps": 1},
                   {"train_init_states": "0-49"}, {"max_steps": 300}, {"train_hard_reset": 1},
                   {"seeds": [0, 1, 2]}, {"seeds": [0, 1, 2, 4]}, {"policy_path": "lerobot/smolvla_libero"},
                   {"git_commits": ["a", "b"]}, {"git_commits": ["None"]}, {"git_dirty": True}):
        with pytest.raises(ValueError):
            ans.check_protocol(protocol_data(**change))


def test_analysis_defaults_are_the_protocol():
    import inspect

    a = inspect.signature(ans.analyse).parameters
    assert (a["n_critics"].default, a["n_boot"].default, a["hidden"].default, a["n_updates"].default) == (3, 2000, 256, None)
    f = inspect.signature(ans.fit_critic).parameters
    assert (f["hidden"].default, f["layers"].default, f["lr"].default, f["batch"].default) == (256, 3, 3e-4, 256)
    # n_updates=None: one update per training transition.
    d = synthetic(0.0, n_episodes=40, n_dec=2, seed=3)
    out = ans.analyse(d, n_critics=1, n_boot=50, hidden=8)
    assert out["n_updates"] == int((~d["held"]).sum()) == 64
