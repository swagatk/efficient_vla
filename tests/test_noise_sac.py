import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

from noise_sac import (  # noqa: E402
    ChunkReplay,
    EpisodeTransitions,
    NoiseActor,
    NoiseSAC,
    NoiseSACConfig,
    Transition,
    soft_update,
)


def cfg(**kw):
    base = dict(feature_dim=4, action_dim=3, hidden=32, layers=2)
    base.update(kw)
    return NoiseSACConfig(**base)


def test_untrained_actor_is_input_independent_and_close_to_unit_gaussian_noise():
    torch.manual_seed(0)
    c = NoiseSACConfig(feature_dim=4, action_dim=70)
    actor = NoiseActor(c)
    feats = torch.randn(4000, 4) * 10
    u_det, logp_det = actor(feats, deterministic=True)
    assert torch.equal(u_det, torch.zeros_like(u_det)) and logp_det is None
    u, logp = actor(feats)
    noise = c.noise_scale * u
    assert u.abs().max() <= 1 and torch.isfinite(logp).all() and logp.shape == (4000, 1)
    assert 0.9 < noise.std().item() < 1.1 and abs(noise.mean().item()) < 0.02


def test_log_prob_matches_the_tanh_transformed_gaussian():
    torch.manual_seed(1)
    actor = NoiseActor(cfg())
    feats = torch.randn(64, 4)
    u, logp = actor(feats)
    mean, log_std = actor.net(feats).chunk(2, dim=-1)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, log_std.clamp(-10, 2).exp()), [torch.distributions.TanhTransform()])
    ref = dist.log_prob(u.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1, keepdim=True)
    assert torch.allclose(logp, ref, atol=1e-3)


def test_transitions_of_a_successful_and_a_failed_episode():
    f = [np.full(4, k, np.float32) for k in range(4)]
    u = [np.full(3, k / 10, np.float32) for k in range(3)]
    tr = EpisodeTransitions()
    for k in range(3):
        tr.decide(f[k], u[k])
    ts = tr.finish(success=True)
    assert [t.reward for t in ts] == [0.0, 0.0, 1.0] and [t.terminal for t in ts] == [False, False, True]
    assert [t.feats[0] for t in ts] == [0, 1, 2] and [t.next_feats[0] for t in ts[:2]] == [1, 2]
    assert np.allclose(ts[1].u, 0.1) and not ts[2].next_feats.any()
    with pytest.raises(RuntimeError):
        tr.decide(f[3], u[0])
    with pytest.raises(RuntimeError):
        tr.finish(success=True)

    tr = EpisodeTransitions()
    tr.decide(f[0], u[0])
    tr.decide(f[1], u[1])
    with pytest.raises(ValueError):
        _failed_without_bootstrap()
    ts = tr.finish(success=False, bootstrap_feats=f[3])
    assert [t.reward for t in ts] == [0.0, 0.0] and not any(t.terminal for t in ts)
    assert ts[-1].next_feats[0] == 3 and ts[-1].feats[0] == 1
    assert EpisodeTransitions().finish(success=True) == []


def _failed_without_bootstrap():
    tr = EpisodeTransitions()
    tr.decide(np.zeros(4, np.float32), np.zeros(3, np.float32))
    tr.finish(success=False)


def test_replay_sample_and_parts_round_trip(tmp_path):
    rb = ChunkReplay(capacity=6, feature_dim=4, action_dim=3, seed=0)
    for k in range(2):
        rb.add(Transition(np.full(4, k, np.float32), np.full(3, k, np.float32), float(k), np.full(4, k + 1, np.float32),
                          k % 2 == 0))
    rb.save_part(tmp_path / "a.npz", 0)
    for k in range(2, 5):
        rb.add(Transition(np.full(4, k, np.float32), np.full(3, k, np.float32), float(k), np.full(4, k + 1, np.float32),
                          k % 2 == 0))
    rb.save_part(tmp_path / "b.npz", 2)
    assert len(rb) == 5
    b = rb.sample(400)
    assert b["feats"].shape == (400, 4) and b["u"].shape == (400, 3)
    assert b["reward"].shape == (400, 1) and b["terminal"].shape == (400, 1) and b["feats"].dtype == torch.float32
    assert torch.equal(b["next_feats"][:, 0], b["feats"][:, 0] + 1) and torch.equal(b["reward"][:, 0], b["u"][:, 0])
    assert torch.equal(b["terminal"][:, 0], (b["reward"][:, 0] % 2 == 0).float())
    assert set(b["reward"][:, 0].tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0}          # every stored transition, none beyond
    back = ChunkReplay(capacity=6, feature_dim=4, action_dim=3)
    back.load_parts([tmp_path / "a.npz", tmp_path / "b.npz"])
    for k in ChunkReplay.ARRAYS:
        assert np.array_equal(getattr(rb, k)[:5], getattr(back, k)[:5])
    with pytest.raises(ValueError):                                              # out of order
        ChunkReplay(capacity=6, feature_dim=4, action_dim=3).load_parts([tmp_path / "b.npz"])
    with pytest.raises(ValueError):                                              # beyond capacity
        ChunkReplay(capacity=4, feature_dim=4, action_dim=3).load_parts([tmp_path / "a.npz", tmp_path / "b.npz"])
    exact = ChunkReplay(capacity=5, feature_dim=4, action_dim=3)
    exact.load_parts([tmp_path / "a.npz", tmp_path / "b.npz"])                  # exactly full is fine
    assert len(exact) == 5
    rb.add(Transition(np.zeros(4, np.float32), np.zeros(3, np.float32), 0.0, np.zeros(4, np.float32), False))
    with pytest.raises(RuntimeError):
        rb.add(Transition(np.zeros(4, np.float32), np.zeros(3, np.float32), 0.0, np.zeros(4, np.float32), False))


def test_sac_objectives_known_answers():
    from noise_sac import actor_objective, alpha_objective, td_target

    r, d = torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0], [1.0]])
    q1, q2, logp = torch.tensor([[0.5], [9.0]]), torch.tensor([[0.8], [9.0]]), torch.tensor([[-2.0], [5.0]])
    # 0 + 0.9 * (min(0.5, 0.8) - 0.1 * -2) = 0.63 ; terminal: just the reward
    assert torch.allclose(td_target(r, d, 0.9, q1, q2, 0.1, logp), torch.tensor([[0.63], [1.0]]))
    # mean(0.1 * logp - min(q1, q2)) = mean(-0.2 - 0.5, 0.5 - 9.0) = -4.6
    assert actor_objective(0.1, logp, q1, q2).item() == pytest.approx(-4.6)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    loss = alpha_objective(log_alpha, torch.tensor([[-80.0]]), -70.0)   # entropy 80 above target -70
    loss.backward()
    assert loss.item() == 0.0 and log_alpha.grad.item() == pytest.approx(150.0)   # gradient descent lowers alpha
    assert NoiseSAC(cfg(action_dim=70, feature_dim=4)).target_entropy == -70.0


def test_critic_shapes_and_frozen_target():
    learner = NoiseSAC(cfg())
    q1, q2 = learner.critic(torch.zeros(5, 4), torch.zeros(5, 3))
    assert q1.shape == (5, 1) and q2.shape == (5, 1)
    assert not any(p.requires_grad for p in learner.critic_target.parameters())
    b = batch(np.random.randn(8, 4), np.random.uniform(-1, 1, (8, 3)), np.ones(8), np.random.randn(8, 4), np.ones(8))
    out = learner.update(b)
    assert out["entropy"] > 0          # the untrained actor: about 0.5 nats per dimension, logged as -logp


def batch(feats, u, reward, next_feats, terminal):
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32))
    return {"feats": t(feats), "u": t(u), "reward": t(reward)[:, None], "next_feats": t(next_feats),
            "terminal": t(terminal)[:, None]}


def test_critic_learns_discounted_success_on_a_two_state_chain():
    # s0 -> s1 with reward 0, s1 -> success (terminal, reward 1). Q(s1) -> 1, Q(s0) -> gamma.
    torch.manual_seed(0)
    # Each state appears with several u, so Q generalises over the u the target samples from the actor.
    s0, s1, end = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]
    u = np.random.default_rng(0).uniform(-1, 1, (32, 3))
    b = batch([s0, s1] * 16, u, [0.0, 1.0] * 16, [s1, end] * 16, [0.0, 1.0] * 16)
    learner = NoiseSAC(cfg(init_alpha=1e-8, gamma=0.9, tau=0.2, lr=3e-3))
    for _ in range(600):
        learner.update(b, update_actor=False)
    q = torch.min(*learner.critic(b["feats"], b["u"]))[:, 0]
    assert q[1::2].mean().item() == pytest.approx(1.0, abs=0.03) and q[0::2].mean().item() == pytest.approx(0.9, abs=0.05)
    # A time-limit transition (terminal 0) bootstraps instead of stopping.
    learner2 = NoiseSAC(cfg(init_alpha=1e-8, gamma=0.9, tau=0.2, lr=3e-3))
    b2 = batch([s0, s1] * 16, u, [0.0, 1.0] * 16, [s1, end] * 16, [0.0, 0.0] * 16)
    for _ in range(600):
        learner2.update(b2, update_actor=False)
    assert torch.min(*learner2.critic(b2["feats"], b2["u"]))[1::2, 0].mean().item() > 1.2


def test_critic_only_updates_leave_actor_and_alpha_unchanged():
    torch.manual_seed(0)
    learner = NoiseSAC(cfg())
    before = [p.clone() for p in learner.actor.parameters()]
    a0 = learner.alpha
    b = batch(np.random.randn(8, 4), np.random.uniform(-1, 1, (8, 3)), np.ones(8), np.random.randn(8, 4), np.ones(8))
    out = learner.update(b, update_actor=False)
    assert "actor_loss" not in out and learner.alpha == a0
    assert all(torch.equal(p, q) for p, q in zip(before, learner.actor.parameters()))
    out = learner.update(b, update_actor=True)
    assert "actor_loss" in out and learner.alpha != a0
    assert any(not torch.equal(p, q) for p, q in zip(before, learner.actor.parameters()))


def test_actor_learns_the_rewarded_noise_on_a_bandit():
    # One-step episodes: reward 1 when u[0] > 0.5. The deterministic actor must end there.
    torch.manual_seed(0)
    learner = NoiseSAC(cfg(lr=3e-3, init_alpha=1e-3))
    rb = ChunkReplay(5000, 4, 3, seed=0)
    feats = np.ones(4, np.float32)
    for _ in range(400):
        u = learner.act(feats, deterministic=False)
        rb.add(Transition(feats, u, float(u[0] > 0.5), np.zeros(4, np.float32), True))
    for k in range(500):
        learner.update(rb.sample(64))
        if k % 5 == 0:
            u = learner.act(feats, deterministic=False)
            rb.add(Transition(feats, u, float(u[0] > 0.5), np.zeros(4, np.float32), True))
    assert learner.act(feats, deterministic=True)[0] > 0.5


def test_checkpoint_round_trip_gives_identical_outputs():
    torch.manual_seed(0)
    learner = NoiseSAC(cfg())
    b = batch(np.random.randn(16, 4), np.random.uniform(-1, 1, (16, 3)), np.ones(16), np.random.randn(16, 4), np.ones(16))
    for _ in range(5):
        learner.update(b)
    sd = learner.state_dict()
    copy = NoiseSAC.from_state_dict(sd)
    f = np.random.randn(4).astype(np.float32)
    assert np.array_equal(learner.act(f, True), copy.act(f, True))
    assert torch.equal(learner.critic(b["feats"], b["u"])[0], copy.critic(b["feats"], b["u"])[0])
    assert torch.equal(learner.critic_target(b["feats"], b["u"])[1], copy.critic_target(b["feats"], b["u"])[1])
    assert copy.alpha == pytest.approx(learner.alpha)
    with pytest.raises(ValueError):
        NoiseSAC.from_state_dict({**sd, "format": "other"})


def test_soft_update():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        a.weight.fill_(0.0), b.weight.fill_(1.0)
    soft_update(a, b, 0.25)
    assert torch.allclose(a.weight, torch.full_like(a.weight, 0.25))
