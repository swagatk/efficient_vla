import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "noise_reachability"))

import noise_awr as na  # noqa: E402

FEATS = 12


def small_cfg(**kw):
    return na.NoiseAWRConfig(feature_dim=FEATS, noise_dim=4, hidden=32, layers=2, batch_size=64, **kw)


def batch(n=8, success=1.0):
    return {"feats": torch.rand(n, FEATS), "z": torch.randn(n, 4), "ret": torch.rand(n),
            "success": torch.full((n,), success)}


def test_untrained_actor_is_the_standard_normal_whatever_its_input():
    actor = na.GaussianNoiseActor(small_cfg())
    mean, log_std = actor(torch.rand(5, FEATS))
    assert torch.equal(mean, torch.zeros(5, 4)) and torch.equal(log_std, torch.zeros(5, 4))
    z = actor.sample(torch.rand(20_000, FEATS))
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    assert torch.equal(actor.sample(torch.rand(3, FEATS), deterministic=True), torch.zeros(3, 4))


def test_log_prob_is_averaged_over_noise_dimensions():
    actor = na.GaussianNoiseActor(small_cfg())
    lp = actor.log_prob(torch.rand(2, FEATS), torch.zeros(2, 4)).detach()
    assert lp.shape == (2,) and lp == pytest.approx(-0.5 * np.log(2 * np.pi), abs=1e-6)


def test_log_std_is_clamped():
    actor = na.GaussianNoiseActor(small_cfg())
    nn_bias = torch.full((4,), 50.0)
    actor.log_std.bias.data.copy_(nn_bias)
    assert actor(torch.rand(1, FEATS))[1].max() == 2.0
    actor.log_std.bias.data.copy_(-nn_bias)
    assert actor(torch.rand(1, FEATS))[1].min() == -20.0


def test_two_hot_known_answers_and_expectation():
    t = na.two_hot(torch.tensor([0.0, 1.0, 0.5, 2.0, -1.0]), 3)
    assert t.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    v = torch.rand(100)
    t = na.two_hot(v, 16)
    assert t.sum(-1) == pytest.approx(torch.ones(100), abs=1e-6)
    assert (t * torch.linspace(0, 1, 16)).sum(-1) == pytest.approx(v, abs=1e-5)


def test_awr_weights_filter_failures_and_cap():
    w = na.awr_weights(torch.tensor([0.0, 0.5, 100.0, 5.0]), torch.tensor([1.0, 1.0, 1.0, 0.0]), 0.5, 20.0)
    assert w == pytest.approx(torch.tensor([1.0, float(np.e), 20.0, 0.0]))


def test_episode_returns_discount_to_the_success_decision():
    assert na.episode_returns(3, True, 0.5) == pytest.approx(np.array([0.25, 0.5, 1.0]))
    assert na.episode_returns(1, True, 0.99).tolist() == [1.0]
    assert na.episode_returns(3, False, 0.5).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        na.episode_returns(0, True, 0.5)


def test_replay_balances_outcomes_and_refuses_bad_arrays():
    n = 1000
    arrays = {"feats": np.zeros((n, FEATS), np.float16), "z": np.zeros((n, 4), np.float32),
              "ret": np.zeros(n, np.float32), "success": (np.arange(n) < 10).astype(np.float32)}
    b = na.Replay(arrays, 0).sample(200, 0.5)
    assert b["success"].sum() == 100 and len(b["z"]) == 200
    half = dict(arrays, success=(np.arange(n) < 500).astype(np.float32))
    assert na.Replay(half, 1).sample(2000, 0.5)["success"].sum() == 1000    # failures draw no successes
    only = dict(arrays, success=np.zeros(n, np.float32))
    assert na.Replay(only, 0).sample(50, 0.5)["success"].sum() == 0
    with pytest.raises(ValueError):
        na.Replay(dict(arrays, ret=np.zeros(3)), 0)
    with pytest.raises(ValueError):
        na.Replay({k: v for k, v in arrays.items() if k != "ret"}, 0)


def test_actor_fits_successes_and_ignores_failures():
    torch.manual_seed(0)
    learner = na.NoiseAWR(small_cfg(actor_lr=3e-3, value_lr=1e-3))
    f = torch.rand(64, FEATS)
    b = {"feats": f, "z": torch.cat([torch.full((32, 4), 1.5), torch.full((32, 4), -1.5)]),
         "ret": torch.cat([torch.ones(32), torch.zeros(32)]), "success": torch.cat([torch.ones(32), torch.zeros(32)])}
    for _ in range(600):
        learner.update(b, update_actor=True)
    mean = learner.actor(f)[0].mean()
    assert abs(mean - 1.5) < 0.2                                # failures carry no weight
    v = learner.value.value(learner.value(f))
    assert v[:32].mean() > 0.6 and v[32:].mean() < 0.4          # overfits one batch


def test_value_only_and_no_success_leave_the_actor_alone():
    learner = na.NoiseAWR(small_cfg())
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    assert "actor_loss" not in learner.update(batch(), update_actor=False)
    assert "actor_loss" not in learner.update(batch(success=0.0), update_actor=True)
    assert all(torch.equal(before[k], v) for k, v in learner.actor.state_dict().items())
    assert "actor_loss" in learner.update(batch(), update_actor=True)
    assert any(not torch.equal(before[k], v) for k, v in learner.actor.state_dict().items())


def test_gradients_reach_every_actor_and_value_parameter():
    learner = na.NoiseAWR(small_cfg())
    learner.update(batch(), update_actor=True)      # zero heads: the body has no gradient until they have moved
    learner.update(batch(), update_actor=True)
    for name, p in [*learner.actor.named_parameters(), *learner.value.named_parameters()]:
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_checkpoint_round_trip_gives_identical_outputs():
    learner = na.NoiseAWR(small_cfg())
    learner.update(batch(), update_actor=True)
    other = na.NoiseAWR.from_state_dict(learner.state_dict())
    f = torch.rand(4, FEATS)
    assert torch.equal(learner.actor(f)[0], other.actor(f)[0])
    assert torch.equal(learner.value(f), other.value(f))
    assert learner.actor_opt.state_dict()["state"].keys() == other.actor_opt.state_dict()["state"].keys()
    with pytest.raises(ValueError):
        na.NoiseAWR.from_state_dict({"format": "other"})


def test_network_shapes_of_the_paper():
    cfg = na.NoiseAWRConfig(feature_dim=FEATS)
    actor, value = na.GaussianNoiseActor(cfg), na.BinnedValue(cfg)
    kinds = [type(m).__name__ for m in actor.body]
    assert kinds == ["Linear", "LayerNorm", "ReLU"] * 4
    assert [type(m).__name__ for m in value.net[0]] == ["Linear", "ReLU"] * 4
    assert actor.mean.out_features == 32 and value.net[1].out_features == 16


def test_a_sample_is_the_mean_plus_scaled_noise():
    actor = na.GaussianNoiseActor(small_cfg())
    actor.mean.bias.data.fill_(3.0)
    actor.log_std.bias.data.fill_(np.log(0.5))
    z = actor.sample(torch.rand(20_000, FEATS))
    assert abs(z.mean() - 3.0) < 0.02 and abs(z.std() - 0.5) < 0.02


def test_expected_value_uses_the_support_on_zero_to_one():
    value = na.BinnedValue(small_cfg())
    assert value.support[0] == 0.0 and value.support[-1] == 1.0
    logits = torch.tensor([[-50.0] * 15 + [50.0], [50.0] + [-50.0] * 15, [0.0] * 16])
    assert value.value(logits) == pytest.approx(torch.tensor([1.0, 0.0, 0.5]), abs=1e-5)


def test_value_loss_is_lowest_at_the_two_hot_target():
    b = na.two_hot(torch.tensor([0.3]), 16)
    best = torch.nn.functional.cross_entropy(torch.log(b + 1e-9), b)
    assert best < torch.nn.functional.cross_entropy(torch.zeros(1, 16), b)


def test_actor_update_descends_on_the_weighted_likelihood():
    torch.manual_seed(1)
    learner = na.NoiseAWR(small_cfg(actor_lr=1e-3))
    b = batch(32)
    b["z"] = torch.full((32, 4), 0.8)
    before = learner.actor.log_prob(b["feats"], b["z"]).mean().item()
    for _ in range(20):
        learner.update(b, update_actor=True)
    assert learner.actor.log_prob(b["feats"], b["z"]).mean().item() > before + 0.05


def test_advantage_is_the_return_minus_the_value():
    learner = na.NoiseAWR(small_cfg(beta=1.0, max_weight=1e6))
    learner.value.net[1].weight.data.zero_()
    learner.value.net[1].bias.data.fill_(0.0)              # uniform bins: V = 0.5 everywhere
    seen = []
    real = na.awr_weights
    na.awr_weights = lambda adv, *a: seen.append(adv) or real(adv, *a)
    try:
        b = batch(4)
        b["ret"] = torch.tensor([1.0, 0.75, 0.5, 0.0])
        learner.update(b, update_actor=True)
    finally:
        na.awr_weights = real
    # The value update precedes the advantage, so V has moved only by one small step from 0.5.
    assert seen[0] == pytest.approx(torch.tensor([0.5, 0.25, 0.0, -0.5]), abs=0.01)


def test_gradient_norm_is_clipped():
    learner = na.NoiseAWR(small_cfg(grad_clip=1e-6))
    b = batch(16)
    b["z"] = torch.full((16, 4), 9.0)
    learner.update(b, update_actor=True)
    learner.update(b, update_actor=True)
    norm = torch.nn.utils.get_total_norm([p.grad for p in learner.actor.parameters()])
    assert norm <= 1e-6 * 1.01
