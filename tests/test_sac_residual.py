"""CPU tests for the residual SAC head, replay buffer and update (small images, no ImageNet download)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl_based_tuning"))

from sac_residual import (CHECKPOINT_FORMAT, HeadConfig, ReplayBuffer, SACConfig, SACLearner,  # noqa: E402
                          SACResidualHead, head_from_checkpoint, image_for_head)
from vla_core.residual import apply_residual  # noqa: E402

SIZE = 32


def make_head(seed=0):
    torch.manual_seed(seed)
    return SACResidualHead(HeadConfig(image_size=SIZE, pretrained_vision=False))


def batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = lambda: torch.randint(0, 256, (n, 3, SIZE, SIZE), dtype=torch.uint8, generator=g)
    vec = lambda d: torch.rand((n, d), generator=g) * 2 - 1
    return {"image": img(), "state": vec(8), "base_action": vec(7), "delta": vec(7).clamp(-0.99, 0.99),
            "reward": vec(1), "next_image": img(), "next_state": vec(8), "next_base_action": vec(7),
            "terminal": torch.zeros(n, 1)}


def test_shapes_and_initial_policy_is_the_base_policy():
    head, b = make_head(), batch()
    feats = head.features(b["image"], b["state"], b["base_action"])
    assert feats.shape == (8, 512 + 64 + 7)
    delta, logp = head.sample(feats, deterministic=True)
    assert logp is None and torch.equal(delta, torch.zeros(8, 7))
    base = b["base_action"][0].numpy()
    assert np.array_equal(apply_residual(base, delta[0].detach().numpy(), 0.1), base.clip(-1, 1))
    delta, logp = head.sample(feats)
    assert delta.shape == (8, 7) and logp.shape == (8, 1) and torch.isfinite(logp).all()
    assert (delta.abs() < 1).all()


def test_batchnorm_stays_frozen():
    head = make_head().train()
    assert all(not m.training for m in head.vision_encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = head.vision_encoder.bn1.running_mean.clone()
    b = batch()
    head.features(b["image"], b["state"], b["base_action"])
    assert torch.equal(before, head.vision_encoder.bn1.running_mean)


def test_update_trains_actor_and_critic_and_soft_updates_target():
    head = make_head()
    learner = SACLearner(head, SACConfig(tau=0.1, auto_alpha=True))
    actor0 = head.actor_mean.weight.clone()
    critic0 = head.q1[0].weight.clone()
    target0 = learner.target.q1[0].weight.clone()
    out = learner.update(batch())
    assert {"critic_loss", "actor_loss", "alpha", "alpha_loss"} <= set(out)
    assert not torch.equal(actor0, head.actor_mean.weight)
    assert not torch.equal(critic0, head.q1[0].weight)
    expected = 0.9 * target0 + 0.1 * head.q1[0].weight
    assert torch.allclose(learner.target.q1[0].weight, expected, atol=1e-6)
    assert all(p.grad is None for p in head.q1.parameters())  # actor step leaves no critic gradients behind
    assert not any(p.requires_grad for p in learner.target.parameters())


def test_critic_fits_a_fixed_batch():
    head = make_head()
    learner = SACLearner(head, SACConfig(gamma=0.0), lr=1e-3)
    b = batch(16)
    b["terminal"] = torch.ones(16, 1)
    first = learner.update(b)["critic_loss"]
    for _ in range(150):
        last = learner.update(b)["critic_loss"]
    assert last < 0.05 * first


def test_checkpoint_round_trip_and_old_format_rejected():
    head = make_head()
    learner = SACLearner(head, SACConfig())
    learner.update(batch())
    sd = learner.state_dict()
    assert sd["format"] == CHECKPOINT_FORMAT
    restored = head_from_checkpoint(sd)
    b = batch(seed=1)
    with torch.no_grad():
        a = head.sample(head.features(b["image"], b["state"], b["base_action"]), deterministic=True)[0]
        c = restored.sample(restored.features(b["image"], b["state"], b["base_action"]), deterministic=True)[0]
    assert torch.equal(a, c)
    learner2 = SACLearner(make_head(seed=5), SACConfig())
    learner2.load_state_dict(sd)
    assert torch.equal(learner2.head.q2[0].weight, head.q2[0].weight)
    with pytest.raises(ValueError):
        head_from_checkpoint({"agent_trainable_state_dict": {}})


def test_same_seed_same_update():
    losses = []
    for _ in range(2):
        torch.manual_seed(3)
        learner = SACLearner(make_head(seed=3), SACConfig())
        losses.append(learner.update(batch(seed=4))["critic_loss"])
    assert losses[0] == losses[1]


def test_replay_buffer_ring_and_success_sampling():
    buf = ReplayBuffer(capacity=5, image_size=SIZE, seed=0)
    img = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    idx = [buf.add(img + k, np.full(8, k), np.zeros(7), np.zeros(7), float(k), img, np.zeros(8), np.zeros(7), False)
           for k in range(7)]
    assert len(buf) == 5 and idx[5:] == [0, 1]
    assert sorted(buf.reward[:, 0].tolist()) == [2, 3, 4, 5, 6]
    buf.mark_success([3])
    s = buf.sample(10, success_fraction=1.0)
    assert s["image"].shape == (10, 3, SIZE, SIZE) and s["image"].dtype == torch.uint8
    # At most as many success samples as success transitions (one here), as in the pre-fix buffer.
    assert (s["reward"] == 3).sum() >= 1
    assert torch.all(buf.sample(1, success_fraction=1.0)["reward"] == 3)
    assert buf.n_success() == 1
    buf.add(img, np.zeros(8), np.zeros(7), np.zeros(7), 0.0, img, np.zeros(8), np.zeros(7), True)  # overwrites slot 2
    buf.add(img, np.zeros(8), np.zeros(7), np.zeros(7), 0.0, img, np.zeros(8), np.zeros(7), True)  # overwrites slot 3
    assert buf.n_success() == 0


def test_image_for_head():
    img = torch.rand(1, 3, 64, 64)
    out = image_for_head(img, SIZE)
    assert out.shape == (1, 3, SIZE, SIZE) and out.dtype == torch.uint8
    same = image_for_head(img, 64)
    assert torch.equal(same, (img * 255).round().to(torch.uint8))


def test_replay_buffer_save_load_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=6, image_size=SIZE, seed=1)
    img = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    for k in range(8):  # wraps around
        buf.add(img + k, np.full(8, k), np.zeros(7), np.full(7, 0.1 * k), float(k), img, np.zeros(8), np.zeros(7), k == 7)
    buf.mark_success([1])
    path = tmp_path / "replay.npz"
    buf.save(path)
    other = ReplayBuffer(capacity=6, image_size=SIZE, seed=99)
    other.load(path)
    assert len(other) == 6 and other.pos == buf.pos and other.n_success() == 1
    for k in ReplayBuffer.ARRAYS:
        assert np.array_equal(getattr(other, k), getattr(buf, k))
    a, b = buf.sample(4, 0.5), other.sample(4, 0.5)  # same RNG state -> same minibatch
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=3, image_size=SIZE).load(path)


def test_cli_defaults_select_the_lerobot_base_policy():
    import train_online_sac as t
    args = t.parse_args([])
    assert args.policy_path == "lerobot/smolvla_libero" and args.n_action_steps == 1
    assert args.rename_map == t.DEFAULT_RENAME_MAP
    old = t.parse_args(["--policy_path", "HuggingFaceVLA/smolvla_libero", "--n_action_steps", "-1", "--rename_map", "{}"])
    assert old.n_action_steps is None and old.rename_map is None
