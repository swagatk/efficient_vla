"""CPU tests for self-filtered BC: rollout storage, sample construction, episode selection, and LoRA adapters
(on a small random module standing in for SmolVLA's action expert)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "self_filtered_bc"))

from rollout_data import (IMAGE_KEYS, PAD_KEY, STATE_KEY, EpisodeRecorder, RolloutDataset,  # noqa: E402
                          action_chunk, image_to_uint8, load_episode, save_episode, select_episodes)
from vla_core.lora import (LoraSettings, add_lora, load_heads, load_lora, resolve_targets,  # noqa: E402
                           save_lora, set_heads_trainable, trainable_parameters)

H = W = 8


def fake_env_obs(rng):
    """env_obs as BasePolicy.act returns it: (1, 3, H, W) float images from uint8 renders, (1, 8) state."""
    obs = {k: torch.from_numpy(rng.integers(0, 256, (1, 3, H, W)).astype(np.float32) / 255.0) for k in IMAGE_KEYS}
    obs[STATE_KEY] = torch.from_numpy(rng.normal(size=(1, 8)).astype(np.float32))
    return obs


def record(n_steps, stride=1, seed=0):
    rng = np.random.default_rng(seed)
    rec, obs_log = EpisodeRecorder(frame_stride=stride), []
    for t in range(n_steps):
        obs = fake_env_obs(rng)
        obs_log.append(obs)
        rec.add(obs, np.full(7, t, np.float32))
    return rec, obs_log


def test_image_to_uint8_is_lossless_and_rejects_non_renders():
    img = np.random.default_rng(0).integers(0, 256, (3, H, W)).astype(np.uint8)
    x = torch.from_numpy(img).float()[None] / 255.0
    assert np.array_equal(image_to_uint8(x), img)
    with pytest.raises(ValueError):
        image_to_uint8(x + 0.5 / 255.0)
    with pytest.raises(ValueError):
        image_to_uint8(torch.zeros(2, 3, H, W))


def test_action_chunk_pads_past_episode_end():
    actions = np.arange(10, dtype=np.float32)[:, None].repeat(7, 1)
    chunk, pad = action_chunk(actions, 7, 5)
    assert chunk[:, 0].tolist() == [7, 8, 9, 9, 9] and pad.tolist() == [False, False, False, True, True]
    chunk, pad = action_chunk(actions, 0, 5)
    assert chunk[:, 0].tolist() == [0, 1, 2, 3, 4] and not pad.any()
    with pytest.raises(IndexError):
        action_chunk(actions, 10, 5)


def test_save_load_round_trip_and_dataset_samples(tmp_path):
    rec, obs_log = record(7, stride=2)
    path = tmp_path / "ep.npz"
    save_episode(path, rec, {"task": "do it", "init_state": 12, "seed": 0, "success": True, "steps": 7})
    ep = load_episode(path)
    assert ep["frame_steps"].tolist() == [0, 2, 4, 6] and ep["actions"].shape == (7, 7)
    assert ep["meta"]["frame_stride"] == 2 and ep["meta"]["n_steps"] == 7 and ep["meta"]["init_state"] == 12
    assert not list(tmp_path.glob("*.tmp*"))

    ds = RolloutDataset([path], chunk_size=4)
    assert len(ds) == 4
    s = ds[2]  # frame recorded at step 4
    for k in IMAGE_KEYS:
        assert s[k].shape == (3, H, W) and torch.equal(s[k], obs_log[4][k][0])
    assert torch.equal(s[STATE_KEY], obs_log[4][STATE_KEY][0])
    assert s["action"][:, 0].tolist() == [4, 5, 6, 6] and s[PAD_KEY].tolist() == [False, False, False, True]
    assert s["task"] == "do it"
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    assert batch["action"].shape == (2, 4, 7) and batch["task"] == ["do it", "do it"]


def test_select_episodes():
    metas = [{"success": s} for s in (True, False, False, True, False, True, False, False)]
    assert select_episodes(metas, "success") == [0, 3, 5]
    assert select_episodes(metas, "all") == list(range(8))
    matched = select_episodes(metas, "all_matched", seed=1)
    assert len(matched) == 3 and len(set(matched)) == 3 and matched == select_episodes(metas, "all_matched", seed=1)
    with pytest.raises(ValueError):
        select_episodes(metas, "failures")


class TinyExpert(nn.Module):
    """Stands in for lm_expert: named q/k/v/o and MLP projections."""

    def __init__(self, d=16):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(d, d))

    def forward(self, x):
        h = self.o_proj(self.q_proj(x) * torch.sigmoid(self.k_proj(x)) + self.v_proj(x))
        return self.down_proj(torch.relu(self.gate_proj(h)) * self.up_proj(h))


class TinyVLA(nn.Module):
    """Stands in for VLAFlowMatching: an expert plus the four action heads."""

    def __init__(self, d=16):
        super().__init__()
        self.expert = TinyExpert(d)
        self.action_in_proj, self.action_out_proj = nn.Linear(7, d), nn.Linear(d, 7)
        self.action_time_mlp_in, self.action_time_mlp_out = nn.Linear(2 * d, d), nn.Linear(d, d)

    def forward(self, a):
        return self.action_out_proj(self.expert(self.action_in_proj(a)))


def test_resolve_targets():
    assert resolve_targets("attn") == ("q_proj", "v_proj") and len(resolve_targets("all")) == 7
    assert resolve_targets("q_proj, o_proj") == ("q_proj", "o_proj")
    with pytest.raises(ValueError):
        resolve_targets(" , ")


def test_fresh_lora_reproduces_base_and_only_lora_trains():
    torch.manual_seed(0)
    expert = TinyExpert()
    x = torch.randn(5, 16)
    before = expert(x).detach()
    frozen = {n: p.detach().clone() for n, p in expert.named_parameters()}
    peft_expert = add_lora(expert, LoraSettings(r=4, alpha=8))
    assert torch.equal(peft_expert(x), before)

    params = trainable_parameters(peft_expert)
    names = [n for n, p in peft_expert.named_parameters() if p.requires_grad]
    assert params and all("lora_" in n for n in names)
    assert len(names) == 2 * 7  # lora_A and lora_B on each of the seven projections
    opt = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        peft_expert(x).pow(2).mean().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in peft_expert.named_parameters()
                   if "lora_B" in n)
        opt.step()
    assert not torch.equal(peft_expert(x), before)
    for n, p in peft_expert.get_base_model().named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, frozen[n.replace(".base_layer", "")]), n


def test_lora_overfits_one_batch():
    torch.manual_seed(0)
    model = TinyVLA()
    for p in model.parameters():
        p.requires_grad_(False)
    model.expert = add_lora(model.expert, LoraSettings(r=8, alpha=16))
    a, target = torch.randn(8, 7), torch.randn(8, 7)
    opt = torch.optim.AdamW(trainable_parameters(model), lr=3e-2)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = (model(a) - target).pow(2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first / 10


def test_save_load_round_trip_with_heads(tmp_path):
    torch.manual_seed(0)
    model = TinyVLA()
    for p in model.parameters():
        p.requires_grad_(False)
    model.expert = add_lora(model.expert, LoraSettings(r=4, alpha=8))
    heads = set_heads_trainable(model, True)
    assert len(heads) == 8
    with torch.no_grad():
        for p in trainable_parameters(model):
            p.add_(0.1 * torch.randn_like(p))
    a = torch.randn(3, 7)
    out = model(a)
    save_lora(model.expert, tmp_path / "adapter", heads=model)

    torch.manual_seed(0)
    reloaded = TinyVLA()  # same base weights as `model` before training
    reloaded.expert = load_lora(reloaded.expert, tmp_path / "adapter")
    assert load_heads(reloaded, tmp_path / "adapter")
    assert torch.equal(reloaded(a), out)
    assert not trainable_parameters(reloaded.expert)
    assert not load_heads(TinyVLA(), tmp_path)  # no heads file there


def save_fake_episodes(tmp_path, outcomes):
    """One saved episode per (init_state, success) in ``outcomes``."""
    for k, (init_state, success) in enumerate(outcomes):
        rec, _ = record(3, seed=k)
        save_episode(tmp_path / "episodes" / f"seed0_init{init_state:02d}.npz", rec,
                     {"task": "t", "task_id": 6, "policy_path": "p", "init_state": init_state, "seed": 0,
                      "success": success, "steps": 3})


def test_split_holds_out_validation_init_states(tmp_path):
    import self_filtered_bc as sfbc

    save_fake_episodes(tmp_path, [(10, True), (11, False), (12, True), (13, True), (14, False)])
    train, val, metas = sfbc.split_episodes(tmp_path, "success", {13, 14}, seed=0)
    assert [p.name for p in train] == ["seed0_init10.npz", "seed0_init12.npz"]
    assert [p.name for p in val] == ["seed0_init13.npz"] and len(metas) == 5
    train, _, _ = sfbc.split_episodes(tmp_path, "all_matched", {13, 14}, seed=0)
    assert len(train) == 2 and not {p.name for p in train} & {"seed0_init13.npz", "seed0_init14.npz"}


def test_policy_batch_passes_pad_mask_after_preprocessing(tmp_path):
    import self_filtered_bc as sfbc

    save_fake_episodes(tmp_path, [(10, True)])
    ds = RolloutDataset(sorted((tmp_path / "episodes").glob("*.npz")), chunk_size=5)
    seen = {}

    def preprocessor(batch):
        seen.update(batch)
        return {k: v for k, v in batch.items()}

    out = sfbc.policy_batch(torch.utils.data.default_collate([ds[0], ds[1]]), preprocessor, "cpu")
    assert PAD_KEY not in seen and "action" in seen and seen["task"] == ["t", "t"]
    assert out[PAD_KEY].shape == (2, 5) and out[PAD_KEY][1].tolist() == [False, False, True, True, True]


def test_parse_args_defaults():
    import self_filtered_bc as sfbc

    a = sfbc.parse_args(["--stage", "train"])
    assert a.task_id == 6 and a.n_action_steps == 1 and a.selection == "success" and a.train_action_heads == 0
    assert a.lora_dir.endswith("success_seed0/adapter") and a.eval_output_dir.endswith("success_seed0/eval")
    assert sfbc.parse_args(["--stage", "eval", "--n_action_steps", "-1"]).n_action_steps is None
