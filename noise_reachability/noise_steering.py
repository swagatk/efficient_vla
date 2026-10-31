"""Step 3: let a callback choose part of SmolVLA's initial noise at every inference (README.md, step 3).

``install(policy, choose_noise)`` wraps ``policy.model.sample_actions`` on that one policy instance, like
``noise_selection.install``. At each inference the wrapper

1. draws the full (1, chunk_size, max_action_dim) noise with SmolVLA's own ``sample_noise``, exactly as
   ``sample_actions`` does when it is given no noise;
2. computes the prefix features: SmolVLA's prefix pass (``embed_prefix`` and the VLM forward, with the same
   arguments as in ``sample_actions``), averaged over valid tokens, plus the first ``state_dim`` entries of the
   normalised state;
3. calls ``choose_noise(features, drawn)``, ``drawn`` being the (n_exec, action_dim) part of the drawn noise at
   the executed positions and real action dimensions; a returned tensor of that shape replaces that part,
   ``None`` keeps the drawn noise;
4. calls SmolVLA's unchanged ``sample_actions`` with that noise.

With a callback that always returns ``None`` the chunk is the base policy's bit for bit: the prefix pass uses
no random numbers, so the noise draw is the same. The prefix pass is computed twice (once here, once inside
``sample_actions``); that costs time, not correctness.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Steering:
    n_exec: int
    action_dim: int
    state_dim: int
    n_inferences: int = 0


def prefix_features(model, images, img_masks, lang_tokens, lang_masks, state, state_dim: int):
    """(B, hidden + state_dim) float32: the VLM prefix output averaged over valid tokens, and the state."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    embs, pad_masks, att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    outputs, _ = model.vlm_with_expert.forward(attention_mask=att_2d, position_ids=position_ids,
                                               past_key_values=None, inputs_embeds=[embs, None],
                                               use_cache=model.config.use_cache)
    return pooled_features(outputs[0], pad_masks, state, state_dim)


def pooled_features(prefix_out, pad_masks, state, state_dim: int):
    """Mean of ``prefix_out`` (B, L, H) over positions where ``pad_masks`` (B, L) is true, then state[:, :state_dim]."""
    import torch

    m = pad_masks.to(torch.float32)[..., None]
    pooled = (prefix_out.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return torch.cat([pooled, state[:, :state_dim].float()], dim=-1)


def executed_shape(policy) -> tuple[int, int]:
    """(executed positions, real action dimensions) of one chunk: the part of the noise a callback chooses."""
    cfg = policy.config
    return min(cfg.n_action_steps, cfg.chunk_size), cfg.action_feature.shape[0]


def install(policy, choose_noise, state_dim: int = 8) -> Steering:
    """Make ``policy`` ask ``choose_noise(features, drawn)`` for the executed part of its noise at every inference."""
    model = policy.model
    if "sample_actions" in vars(model):
        raise RuntimeError("a sampler wrapper is already installed on this policy")
    st = Steering(*executed_shape(policy), state_dim=state_dim)
    original = model.sample_actions  # the bound class method

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("noise steering draws its own noise")
        if state.shape[0] != 1:
            raise ValueError(f"noise steering expects one observation per call, got batch {state.shape[0]}")
        noise = model.sample_noise((1, model.config.chunk_size, model.config.max_action_dim), state.device)
        feats = prefix_features(model, images, img_masks, lang_tokens, lang_masks, state, st.state_dim)
        chosen = choose_noise(feats, noise[0, : st.n_exec, : st.action_dim])
        if chosen is not None:
            if tuple(chosen.shape) != (st.n_exec, st.action_dim):
                raise ValueError(f"choose_noise returned shape {tuple(chosen.shape)}, "
                                 f"expected {(st.n_exec, st.action_dim)}")
            noise = noise.clone()
            noise[0, : st.n_exec, : st.action_dim] = chosen.to(device=noise.device, dtype=noise.dtype)
        st.n_inferences += 1
        return original(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)

    model.sample_actions = sample_actions
    return st


def install_repeated(policy, choose_noise, state_dim: int = 8) -> Steering:
    """Step 5b: ``choose_noise(features)`` returns one ``max_action_dim`` vector per inference, repeated along the
    chunk, or ``None`` to keep the base's own sampling. Unlike ``install``, the base draws no noise of its own when a
    vector is chosen."""
    import torch

    model = policy.model
    if "sample_actions" in vars(model):
        raise RuntimeError("a sampler wrapper is already installed on this policy")
    st = Steering(*executed_shape(policy), state_dim=state_dim)
    original = model.sample_actions  # the bound class method
    size, dim = model.config.chunk_size, model.config.max_action_dim

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("noise steering draws its own noise")
        if state.shape[0] != 1:
            raise ValueError(f"noise steering expects one observation per call, got batch {state.shape[0]}")
        feats = prefix_features(model, images, img_masks, lang_tokens, lang_masks, state, st.state_dim)
        chosen = choose_noise(feats)
        if chosen is not None:
            if tuple(chosen.shape) != (dim,):
                raise ValueError(f"choose_noise returned shape {tuple(chosen.shape)}, expected {(dim,)}")
            noise = chosen.to(device=state.device, dtype=torch.float32).reshape(1, 1, dim).repeat(1, size, 1)
        st.n_inferences += 1
        return original(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs)

    model.sample_actions = sample_actions
    return st


def uninstall(policy) -> None:
    """Restore the policy's own sampler (no-op if nothing is installed)."""
    vars(policy.model).pop("sample_actions", None)
