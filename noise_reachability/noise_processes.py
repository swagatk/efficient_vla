"""Noise processes over an episode's inferences, for the mechanism ablation M1 (stage-2 plan; README step 17 when it is
designed). Each process gives the noise of every inference of one episode, deterministically from its noise id:

- ``ar_repeat`` (``rho`` in [0, 1]): one 32-vector per inference, repeated along the chunk as ``repeat`` is, following
  u_0 = the ``repeat`` vector of the id and u_j = rho * u_{j-1} + sqrt(1 - rho^2) * xi_j, xi_j ~ N(0, I) from the id's
  own innovation stream. Every u_j is N(0, I) and corr(u_j, u_{j-1}) = rho. rho = 1 is exactly ``repeat`` (constant);
  rho = 0 is a ``repeat`` noise redrawn at every inference (step 5b's control);
- ``scaled`` (``sigma`` > 0): a fresh full (chunk x dim) noise at every inference times sigma: sampling at a lower
  temperature, which lowers noise diversity without holding the noise constant;
- ``zero``: the zero noise at every inference (the mode-seeking reference).

``deep_noise_search.py --family ar_repeat --rho 0.5`` (or ``scaled --sigma 0.5``, ``zero``) runs them. The families of
steps 5a-16 (sampled, full, repeat) are untouched: they go through ``install_fixed_noise`` as before.
"""
from __future__ import annotations

import math

PROCESS_FAMILIES = ("ar_repeat", "scaled", "zero")
AR_SEED_BASE = 40_000_000       # innovations of ar_repeat (repeat 20,000,000, jitter 30,000,000 in deep_noise_search)
SCALED_SEED_BASE = 50_000_000


def check_params(family: str, rho: float | None, sigma: float | None) -> None:
    if family == "ar_repeat":
        if rho is None or not 0.0 <= rho <= 1.0 or sigma is not None:
            raise ValueError(f"ar_repeat needs --rho in [0, 1] and no --sigma (rho {rho}, sigma {sigma})")
    elif family == "scaled":
        if sigma is None or not sigma > 0 or not math.isfinite(sigma) or rho is not None:
            raise ValueError(f"scaled needs --sigma > 0 and no --rho (rho {rho}, sigma {sigma})")
    elif rho is not None or sigma is not None:
        raise ValueError(f"{family} takes neither --rho nor --sigma")


def family_label(family: str, rho: float | None = None, sigma: float | None = None) -> str:
    """The name used in result and record files: 'repeat', 'ar_repeat_rho0.5', 'scaled_sigma0.5', 'zero'."""
    check_params(family, rho, sigma)
    if family == "ar_repeat":
        return f"ar_repeat_rho{rho:g}"
    if family == "scaled":
        return f"scaled_sigma{sigma:g}"
    return family


class NoiseProcess:
    """The noises of one episode's inferences, in order; ``calls`` counts how many were handed out."""

    def __init__(self, family: str, noise_id: int, chunk_size: int, max_action_dim: int, rho: float | None = None,
                 sigma: float | None = None):
        import torch

        if family not in PROCESS_FAMILIES:
            raise ValueError(f"{family!r} is not a noise process")
        check_params(family, rho, sigma)
        self.family, self.noise_id, self.rho, self.sigma = family, int(noise_id), rho, sigma
        self.shape = (1, chunk_size, max_action_dim)
        self.calls = 0
        self._u = None
        if family == "ar_repeat":
            from deep_noise_search import make_noise

            self._u = make_noise("repeat", noise_id, chunk_size, max_action_dim)[0, 0].clone()
            self._gen = torch.Generator(device="cpu").manual_seed(AR_SEED_BASE + self.noise_id)
        elif family == "scaled":
            self._gen = torch.Generator(device="cpu").manual_seed(SCALED_SEED_BASE + self.noise_id)

    def next(self):
        import torch

        c, d = self.shape[1], self.shape[2]
        if self.family == "zero":
            z = torch.zeros(self.shape, dtype=torch.float32)
        elif self.family == "scaled":
            z = self.sigma * torch.randn(self.shape, generator=self._gen, dtype=torch.float32)
        else:
            if self.calls > 0:
                xi = torch.randn(d, generator=self._gen, dtype=torch.float32)
                self._u = self.rho * self._u + math.sqrt(1.0 - self.rho ** 2) * xi   # rho = 1: exactly u (1u + 0xi)
            z = self._u.reshape(1, 1, d).expand(1, c, d).clone()
        self.calls += 1
        return z


def install_noise_process(policy, process: NoiseProcess) -> None:
    """Make inference j of ``policy`` start from ``process``'s j-th noise, by wrapping that instance's
    ``sample_actions`` as deep_noise_search.install_fixed_noise does; uninstall with uninstall_fixed_noise."""
    model = policy.model
    if "sample_actions" in vars(model):
        raise RuntimeError("a sampler wrapper is already installed on this policy")
    shape = (1, model.config.chunk_size, model.config.max_action_dim)
    if tuple(process.shape) != shape:
        raise ValueError(f"process has shape {tuple(process.shape)}, expected {shape}")
    original = model.sample_actions

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("the noise process supplies the noise")
        if state.shape[0] != 1:
            raise ValueError(f"a noise process expects one observation per call, got batch {state.shape[0]}")
        return original(images, img_masks, lang_tokens, lang_masks, state, noise=process.next().to(state.device),
                        **kwargs)

    model.sample_actions = sample_actions
