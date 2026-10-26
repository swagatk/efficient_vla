"""Step 2: training-free selection among N noise samples per action chunk (README.md).

At each inference the frozen SmolVLA generates N chunks from N noise samples in one batch, and a score picks
the chunk to execute. Scores, none trained, all computed on the executed part of the chunk (the first
``n_exec`` actions) and the real action dimensions (``action_dim``; SmolVLA pads to 32), in the policy's
normalised action space:

- ``accel``: denoising acceleration, the summed squared change of the velocity between consecutive Euler steps
  (a curvature measure in the spirit of arXiv:2607.27933); the smoothest flow is picked.
- ``medoid``: the chunk with the smallest summed distance to the other chunks (agreement between samples).
- ``first``: always sample 0. Same compute as the others, no selection: a control for the batching itself.

``install`` wraps ``policy.model.sample_actions`` on that one policy instance, so LeRobot's own sampler,
un-padding, action queue and ``n_action_steps`` handling run unchanged; ``uninstall`` restores the base policy.
The score functions take tensors only and need no checkpoint; torch is imported inside them, so
``parse_policy_label`` also works under ``python -S``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCORES = ("first", "accel", "medoid")


def acceleration_score(velocities: list[torch.Tensor], n_exec: int, action_dim: int) -> torch.Tensor:
    """(N,) summed squared velocity change between consecutive denoising steps; lower is smoother."""
    import torch

    if len(velocities) < 2:
        raise ValueError("need at least two denoising steps for an acceleration")
    v = torch.stack([x[:, :n_exec, :action_dim].float() for x in velocities])  # (K, N, n_exec, d)
    return (v[1:] - v[:-1]).pow(2).sum(dim=(0, 2, 3))


def medoid_score(chunks: torch.Tensor, n_exec: int, action_dim: int) -> torch.Tensor:
    """(N,) summed Euclidean distance from each chunk to the others; lower is more central."""
    import torch

    x = chunks[:, :n_exec, :action_dim].float().flatten(1)
    return torch.cdist(x[None], x[None])[0].sum(dim=1)


def select_index(score: str, chunks: torch.Tensor, velocities: list[torch.Tensor], n_exec: int,
                 action_dim: int) -> int:
    if score == "first":
        return 0
    if score == "accel":
        return int(acceleration_score(velocities, n_exec, action_dim).argmin())
    if score == "medoid":
        return int(medoid_score(chunks, n_exec, action_dim).argmin())
    raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")


def _repeat(x, n):
    if isinstance(x, (list, tuple)):
        return type(x)(_repeat(t, n) for t in x)
    if x.shape[0] != 1:
        raise ValueError(f"noise selection expects one observation per call, got batch {x.shape[0]}")
    return x.expand(n, *x.shape[1:]).contiguous()


@dataclass
class Selector:
    """What ``install`` puts in place, plus a log of the chosen index per inference."""
    n_samples: int
    score: str
    n_exec: int
    action_dim: int
    chosen: list[int] = field(default_factory=list)


def install(policy, n_samples: int, score: str) -> Selector:
    """Make ``policy`` sample ``n_samples`` chunks per inference and execute the one ``score`` picks."""
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    model = policy.model
    if "sample_actions" in vars(model):
        raise RuntimeError("a selector is already installed on this policy")
    cfg = policy.config
    sel = Selector(n_samples, score, n_exec=min(cfg.n_action_steps, cfg.chunk_size),
                   action_dim=cfg.action_feature.shape[0])
    original = model.sample_actions  # the bound class method

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if noise is not None:
            raise ValueError("noise selection draws its own noise")
        n = sel.n_samples
        noise = model.sample_noise((n, model.config.chunk_size, model.config.max_action_dim), state.device)
        velocities = []
        step = model.denoise_step

        def recording_step(**kw):
            v = step(**kw)
            velocities.append(v)
            return v

        model.denoise_step = recording_step
        try:
            chunks = original(_repeat(images, n), _repeat(img_masks, n), _repeat(lang_tokens, n),
                              _repeat(lang_masks, n), _repeat(state, n), noise=noise, **kwargs)
        finally:
            del model.denoise_step
        i = select_index(sel.score, chunks, velocities, sel.n_exec, sel.action_dim)
        sel.chosen.append(i)
        return chunks[i: i + 1]

    model.sample_actions = sample_actions
    return sel


def uninstall(policy) -> None:
    """Restore the policy's own sampler (no-op if no selector is installed)."""
    vars(policy.model).pop("sample_actions", None)


def parse_policy_label(label: str) -> tuple[int, str] | None:
    """'base' -> None; '<score>_n<N>' (e.g. 'medoid_n8') -> (N, score)."""
    if label == "base":
        return None
    score, _, n = label.rpartition("_n")
    if score not in SCORES or not n.isdigit() or int(n) < 1:
        raise ValueError(f"policy label must be 'base' or '<score>_n<N>' with score in {SCORES}, got {label!r}")
    return int(n), score
