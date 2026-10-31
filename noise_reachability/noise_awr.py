"""Step 5b: NSL-AWR after arXiv:2610.13943, a success-filtered AWR on SmolVLA's initial noise (README.md, step 5b).

One 32-dimensional Gaussian per inference, repeated along the chunk by ``noise_steering.install_repeated``. Nothing
here touches SmolVLA, the simulator or the GPU at import time: the actor, the value model, the episode returns and the
replay work on plain feature vectors, so the tests run on CPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "noise_awr_v1"
LOG_STD_RANGE = (-20.0, 2.0)
REPLAY_ARRAYS = ("feats", "z", "ret", "success")


@dataclass
class NoiseAWRConfig:
    feature_dim: int
    noise_dim: int = 32
    hidden: int = 512
    layers: int = 4
    value_bins: int = 16
    actor_lr: float = 1e-5
    value_lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta: float = 0.5            # AWR temperature
    max_weight: float = 20.0
    gamma: float = 0.99          # per decision
    batch_size: int = 2048
    success_fraction: float = 0.5


def hidden_stack(in_dim: int, hidden: int, layers: int, layer_norm: bool) -> nn.Sequential:
    mods, d = [], in_dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if layer_norm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(nn.ReLU())
        d = hidden
    return nn.Sequential(*mods)


class GaussianNoiseActor(nn.Module):
    """Diagonal Gaussian with zero-initialised heads: untrained, it is N(0, I) whatever its input."""

    def __init__(self, cfg: NoiseAWRConfig):
        super().__init__()
        self.body = hidden_stack(cfg.feature_dim, cfg.hidden, cfg.layers, layer_norm=True)
        self.mean = nn.Linear(cfg.hidden, cfg.noise_dim)
        self.log_std = nn.Linear(cfg.hidden, cfg.noise_dim)
        for head in (self.mean, self.log_std):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(feats)
        return self.mean(h), self.log_std(h).clamp(*LOG_STD_RANGE)

    def log_prob(self, feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """log pi(z | s) averaged over noise dimensions, shape (B,)."""
        mean, log_std = self(feats)
        return torch.distributions.Normal(mean, log_std.exp()).log_prob(z).mean(dim=-1)

    @torch.no_grad()
    def sample(self, feats: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(feats)
        return mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)


class BinnedValue(nn.Module):
    """V(s) as a categorical distribution on ``value_bins`` points of [0, 1]."""

    def __init__(self, cfg: NoiseAWRConfig):
        super().__init__()
        self.net = nn.Sequential(hidden_stack(cfg.feature_dim, cfg.hidden, cfg.layers, layer_norm=False),
                                 nn.Linear(cfg.hidden, cfg.value_bins))
        self.register_buffer("support", torch.linspace(0.0, 1.0, cfg.value_bins))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    def value(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.softmax(dim=-1) * self.support).sum(dim=-1)


def two_hot(values: torch.Tensor, bins: int) -> torch.Tensor:
    """(B, bins) probabilities on [0, 1] whose expectation is ``values`` (clamped to [0, 1])."""
    x = values.clamp(0.0, 1.0) * (bins - 1)
    low = x.floor().long().clamp(max=bins - 2)
    frac = x - low.float()
    out = torch.zeros(*values.shape, bins, device=values.device, dtype=torch.float32)
    out.scatter_(-1, low[..., None], (1.0 - frac)[..., None])
    out.scatter_(-1, (low + 1)[..., None], frac[..., None])
    return out


def awr_weights(advantage: torch.Tensor, success: torch.Tensor, beta: float, max_weight: float) -> torch.Tensor:
    """Equation 10 of the paper: ``Y min(exp(A / beta), w_max)``."""
    return success * torch.exp(advantage / beta).clamp(max=max_weight)


def episode_returns(decisions: int, success: bool, gamma: float) -> np.ndarray:
    """Return of each of an episode's decisions: gamma^(decisions left after it) on success, 0 on failure."""
    if decisions <= 0:
        raise ValueError(f"an episode has at least one decision, got {decisions}")
    if not success:
        return np.zeros(decisions, dtype=np.float32)
    return (gamma ** np.arange(decisions - 1, -1, -1)).astype(np.float32)


class NoiseAWR:
    def __init__(self, cfg: NoiseAWRConfig, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = GaussianNoiseActor(cfg).to(self.device)
        self.value = BinnedValue(cfg).to(self.device)
        opt = dict(betas=tuple(cfg.betas), eps=1e-8, weight_decay=cfg.weight_decay)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr, **opt)
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=cfg.value_lr, **opt)

    def update(self, b: dict, update_actor: bool) -> dict:
        logits = self.value(b["feats"])
        value_loss = F.cross_entropy(logits, two_hot(b["ret"], self.cfg.value_bins))
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_clip)
        self.value_opt.step()
        out = {"value_loss": value_loss.item()}
        if update_actor and bool(b["success"].any()):
            with torch.no_grad():
                adv = b["ret"] - self.value.value(self.value(b["feats"]))
                w = awr_weights(adv, b["success"], self.cfg.beta, self.cfg.max_weight)
            actor_loss = -(w * self.actor.log_prob(b["feats"], b["z"])).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.actor_opt.step()
            out.update(actor_loss=actor_loss.item(), weight_mean=w[b["success"] > 0].mean().item())
        return out

    def state_dict(self) -> dict:
        return {"format": CHECKPOINT_FORMAT, "config": asdict(self.cfg), "actor": self.actor.state_dict(),
                "value": self.value.state_dict(), "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict()}

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu") -> "NoiseAWR":
        if sd.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"checkpoint format {sd.get('format')!r} is not {CHECKPOINT_FORMAT!r}")
        learner = cls(NoiseAWRConfig(**sd["config"]), device)
        learner.actor.load_state_dict(sd["actor"])
        learner.value.load_state_dict(sd["value"])
        learner.actor_opt.load_state_dict(sd["actor_opt"])
        learner.value_opt.load_state_dict(sd["value_opt"])
        return learner


class Replay:
    """Every decision of a run: {feats, z, ret, success} arrays of equal length, sampled half from successes."""

    def __init__(self, arrays: dict, seed: int):
        if set(arrays) != set(REPLAY_ARRAYS) or len({len(v) for v in arrays.values()}) != 1:
            raise ValueError(f"replay needs equal-length arrays {REPLAY_ARRAYS}, got {sorted(arrays)}")
        self.arrays = arrays
        self.succ = np.flatnonzero(arrays["success"] > 0)
        self.fail = np.flatnonzero(arrays["success"] <= 0)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.arrays["z"])

    def sample(self, batch_size: int, success_fraction: float, device="cpu") -> dict:
        n_succ = round(batch_size * success_fraction) if len(self.succ) and len(self.fail) else None
        if n_succ is None:
            idx = self.rng.integers(0, len(self), size=batch_size)
        else:
            idx = np.concatenate([self.rng.choice(self.succ, n_succ), self.rng.choice(self.fail, batch_size - n_succ)])
        return {k: torch.from_numpy(self.arrays[k][idx].astype(np.float32)).to(device) for k in REPLAY_ARRAYS}
