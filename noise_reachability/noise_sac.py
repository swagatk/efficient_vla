"""Step 3: SAC on the initial noise of SmolVLA's flow, DSRL-style (README.md, step 3).

One decision per inference: the actor sees the frozen policy's prefix features and chooses the noise of the
executed positions and real action dimensions (``action_dim`` numbers, as ``noise_scale * tanh(x)``); the
transition reward is 1 on the decision whose chunk reaches success. Nothing here touches SmolVLA, the
simulator or the GPU at import time: actor, critics, replay buffer and the transition bookkeeping work on
plain feature vectors, so the tests run on CPU. ``noise_steering.py`` connects them to the policy.

Unlike ``rl_based_tuning/sac_residual.py`` there is no image encoder to train (the features are the frozen
policy's own), so the replay holds feature vectors and the critic optimiser holds only the Q networks.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "noise_sac_v1"


@dataclass
class NoiseSACConfig:
    feature_dim: int
    action_dim: int = 70             # 10 executed positions x 7 real action dimensions
    noise_scale: float = 2.5         # largest |noise| the actor can choose
    init_std: float = 0.45           # pre-tanh std of the untrained actor: its noise is close to N(0, 1)
    hidden: int = 256
    layers: int = 3
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.01
    target_entropy: float | None = None   # None: -action_dim


def mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [nn.LayerNorm(in_dim)], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class NoiseActor(nn.Module):
    """Tanh-squashed Gaussian on [-1, 1]^action_dim; the executed noise is ``noise_scale`` times its output."""

    def __init__(self, cfg: NoiseSACConfig):
        super().__init__()
        self.cfg = cfg
        self.net = mlp(cfg.feature_dim, 2 * cfg.action_dim, cfg.hidden, cfg.layers)
        last = self.net[-1]
        # The untrained actor ignores its input: mean 0, std init_std before the tanh.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[cfg.action_dim:] = float(np.log(cfg.init_std))

    def forward(self, feats: torch.Tensor, deterministic: bool = False):
        """(u in [-1, 1], log-prob of u or None)."""
        mean, log_std = self.net(feats).chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mean), None
        log_std = log_std.clamp(-10.0, 2.0)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        u = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1 - u.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return u, log_prob


class TwinQ(nn.Module):
    def __init__(self, cfg: NoiseSACConfig):
        super().__init__()
        self.q1 = mlp(cfg.feature_dim + cfg.action_dim, 1, cfg.hidden, cfg.layers)
        self.q2 = mlp(cfg.feature_dim + cfg.action_dim, 1, cfg.hidden, cfg.layers)

    def forward(self, feats, u):
        x = torch.cat([feats, u], dim=-1)
        return self.q1(x), self.q2(x)


def td_target(reward, terminal, gamma, next_q1, next_q2, alpha, next_logp):
    """r + (1 - terminal) * gamma * (min(Q1', Q2') - alpha * log pi(u' | s'))."""
    return reward + (1.0 - terminal) * gamma * (torch.min(next_q1, next_q2) - alpha * next_logp)


def actor_objective(alpha, logp, q1, q2):
    """Mean of alpha * log pi(u | s) - min(Q1, Q2)(s, u), minimised by the actor."""
    return (alpha * logp - torch.min(q1, q2)).mean()


def alpha_objective(log_alpha, logp, target_entropy):
    """Temperature loss: alpha falls while the entropy (-logp) is above the target and rises below it."""
    return -(log_alpha * (logp.detach() + target_entropy)).mean()


class NoiseSAC:
    """Twin-Q SAC with automatic entropy temperature on (features, u) transitions."""

    def __init__(self, cfg: NoiseSACConfig, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = NoiseActor(cfg).to(self.device)
        self.critic = TwinQ(cfg).to(self.device)
        self.critic_target = deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -float(cfg.action_dim) if cfg.target_entropy is None else float(cfg.target_entropy)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp())

    @torch.no_grad()
    def act(self, feats: np.ndarray, deterministic: bool) -> np.ndarray:
        """u in [-1, 1]^action_dim for one feature vector."""
        f = torch.as_tensor(feats, dtype=torch.float32, device=self.device)[None]
        return self.actor(f, deterministic)[0][0].cpu().numpy().astype(np.float32)

    def update(self, b: dict, update_actor: bool = True) -> dict:
        alpha = self.alpha
        with torch.no_grad():
            next_u, next_logp = self.actor(b["next_feats"])
            target = td_target(b["reward"], b["terminal"], self.cfg.gamma, *self.critic_target(b["next_feats"], next_u),
                               alpha, next_logp)
        q1, q2 = self.critic(b["feats"], b["u"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        out = {"critic_loss": critic_loss.item(), "q_mean": q1.mean().item(), "alpha": alpha}
        if update_actor:
            u, logp = self.actor(b["feats"])
            actor_loss = actor_objective(alpha, logp, *self.critic(b["feats"], u))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            alpha_loss = alpha_objective(self.log_alpha, logp, self.target_entropy)
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            out.update(actor_loss=actor_loss.item(), entropy=-logp.mean().item())
        soft_update(self.critic_target, self.critic, self.cfg.tau)
        return out

    def state_dict(self) -> dict:
        return {"format": CHECKPOINT_FORMAT, "config": asdict(self.cfg), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.item(), "alpha_opt": self.alpha_opt.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        check_format(sd)
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        with torch.no_grad():
            self.log_alpha.fill_(sd["log_alpha"])

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu") -> "NoiseSAC":
        check_format(sd)
        learner = cls(NoiseSACConfig(**sd["config"]), device)
        learner.load_state_dict(sd)
        return learner


def check_format(sd: dict) -> None:
    if sd.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format {sd.get('format')!r} is not {CHECKPOINT_FORMAT!r}")


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1.0 - tau).add_(s, alpha=tau)


class ChunkReplay:
    """Append-only buffer of (feats, u, reward, next_feats, terminal) holding every transition of a run (features
    as float16). It is saved in parts: each training process writes only the transitions it added
    (``save_part``), so no file on disk is ever rewritten or deleted, and ``load_parts`` restores them in order."""

    ARRAYS = ("feats", "u", "reward", "next_feats", "terminal")

    def __init__(self, capacity: int, feature_dim: int, action_dim: int, seed: int = 0):
        self.capacity, self.size = int(capacity), 0
        self.feats = np.zeros((self.capacity, feature_dim), dtype=np.float16)
        self.next_feats = np.zeros((self.capacity, feature_dim), dtype=np.float16)
        self.u = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminal = np.zeros((self.capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, t: "Transition") -> None:
        i = self.size
        if i >= self.capacity:
            raise RuntimeError(f"replay is full ({self.capacity} transitions); the run needs a larger --buffer_size")
        self.feats[i], self.u[i], self.reward[i] = t.feats, t.u, t.reward
        self.next_feats[i], self.terminal[i] = t.next_feats, float(t.terminal)
        self.size += 1

    def sample(self, batch_size: int, device="cpu") -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {k: torch.from_numpy(getattr(self, k)[idx].astype(np.float32)).to(device) for k in self.ARRAYS}

    def save_part(self, path, start: int) -> None:
        """Write transitions ``start`` .. ``size - 1`` to ``path`` (through a temporary file)."""
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, start=start, **{k: getattr(self, k)[start: self.size] for k in self.ARRAYS})
        os.replace(tmp, path)

    def load_parts(self, paths) -> None:
        """Append the parts written by ``save_part``, in order; each must start where the previous one ended."""
        for path in paths:
            with np.load(path) as z:
                if int(z["start"]) != self.size:
                    raise ValueError(f"{path} starts at transition {int(z['start'])}, the buffer holds {self.size}")
                n = len(z["u"])
                if self.size + n > self.capacity:
                    raise ValueError(f"parts hold more than the capacity of {self.capacity} transitions")
                for k in self.ARRAYS:
                    getattr(self, k)[self.size: self.size + n] = z[k]
                self.size += n


@dataclass
class Transition:
    feats: np.ndarray
    u: np.ndarray
    reward: float
    next_feats: np.ndarray
    terminal: bool


class EpisodeTransitions:
    """Turns one episode's decisions into transitions.

    ``decide(feats, u)`` at every inference; ``finish(success, bootstrap_feats)`` once at the end. A decision's
    transition ends at the next decision (reward 0, not terminal) or at the episode's end: reward 1 and
    terminal on success, otherwise reward 0 and a bootstrap from ``bootstrap_feats``, the features of the
    final observation (a time limit is not a terminal state).
    """

    def __init__(self):
        self.decisions: list[tuple[np.ndarray, np.ndarray]] = []
        self.transitions: list[Transition] = []
        self.finished = False

    def decide(self, feats: np.ndarray, u: np.ndarray) -> None:
        if self.finished:
            raise RuntimeError("decision after the episode finished")
        if self.decisions:
            prev_f, prev_u = self.decisions[-1]
            self.transitions.append(Transition(prev_f, prev_u, 0.0, feats, False))
        self.decisions.append((np.asarray(feats, dtype=np.float32), np.asarray(u, dtype=np.float32)))

    def finish(self, success: bool, bootstrap_feats: np.ndarray | None = None) -> list[Transition]:
        if self.finished:
            raise RuntimeError("episode finished twice")
        self.finished = True
        if not self.decisions:
            return self.transitions
        last_f, last_u = self.decisions[-1]
        if success:
            self.transitions.append(Transition(last_f, last_u, 1.0, np.zeros_like(last_f), True))
        else:
            if bootstrap_feats is None:
                raise ValueError("a failed episode needs the final observation's features to bootstrap")
            self.transitions.append(Transition(last_f, last_u, 0.0, np.asarray(bootstrap_feats, np.float32), False))
        return self.transitions
