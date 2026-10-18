"""SAC residual head, replay buffer and update step (no environment, base policy or checkpoint I/O here).

The head looks at the base policy's own view (rotated agentview image, 8-dimensional state) and at the base
action, and outputs a correction ``delta`` in [-1, 1]^7 (tanh-squashed Gaussian). The training script adds it
to the un-normalised base action with ``vla_core.residual.apply_residual``, so its largest effect is
``max_residual`` in env action units. The critics score (observation, base action, delta).
"""
from __future__ import annotations

import ast
import os
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "sac_residual_env_units_v2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class HeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    image_size: int = 128
    pretrained_vision: bool = True  # ImageNet ResNet-18; False in tests (no download)
    init_log_std: float = -1.0


def _mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))


class SACResidualHead(nn.Module):
    def __init__(self, cfg: HeadConfig, state_mean=None, state_std=None):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.cfg = cfg
        self.vision_encoder = resnet18(weights=ResNet18_Weights.DEFAULT if cfg.pretrained_vision else None)
        self.vision_encoder.fc = nn.Identity()
        self.state_encoder = nn.Sequential(nn.Linear(cfg.state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        feat_dim = 512 + 64 + cfg.action_dim  # image, state, base action
        self.actor_net = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.actor_mean = nn.Linear(256, cfg.action_dim)
        self.actor_log_std = nn.Linear(256, cfg.action_dim)
        self.q1 = _mlp(feat_dim + cfg.action_dim, 1)
        self.q2 = _mlp(feat_dim + cfg.action_dim, 1)
        # Zero mean at the start: the deterministic policy is exactly the base policy.
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.zeros_(self.actor_log_std.weight)
        nn.init.constant_(self.actor_log_std.bias, cfg.init_log_std)
        mean = torch.zeros(cfg.state_dim) if state_mean is None else torch.as_tensor(state_mean, dtype=torch.float32)
        std = torch.ones(cfg.state_dim) if state_std is None else torch.as_tensor(state_std, dtype=torch.float32)
        self.register_buffer("state_mean", mean.clone())
        self.register_buffer("state_std", std.clamp_min(1e-6).clone())
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.train()

    def train(self, mode: bool = True):
        # Batch-norm statistics stay at their initial (ImageNet) values: training mode would update them from
        # replay minibatches and act on single observations with different normalisation.
        super().train(mode)
        for m in self.vision_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def features(self, image_uint8: torch.Tensor, state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        """image_uint8 (B, 3, S, S) uint8, state (B, 8) raw, base_action (B, 7) env units -> (B, 583)."""
        img = (image_uint8.float() / 255.0 - self.img_mean) / self.img_std
        state = (state.float() - self.state_mean) / self.state_std
        return torch.cat([self.vision_encoder(img), self.state_encoder(state), base_action.float()], dim=-1)

    def sample(self, feats: torch.Tensor, deterministic: bool = False):
        """(delta in [-1, 1], log-prob or None). Deterministic: tanh of the mean."""
        h = self.actor_net(feats)
        mean = self.actor_mean(h)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.actor_log_std(h).clamp(-20.0, 2.0)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        delta = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1 - delta.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return delta, log_prob

    def q_values(self, feats: torch.Tensor, delta: torch.Tensor):
        x = torch.cat([feats, delta], dim=-1)
        return self.q1(x), self.q2(x)


def image_for_head(image_float: torch.Tensor, size: int) -> torch.Tensor:
    """Env-processed image (B, 3, H, W) in [0, 1] -> (B, 3, size, size) uint8 on CPU, for the head and replay."""
    img = image_float.detach().float().cpu()
    if img.shape[-1] != size or img.shape[-2] != size:
        img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (img.clamp(0, 1) * 255.0).round().to(torch.uint8)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions in preallocated CPU arrays.

    Memory is about ``2 * capacity * 3 * image_size**2`` bytes (2.0 GB for 20,000 transitions at 128 px).
    Transitions of successful episodes are flagged so a fraction of each minibatch can be drawn from them.
    """

    def __init__(self, capacity: int, image_size: int, state_dim: int = 8, action_dim: int = 7, seed: int = 0):
        self.capacity, self.size, self.pos = int(capacity), 0, 0
        shape = (self.capacity, 3, image_size, image_size)
        self.image = np.zeros(shape, dtype=np.uint8)
        self.next_image = np.zeros(shape, dtype=np.uint8)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.base_action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.next_base_action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.delta = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminal = np.zeros((self.capacity, 1), dtype=np.float32)
        self.from_success = np.zeros(self.capacity, dtype=bool)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, image, state, base_action, delta, reward, next_image, next_state, next_base_action, terminal) -> int:
        i = self.pos
        self.image[i], self.next_image[i] = image, next_image
        self.state[i], self.next_state[i] = state, next_state
        self.base_action[i], self.next_base_action[i] = base_action, next_base_action
        self.delta[i], self.reward[i], self.terminal[i] = delta, reward, float(terminal)
        self.from_success[i] = False
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def mark_success(self, indices) -> None:
        """Flag the transitions of an episode that ended in success (indices as returned by ``add``)."""
        self.from_success[np.asarray(list(indices), dtype=np.int64)] = True

    ARRAYS = ("image", "next_image", "state", "next_state", "base_action", "next_base_action", "delta", "reward",
              "terminal", "from_success")

    def save(self, path) -> None:
        """Write the filled part of the buffer (uncompressed .npz; images dominate the size)."""
        n = self.size
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, size=n, pos=self.pos, capacity=self.capacity,
                 rng=np.frombuffer(repr(self.rng.bit_generator.state).encode(), dtype=np.uint8),
                 **{k: getattr(self, k)[:n] for k in self.ARRAYS})
        os.replace(tmp, path)

    def load(self, path) -> None:
        """Restore a buffer written by ``save``; the capacity must not be smaller than the saved size."""
        with np.load(path) as z:
            n = int(z["size"])
            if n > self.capacity:
                raise ValueError(f"saved buffer has {n} transitions, capacity is {self.capacity}")
            for k in self.ARRAYS:
                getattr(self, k)[:n] = z[k]
            self.size = n
            self.pos = int(z["pos"]) if int(z["capacity"]) == self.capacity else n % self.capacity
            self.rng.bit_generator.state = ast.literal_eval(bytes(z["rng"]).decode())

    def n_success(self) -> int:
        return int(self.from_success[: self.size].sum())

    def sample(self, batch_size: int, success_fraction: float = 0.0, device="cpu") -> dict:
        n_succ = min(int(batch_size * float(success_fraction)), self.n_success())
        idx = self.rng.integers(0, self.size, size=batch_size - n_succ)
        if n_succ:
            succ = np.flatnonzero(self.from_success[: self.size])
            idx = np.concatenate([idx, self.rng.choice(succ, size=n_succ)])
        t = lambda a: torch.from_numpy(a[idx]).to(device)
        return {"image": t(self.image), "state": t(self.state), "base_action": t(self.base_action),
                "delta": t(self.delta), "reward": t(self.reward), "next_image": t(self.next_image),
                "next_state": t(self.next_state), "next_base_action": t(self.next_base_action),
                "terminal": t(self.terminal)}


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.001
    auto_alpha: bool = False
    target_entropy: float | None = None
    min_alpha: float = 1e-5
    max_alpha: float = 0.2
    action_l2_penalty_coef: float = 0.01


class SACLearner:
    """Twin-Q SAC on the residual head. The critic loss trains the encoders; the actor sees detached features."""

    def __init__(self, head: SACResidualHead, cfg: SACConfig, lr: float = 1e-4, alpha_lr: float = 1e-4):
        self.head, self.cfg = head, cfg
        self.target = deepcopy(head)
        for p in self.target.parameters():
            p.requires_grad_(False)
        actor_modules = (head.actor_net, head.actor_mean, head.actor_log_std)
        critic_modules = (head.vision_encoder, head.state_encoder, head.q1, head.q2)
        self.actor_opt = torch.optim.Adam([p for m in actor_modules for p in m.parameters()], lr=lr)
        self.critic_opt = torch.optim.Adam([p for m in critic_modules for p in m.parameters()], lr=lr)
        self.target_entropy = -float(head.cfg.action_dim) if cfg.target_entropy is None else cfg.target_entropy
        device = next(head.parameters()).device
        self.log_alpha = torch.tensor(float(np.log(max(cfg.alpha, 1e-8))), device=device, requires_grad=cfg.auto_alpha)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr) if cfg.auto_alpha else None

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().clamp(self.cfg.min_alpha, self.cfg.max_alpha))

    def update(self, b: dict, update_actor: bool = True) -> dict:
        """One SAC step on minibatch ``b``. ``update_actor=False`` trains only the critics (and moves the targets),
        which is how the warm-up fits Q to the base policy's own behaviour before the actor may change it."""
        head, target, cfg = self.head, self.target, self.cfg
        alpha = self.alpha
        with torch.no_grad():
            next_feats_actor = head.features(b["next_image"], b["next_state"], b["next_base_action"])
            next_delta, next_logp = head.sample(next_feats_actor)
            next_feats = target.features(b["next_image"], b["next_state"], b["next_base_action"])
            tq1, tq2 = target.q_values(next_feats, next_delta)
            target_q = b["reward"] + (1 - b["terminal"]) * cfg.gamma * (torch.min(tq1, tq2) - alpha * next_logp)

        feats = head.features(b["image"], b["state"], b["base_action"])
        q1, q2 = head.q_values(feats, b["delta"])
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        out = {"critic_loss": critic_loss.item(), "alpha": alpha, "q_mean": q1.mean().item()}

        if update_actor:
            feats = feats.detach()
            delta, logp = head.sample(feats)
            q_new = torch.min(*head.q_values(feats, delta))
            l2 = cfg.action_l2_penalty_coef * delta.pow(2).sum(dim=-1).mean()
            actor_loss = (alpha * logp - q_new).mean() + l2
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            # The actor step also put gradients on the critics; clear them so they never leak into a critic step.
            head.q1.zero_grad(set_to_none=True)
            head.q2.zero_grad(set_to_none=True)
            out.update(actor_loss=actor_loss.item(), entropy=-logp.mean().item())
            if self.alpha_opt is not None:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(float(np.log(cfg.min_alpha)), float(np.log(cfg.max_alpha)))
                out["alpha_loss"] = alpha_loss.item()
        soft_update(target, head, cfg.tau)
        return out

    def state_dict(self) -> dict:
        return {"format": CHECKPOINT_FORMAT, "head_config": asdict(self.head.cfg), "head": self.head.state_dict(),
                "target": self.target.state_dict(), "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(), "log_alpha": self.log_alpha.item(),
                "alpha_opt": None if self.alpha_opt is None else self.alpha_opt.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        check_checkpoint_format(sd)
        self.head.load_state_dict(sd["head"])
        self.target.load_state_dict(sd["target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        with torch.no_grad():
            self.log_alpha.fill_(sd["log_alpha"])
        if self.alpha_opt is not None and sd.get("alpha_opt") is not None:
            self.alpha_opt.load_state_dict(sd["alpha_opt"])


def check_checkpoint_format(sd: dict) -> None:
    if sd.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format {sd.get('format')!r} is not {CHECKPOINT_FORMAT!r}. Checkpoints from the pre-fix "
            "train_online_sac.py added the residual to normalised actions and cannot be reused.")


def head_from_checkpoint(sd: dict, device="cpu") -> SACResidualHead:
    """The trained head of a checkpoint written by ``SACLearner.state_dict``."""
    check_checkpoint_format(sd)
    cfg = HeadConfig(**{**sd["head_config"], "pretrained_vision": False})
    head = SACResidualHead(cfg).to(device)
    head.load_state_dict(sd["head"])
    return head.eval()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1.0 - tau).add_(s, alpha=tau)
    for t, s in zip(target.buffers(), source.buffers()):
        t.copy_(s)
