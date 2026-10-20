"""LoRA adapters on SmolVLA's action expert (``policy.model.vlm_with_expert.lm_expert``).

Extracted on second use from ``supervised_task_specific_lora_adapters/train_supervised_lora.py`` (whose
evaluation is retired, see its README). Differences from that script: only the LoRA matrices train by default
(the action read-out heads stay frozen unless asked for), so a freshly initialised adapter reproduces the base
policy exactly (PEFT initialises ``lora_B`` to zero); and nothing here replaces LeRobot's ``apply_rope``.

The functions take any ``nn.Module`` with named ``nn.Linear`` sub-modules, so tests run on small random
modules without the checkpoint.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

ATTN_TARGETS = ("q_proj", "v_proj")
ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ACTION_HEADS = ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
HEADS_FILE = "action_heads.pt"


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=lambda: ALL_TARGETS)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


def resolve_targets(spec: str) -> tuple[str, ...]:
    """'attn' -> q/v projections, 'all' -> all seven projections, else a comma-separated list of names."""
    if spec == "attn":
        return ATTN_TARGETS
    if spec == "all":
        return ALL_TARGETS
    names = tuple(s.strip() for s in spec.split(",") if s.strip())
    if not names:
        raise ValueError(f"no LoRA target modules in {spec!r}")
    return names


def add_lora(module: nn.Module, settings: LoraSettings) -> nn.Module:
    """Wrap ``module`` with fresh LoRA adapters; returns the PEFT model to put in the module's place.

    Every original parameter is frozen; only ``lora_`` parameters require gradients.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=settings.r, lora_alpha=settings.alpha, lora_dropout=settings.dropout,
                        target_modules=list(settings.target_modules), bias="none")
    peft_module = get_peft_model(module, config)
    for name, p in peft_module.named_parameters():
        p.requires_grad_("lora_" in name)
    return peft_module


def load_lora(module: nn.Module, adapter_dir: str | os.PathLike, trainable: bool = False) -> nn.Module:
    """Wrap ``module`` with the adapter saved in ``adapter_dir`` by ``save_lora``."""
    from peft import PeftModel

    peft_module = PeftModel.from_pretrained(module, str(adapter_dir), is_trainable=trainable)
    for name, p in peft_module.named_parameters():
        p.requires_grad_(trainable and "lora_" in name)
    return peft_module


def save_lora(peft_module: nn.Module, adapter_dir: str | os.PathLike, heads: nn.Module | None = None) -> None:
    """Save the adapter (and, if given, the state of the trained action heads) to ``adapter_dir``."""
    os.makedirs(adapter_dir, exist_ok=True)
    peft_module.save_pretrained(str(adapter_dir))
    if heads is not None:
        torch.save({name: getattr(heads, name).state_dict() for name in ACTION_HEADS if hasattr(heads, name)},
                   os.path.join(adapter_dir, HEADS_FILE))


def load_heads(model: nn.Module, adapter_dir: str | os.PathLike) -> bool:
    """Load action-head weights saved next to an adapter into ``model``; False if none were saved."""
    path = os.path.join(adapter_dir, HEADS_FILE)
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, sd in state.items():
        getattr(model, name).load_state_dict(sd)
    return True


def set_heads_trainable(model: nn.Module, trainable: bool) -> list[nn.Parameter]:
    """Set ``requires_grad`` on the action read-out heads of a SmolVLA ``VLAFlowMatching`` model."""
    params = []
    for name in ACTION_HEADS:
        for p in getattr(model, name).parameters():
            p.requires_grad_(trainable)
            params.append(p)
    return params


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]
