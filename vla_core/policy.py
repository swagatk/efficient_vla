"""Frozen SmolVLA base policy with lerobot-eval's per-step observation and action processing.

`BasePolicy.act` runs exactly the chain of ``lerobot_eval.rollout`` for one environment: LeRobot's
``preprocess_observation`` -> task string -> LIBERO env processor (180-degree image rotation, 8-dimensional
eef-pos / axis-angle / gripper state) -> policy preprocessor (renaming, normalisation, tokenisation, device)
-> ``select_action`` under ``torch.inference_mode`` -> policy postprocessor (un-normalisation) -> env
postprocessor. The returned action is in the environment's action units, which is where a residual must be
added (see ``vla_core.residual``).

The base policy is not an ``nn.Module`` attribute of any trainable model, so calling ``.eval()`` or
``.train()`` on a residual head never recurses into SmolVLA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_core.envs import add_batch_dim

# The LIBERO env key of the third-person camera; after the env processor it is the first image the policy sees.
AGENTVIEW_KEY = "observation.images.image"


@dataclass
class BasePolicy:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any

    def reset(self) -> None:
        """Clear the action queue; call at the start of every episode, as lerobot-eval does."""
        self.policy.reset()

    def act(self, observation: dict, task: str) -> tuple[np.ndarray, dict]:
        """Base action for one LiberoEnv observation, in env units, and the env-processed observation.

        The env-processed observation (rotated agentview image in [0, 1], raw 8-dimensional state) is what a
        residual head should look at, so it sees the same pixels as the base policy.
        """
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        obs = preprocess_observation(add_batch_dim(observation))
        obs["task"] = [task]
        env_obs = self.env_preprocessor(obs)
        policy_obs = self.preprocessor(dict(env_obs))
        with torch.inference_mode():
            action = self.policy.select_action(policy_obs)
        action = self.postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        action = action.to("cpu").numpy()
        if action.shape[0] != 1:
            raise ValueError(f"expected one action, got shape {action.shape}")
        return action[0].astype(np.float32), env_obs


def load_base_policy(policy_path: str, env_cfg, device: str = "cuda", n_action_steps: int | None = None,
                     rename_map: dict[str, str] | None = None) -> BasePolicy:
    """Load and freeze a SmolVLA checkpoint the way lerobot-eval does for ``--policy.path``.

    ``n_action_steps`` and ``rename_map`` mean the same as evaluate_all.py's flags of the same name.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = Path(policy_path)
    cfg.device = device
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename_map or None)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename_map or {}}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return BasePolicy(policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor)
