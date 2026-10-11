import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hashlib

import transformers
import sys
sys.path.insert(0, "/home/swagat/lerobot/src")

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
except ImportError:
    pass



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.cond_fc = nn.Linear(cond_dim, dim * 2)
        self.act = nn.GELU()

    def forward(self, x, cond):
        res = x
        x = self.fc1(x)
        # FiLM conditioning
        scale, shift = self.cond_fc(cond).chunk(2, dim=-1)
        x = x * (scale + 1.0) + shift
        x = self.act(x)
        x = self.fc2(x)
        return x + res

class ConditionalDDPMHead(nn.Module):
    def __init__(
        self,
        action_dim=7,
        chunk_size=16,
        cond_dim=2048,
        hidden_dim=256,
        num_layers=4,
        cond_on_base_action=False,
        time_scale=1.0,
        standardize_cond=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_on_base_action = cond_on_base_action
        # Flow time lives in [0, 1]; the sinusoidal embedding only resolves it when scaled up.
        self.time_scale = float(time_scale)
        self.standardize_cond = standardize_cond
        
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        
        # Flattened action projection
        self.action_in = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden_dim),
            nn.GELU()
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        if cond_on_base_action:
            # A residual is only defined relative to the base action sample it corrects.
            self.base_action_proj = nn.Sequential(
                nn.Linear(action_dim * chunk_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        if standardize_cond:
            self.register_buffer("cond_mean", torch.zeros(cond_dim))
            self.register_buffer("cond_std", torch.ones(cond_dim))
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.action_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )

    def set_cond_stats(self, mean, std):
        self.cond_mean.copy_(mean.to(self.cond_mean))
        self.cond_std.copy_(std.to(self.cond_std).clamp_min(1e-6))
        
    def forward(self, x, time, cond, base_action=None):
        """
        x: [B, chunk_size, action_dim] NOISY action trajectory
        time: [B] noise level or timestep
        cond: [B, cond_dim] semantic feature from frozen brain
        base_action: [B, chunk_size, action_dim] base action chunk (required if cond_on_base_action)
        """
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.action_in(x)
        
        if self.standardize_cond:
            cond = (cond - self.cond_mean) / self.cond_std
        t_emb = self.time_emb(time * self.time_scale)
        c_emb = self.cond_proj(cond)
        emb = t_emb + c_emb
        if self.cond_on_base_action:
            if base_action is None:
                raise ValueError("This head was built with cond_on_base_action=True; base_action is required.")
            emb = emb + self.base_action_proj(base_action.reshape(B, -1))
        
        for block in self.blocks:
            x = block(x, emb)
            
        out = self.action_out(x)
        return out.view(B, self.chunk_size, self.action_dim)


def flow_matching_loss(head, semantics, target_actions, base_action=None, delta_l2_weight=0.0, noise=None, timesteps=None):
    """Flow-matching loss of the head on precomputed conditioning. Returns (total, flow, delta_l2)."""
    B = target_actions.shape[0]
    device = target_actions.device
    
    if noise is None:
        noise = torch.randn_like(target_actions)
    if timesteps is None:
        timesteps = torch.rand((B,), device=device)
    
    t_expand = timesteps[:, None, None]
    x_t = t_expand * target_actions + (1 - t_expand) * noise
    target_velocity = target_actions - noise
    
    pred_velocity = head(x_t, timesteps, semantics, base_action=base_action if head.cond_on_base_action else None)

    flow_loss = F.mse_loss(pred_velocity, target_velocity)
    total_loss = flow_loss
    delta_l2_loss = torch.tensor(0.0, device=device)

    if delta_l2_weight > 0.0:
        pred_delta = x_t + (1.0 - t_expand) * pred_velocity
        delta_l2_loss = pred_delta.square().mean()
        total_loss = total_loss + float(delta_l2_weight) * delta_l2_loss
    return total_loss, flow_loss, delta_l2_loss


@torch.no_grad()
def sample_flow(head, semantics, base_action=None, steps=10, return_intermediates=False, noise=None):
    """Euler-integrate the head's flow from noise (t=0) to an action/residual chunk (t=1) in float32."""
    B = semantics.shape[0]
    device_type = "cuda" if semantics.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        semantics = semantics.float()
        if not head.cond_on_base_action:
            base_action = None
        elif base_action is not None:
            base_action = base_action.float()

        x_t = noise if noise is not None else torch.randn((B, head.chunk_size, head.action_dim), device=semantics.device)
        intermediates = [x_t.clone()] if return_intermediates else None

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=semantics.device)
            v_pred = head(x_t, t, semantics, base_action=base_action)
            x_t = x_t + v_pred * dt
            if return_intermediates:
                intermediates.append(x_t.clone())

    if return_intermediates:
        return x_t, intermediates
    return x_t


class HybridFrozenBrainDiffusionHands(nn.Module):
    """
    Implements the 'Frozen Brain, Diffusion Hands' Architecture.
    Strips standard action head, freezes the VLM backbone, and attaches a Diffusion head.
    """
    def set_base_policy_num_steps(self, num_steps: int):
        if not hasattr(self, "base_policy") or self.base_policy is None:
            return
        if hasattr(self.base_policy, "config"):
            self.base_policy.config.num_steps = num_steps
        if hasattr(self.base_policy, "model") and hasattr(self.base_policy.model, "config"):
            self.base_policy.model.config.num_steps = num_steps

    def __init__(
        self,
        base_policy_path,
        action_dim=7,
        chunk_size=16,
        cond_dim=960, # SmolVLM2 500M hidden_size / pooled semantic feature dimension
        diff_hidden_dim=256,
        diff_layers=5,
        device="cuda",
        cond_on_base_action=False,
        time_scale=1.0,
        standardize_cond=False,
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim
        self.head_config = {
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "cond_dim": cond_dim,
            "hidden_dim": diff_hidden_dim,
            "num_layers": diff_layers,
            "cond_on_base_action": cond_on_base_action,
            "time_scale": time_scale,
            "standardize_cond": standardize_cond,
        }
        
        print(f"[HybridDiffusion] Loading frozen base policy from: {base_policy_path}")
        try:
            self.base_policy = SmolVLAPolicy.from_pretrained(base_policy_path)
        except NameError:
            print("SmolVLAPolicy not found. Make sure LeRobot is installed with SmolVLAPolicy support.")
            
        # Keep base policy in bfloat16 to match pretrained SmolVLA checkpoint
        self.base_policy.to(torch.bfloat16)
        self.base_policy.to(self.device)
        self.base_policy.eval()
        # Prevent base_policy.select_action from recursively traversing all submodules on every rollout step
        self.base_policy.eval = lambda *args, **kwargs: self.base_policy
        # Maintain standard 10-step flow matching sampling for base policy
        self.set_base_policy_num_steps(10)
        
        # 1. Strip the standard action head completely by completely freezing the 500M parameters backbone
        # so it acts purely as a semantic feature extractor. 
        for param in self.base_policy.parameters():
            param.requires_grad = False
            
        print("[HybridDiffusion] All base params frozen.")
        
        # 2. Attach continuous Diffusion Policy head
        self.diffusion_head = ConditionalDDPMHead(**self.head_config)
        self.diffusion_head.to(self.device)
        print(f"[HybridDiffusion] Attached highly specialized Diffusion Head with condition dim {cond_dim}.")
        self._lang_token_cache = {}

    def _normalize_text_batch(self, raw_text):
        if raw_text is None:
            return None

        # Tensor payloads can either be token ids or encoded strings from a custom collate.
        if isinstance(raw_text, torch.Tensor):
            if raw_text.dtype in (torch.int32, torch.int64, torch.long):
                return raw_text
            if raw_text.ndim == 0:
                return [str(raw_text.item())]
            return [str(x) for x in raw_text.tolist()]

        if isinstance(raw_text, str):
            return [raw_text]

        if not isinstance(raw_text, (list, tuple)):
            return [str(raw_text)]

        normalized = []
        for item in raw_text:
            if isinstance(item, tuple) and len(item) > 0:
                item = item[0]
            normalized.append(str(item))
        return normalized

    def _prepare_language_tensors(self, batch):
        # Fast path: language is already tokenized.
        if "observation.language.tokens" in batch:
            lang_tokens = batch["observation.language.tokens"]
            if not isinstance(lang_tokens, torch.Tensor):
                lang_tokens = torch.as_tensor(lang_tokens)
            lang_tokens = lang_tokens.to(self.device)

            lang_masks = batch.get("observation.language.attention_mask", None)
            if lang_masks is None:
                lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
            elif not isinstance(lang_masks, torch.Tensor):
                lang_masks = torch.as_tensor(lang_masks)
            lang_masks = lang_masks.to(self.device).bool()

            batch["observation.language.tokens"] = lang_tokens
            batch["observation.language.attention_mask"] = lang_masks
            return lang_tokens, lang_masks

        # Try common string containers used by this repo/dataset stack.
        text_candidates = [
            batch.get("task", None),
            batch.get("language_instruction", None),
            batch.get("observation.language_instruction", None),
            batch.get("instruction", None),
        ]
        texts = None
        for candidate in text_candidates:
            texts = self._normalize_text_batch(candidate)
            if texts is not None:
                break

        if texts is None:
            # Last resort: scan keys for any language/instruction field.
            for key, value in batch.items():
                if "language" in key or "instruction" in key or key == "task":
                    texts = self._normalize_text_batch(value)
                    if texts is not None:
                        break

        if texts is None:
            raise ValueError(
                f"No language tokens or text field found in batch keys: {list(batch.keys())}"
            )

        if not hasattr(self, "_single_string_token_cache"):
            self._single_string_token_cache = {}

        tokens_list = []
        masks_list = []
        processor = self.base_policy.model.vlm_with_expert.processor

        for text in texts:
            if text not in self._single_string_token_cache:
                text_out = processor(text=text, return_tensors="pt", padding=True, truncation=True)
                t_ids = text_out["input_ids"].to(self.device)
                t_mask = text_out["attention_mask"].to(self.device).bool()
                self._single_string_token_cache[text] = (t_ids, t_mask)

            t_ids, t_mask = self._single_string_token_cache[text]
            tokens_list.append(t_ids)
            masks_list.append(t_mask)

        max_len = max(t.shape[1] for t in tokens_list)
        batch_size = len(tokens_list)

        lang_tokens = torch.zeros((batch_size, max_len), dtype=tokens_list[0].dtype, device=self.device)
        lang_masks = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

        for i, (t_ids, t_mask) in enumerate(zip(tokens_list, masks_list)):
            l = t_ids.shape[1]
            lang_tokens[i, :l] = t_ids[0]
            lang_masks[i, :l] = t_mask[0]

        # Cache tokenized values so subsequent calls in the same step reuse them.
        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_masks
        return lang_tokens, lang_masks

    def _autocast(self):
        # The frozen base runs in bfloat16 while batches arrive as float32.
        enabled = str(self.device).startswith("cuda")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)

    def encode(self, batch, with_base_action=False, noise=None, num_base_samples=1):
        """
        Runs the frozen SmolVLA prefix (images + language + state) once and returns the
        mean-pooled semantic conditioning vector [B, cond_dim]. With with_base_action=True the
        same prefix KV cache is reused to sample the base policy's action chunk, returned as
        [B, chunk_size, action_dim] (or [B, S, chunk_size, action_dim] if num_base_samples > 1),
        in the policy's normalized action space.
        """
        with torch.no_grad(), self._autocast():
            vla_model = self.base_policy.model
            
            # Prepare inputs just like SmolVLA forward pass
            images, img_masks = self.base_policy.prepare_images(batch)
            state = self.base_policy.prepare_state(batch)

            # Ensure language exists in tokenized form for downstream SmolVLA calls.
            lang_tokens, lang_masks = self._prepare_language_tensors(batch)
            
            # 1. Get prefix embeddings (images + language + state)
            prefix_embs, prefix_pad_masks, prefix_att_masks = vla_model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            
            # 2. Pass through frozen VLM to get heavy semantic features
            # use_cache=True with past_key_values=None triggers fill_kv_cache=True (standard prefix self-attention prefill)
            hidden_states_list, past_key_values = vla_model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            
            # Mean pool over prefix sequence length to get 1D vector (matches trained checkpoint)
            pooled_semantics = hidden_states_list[0].mean(dim=1).float()

            base_action = None
            if with_base_action:
                from lerobot.policies.common.flow_matching import euler_integrate

                bsize = state.shape[0]
                noise_shape = (bsize, vla_model.config.chunk_size, vla_model.config.max_action_dim)
                samples = []
                for sample_idx in range(num_base_samples):
                    x_1 = noise if (noise is not None and sample_idx == 0) else vla_model.sample_noise(noise_shape, state.device)
                    actions = euler_integrate(
                        lambda x_t, timestep: vla_model.denoise_step(
                            x_t=x_t,
                            prefix_pad_masks=prefix_pad_masks,
                            past_key_values=past_key_values,
                            timestep=timestep,
                        ),
                        x_1,
                        vla_model.config.num_steps,
                    )
                    samples.append(actions[:, : self.chunk_size, : self.action_dim].float())
                base_action = samples[0] if num_base_samples == 1 else torch.stack(samples, dim=1)
            del past_key_values, hidden_states_list, prefix_embs

        if with_base_action:
            return pooled_semantics, base_action
        return pooled_semantics

    def extract_semantic_features(self, batch):
        """
        Passes images + language + robot state through the frozen SmolVLA backbone
        and extracts 1D pooled vision-language-state semantic conditioning vectors.
        """
        return self.encode(batch)
        
    def flow_loss(self, semantics, target_actions, base_action=None, delta_l2_weight=0.0):
        return flow_matching_loss(self.diffusion_head, semantics, target_actions, base_action, delta_l2_weight)

    def compute_loss(
        self,
        batch,
        gt_actions,
        residual_target=False,
        delta_l2_weight=0.0,
        return_metrics=False,
    ):
        """
        Train ONLY the new Diffusion head.
        """
        base_action = None
        if residual_target:
            # Same base sampler (config.num_steps) and same observation as at rollout time, so the
            # residual target is defined against the action the base would actually execute.
            semantics, base_action = self.encode(batch, with_base_action=True)
            if base_action.shape[1] > gt_actions.shape[1]:
                base_action = base_action[:, :gt_actions.shape[1], :]
            target_actions = gt_actions - base_action
        else:
            semantics = self.encode(batch)
            target_actions = gt_actions

        total_loss, flow_loss, delta_l2_loss = self.flow_loss(
            semantics,
            target_actions,
            base_action=base_action if self.diffusion_head.cond_on_base_action else None,
            delta_l2_weight=delta_l2_weight if residual_target else 0.0,
        )

        if return_metrics:
            metrics = {
                "flow_loss": float(flow_loss.detach().item()),
                "delta_l2_loss": float(delta_l2_loss.detach().item()),
                "target_action_abs_mean": float(target_actions.detach().abs().mean().item()),
            }
            if residual_target:
                metrics["base_action_abs_mean"] = float(base_action.detach().abs().mean().item())
            return total_loss, metrics

        return total_loss

    def sample_head(self, semantics, base_action=None, steps=10, return_intermediates=False):
        return sample_flow(self.diffusion_head, semantics, base_action, steps, return_intermediates)

    def select_action(self, batch, steps=10, return_intermediates=False):
        """
        Inference via fast Euler integration (Consistency/Flow matching style).
        """
        base_action = None
        if self.diffusion_head.cond_on_base_action:
            semantics, base_action = self.encode(batch, with_base_action=True)
        else:
            semantics = self.encode(batch)
        return self.sample_head(semantics, base_action, steps=steps, return_intermediates=return_intermediates)

    def select_residual_action(self, batch, steps=10, residual_alpha=1.0):
        """
        Residual-mode inference: base chunk and residual chunk come from the same observation and
        the same prefix pass. Returns (base + alpha * residual, base, residual), each
        [B, chunk_size, action_dim] in normalized action space.
        """
        semantics, base_action = self.encode(batch, with_base_action=True)
        residual = self.sample_head(semantics, base_action, steps=steps)
        return base_action + residual_alpha * residual, base_action, residual

    def get_trainable_state_dict(self):
        return {
            "diffusion_head": self.diffusion_head.state_dict()
        }

    def load_trainable_state_dict(self, state_dict):
        """Load the state dict into the diffusion head."""
        # Handle cases where the state dict might be nested inside 'diffusion_head' key
        if "diffusion_head" in state_dict:
            self.diffusion_head.load_state_dict(state_dict["diffusion_head"])
        else:
            self.diffusion_head.load_state_dict(state_dict)
