import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
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
    def __init__(self, action_dim=7, chunk_size=16, cond_dim=2048, hidden_dim=256, num_layers=4):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
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
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.action_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
        
    def forward(self, x, time, cond):
        """
        x: [B, chunk_size, action_dim] NOISY action trajectory
        time: [B] noise level or timestep
        cond: [B, cond_dim] semantic feature from frozen brain
        """
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.action_in(x)
        
        t_emb = self.time_emb(time)
        c_emb = self.cond_proj(cond)
        emb = t_emb + c_emb
        
        for block in self.blocks:
            x = block(x, emb)
            
        out = self.action_out(x)
        return out.view(B, self.chunk_size, self.action_dim)


class HybridFrozenBrainDiffusionHands(nn.Module):
    """
    Implements the 'Frozen Brain, Diffusion Hands' Architecture.
    Strips standard action head, freezes the VLM backbone, and attaches a Diffusion head.
    """
    def __init__(
        self,
        base_policy_path,
        action_dim=7,
        chunk_size=16,
        cond_dim=2048, # Default for SmolVLM hidden_size depending on variant. Update if different.
        diff_hidden_dim=256,
        diff_layers=5,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.cond_dim = cond_dim
        
        print(f"[HybridDiffusion] Loading frozen base policy from: {base_policy_path}")
        try:
            self.base_policy = SmolVLAPolicy.from_pretrained(base_policy_path)
        except NameError:
            print("SmolVLAPolicy not found. Make sure LeRobot is installed with SmolVLAPolicy support.")
            
        # Keep base policy in fp32 to avoid dtype mismatch with SmolVLA action-noise path.
        self.base_policy.to(torch.float32)
        self.base_policy.to(self.device)
        self.base_policy.eval()
        
        # 1. Strip the standard action head completely by completely freezing the 500M parameters backbone
        # so it acts purely as a semantic feature extractor. 
        for param in self.base_policy.parameters():
            param.requires_grad = False
            
        print("[HybridDiffusion] All base params frozen.")
        
        # 2. Attach continuous Diffusion Policy head
        self.diffusion_head = ConditionalDDPMHead(
            action_dim=action_dim,
            chunk_size=chunk_size,
            cond_dim=cond_dim,
            hidden_dim=diff_hidden_dim,
            num_layers=diff_layers
        )
        self.diffusion_head.to(self.device)
        print(f"[HybridDiffusion] Attached highly specialized Diffusion Head with condition dim {cond_dim}.")

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

        processor = self.base_policy.model.vlm_with_expert.processor
        text_out = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        lang_tokens = text_out["input_ids"].to(self.device)
        lang_masks = text_out["attention_mask"].to(self.device).bool()

        # Cache tokenized values so subsequent calls in the same step reuse them.
        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_masks
        return lang_tokens, lang_masks

    def extract_semantic_features(self, batch):
        """
        Uses the frozen SmolVLA backbone purely as a semantic feature extractor.
        """
        with torch.no_grad():
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
            
            # Dynamically infer the make_att_2d_masks function from the backbone if not imported
            # usually it is in lerobot.policies.smolvla.modeling_smolvla
            try:
                from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            except ImportError:
                # Fallback to a simple attention mask
                prefix_att_2d_masks = prefix_att_masks.unsqueeze(1) & prefix_att_masks.unsqueeze(2)

            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            
            # 2. Pass through frozen VLM to get heavy semantic features
            outputs = vla_model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.base_policy.config.use_cache,
                fill_kv_cache=True,
            )
            
            # Depending on transformer return type unpacking
            # outputs[0] contains the hidden states for inputs_embeds
            # Since we passed [prefix_embs, None], it returns [prefix_hidden_states, suffix_hidden_states]
            # outputs is a tuple: (list_of_hidden_states, past_key_values)
            
            if isinstance(outputs, tuple) and len(outputs) == 2:
                hidden_states_list, past_key_values = outputs
                hidden_states = hidden_states_list[0]
            else:
                hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                if isinstance(hidden_states, (tuple, list)):
                    hidden_states = hidden_states[0] # Handle nested tuples
            
            # Mean pool over sequence length to get 1D vector (or use CLS/final token)
            # shape: [B, seq_len, cond_dim] => [B, cond_dim]
            pooled_semantics = hidden_states.mean(dim=1)
            
        return pooled_semantics.float() # Convert to float32 for diffusion head
        
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

        When residual_target=True, train on action deltas relative to the frozen
        base policy: delta = gt_actions - base_action.
        """
        semantics = self.extract_semantic_features(batch)

        if residual_target:
            with torch.no_grad():
                self._prepare_language_tensors(batch)
                use_cuda_autocast = str(self.device).startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
                    base_action = self.base_policy.select_action(batch)
                base_action = base_action.to(torch.float32)
            if base_action.ndim == 2:
                base_action = base_action.unsqueeze(1)
            if base_action.shape[1] == 1 and gt_actions.shape[1] > 1:
                base_action = base_action.expand(-1, gt_actions.shape[1], -1)
            target_actions = gt_actions - base_action
        else:
            target_actions = gt_actions
        
        B = gt_actions.shape[0]
        device = gt_actions.device
        
        # Sample random noise & timesteps for Conditional Flow Matching / DDPM
        noise = torch.randn_like(target_actions)
        timesteps = torch.rand((B,), device=device) # continuous [0, 1]
        
        # Flow Matching style interpolation
        # x_t = t * target_actions + (1 - t) * noise
        t_expand = timesteps[:, None, None]
        x_t = t_expand * target_actions + (1 - t_expand) * noise
        target_velocity = target_actions - noise
        
        # Predict velocity
        pred_velocity = self.diffusion_head(x_t, timesteps, semantics)

        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        total_loss = flow_loss
        delta_l2_loss = torch.tensor(0.0, device=device)

        if residual_target and delta_l2_weight > 0.0:
            # Reconstruct predicted clean delta from flow velocity parameterization.
            pred_delta = x_t + (1.0 - t_expand) * pred_velocity
            delta_l2_loss = pred_delta.square().mean()
            total_loss = total_loss + float(delta_l2_weight) * delta_l2_loss

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

    def select_action(self, batch, steps=10, return_intermediates=False):
        """
        Inference via fast Euler integration (Consistency/Flow matching style).
        """
        semantics = self.extract_semantic_features(batch)
        B = semantics.shape[0]
        
        x_t = torch.randn((B, self.chunk_size, self.action_dim), device=self.device)
        intermediates = [x_t.clone()] if return_intermediates else None
        
        # ODE Solver
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=self.device)
            v_pred = self.diffusion_head(x_t, t, semantics)
            x_t = x_t + v_pred * dt
            if return_intermediates:
                intermediates.append(x_t.clone())
                
        if return_intermediates:
            return x_t, intermediates
            
        return x_t

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
