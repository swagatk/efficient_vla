import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Handle import path differences between LeRobot versions (v0.3 vs v0.4+)
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PreTrainedPolicy
except ImportError:
    try:
        from lerobot.common.policies.pretrained import PreTrainedPolicy
    except ImportError:
        try:
            from lerobot.policies.pretrained import PreTrainedPolicy
        except ImportError:
            from lerobot.policies import PreTrainedPolicy

class ResidualVLAPolicy(nn.Module):
    def __init__(
        self,
        base_policy_path,
        action_dim=7,
        state_dim=8,
        residual_scale=0.02,
        init_log_std=0.0,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.residual_scale = residual_scale
        
        print(f"[ResidualVLA] Loading frozen base policy from: {base_policy_path}")
        self.base_policy = PreTrainedPolicy.from_pretrained(base_policy_path)
        self.base_policy.to(torch.bfloat16)
        self.base_policy.to(self.device)
        self.base_policy.eval()
        
        # 1. Freeze all base policy parameters
        for param in self.base_policy.parameters():
            param.requires_grad = False
            
        print("[ResidualVLA] Initializing trainable residual head and critic...")
        # 2. A lightweight visual encoder for the residual head
        # We use a small ResNet18 to keep VRAM overhead extremely low (< 0.5 GB)
        self.vision_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vision_encoder.fc = nn.Identity() # Output is 512-dim feature vector
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        fused_dim = 512 + 64
        
        # 3. Actor: Residual action head
        self.actor_residual = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 4. Critic: Value function for PPO/Advantage computation
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 5. Zero-initialize the final layer of the actor so delta_action starts exactly at 0.
        # This is critical! It ensures step 1 of RL performs identically to the baseline.
        nn.init.zeros_(self.actor_residual[-1].weight)
        nn.init.zeros_(self.actor_residual[-1].bias)
        
        # 6. Learnable standard deviation for better exploration
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        self.to(device)

    def get_trainable_state_dict(self):
        return {
            "vision_encoder": self.vision_encoder.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "actor_residual": self.actor_residual.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std,
        }

    def load_trainable_state_dict(self, state_dict):
        self.vision_encoder.load_state_dict(state_dict["vision_encoder"])
        if "state_encoder" in state_dict:
            self.state_encoder.load_state_dict(state_dict["state_encoder"])
        self.actor_residual.load_state_dict(state_dict["actor_residual"])
        self.critic.load_state_dict(state_dict["critic"])
        if "log_std" in state_dict:
            self.log_std.data.copy_(state_dict["log_std"])

    def forward_residual(self, batch):
        """Only runs the lightweight residual network. Useful for batched PPO updates."""
        # 1. Extract features for residual head
        img = batch.get('observation.images.image', batch.get('observation.image')) # Expected [B, C, H, W]
        if img is None:
            raise KeyError("Expected image in batch under 'observation.images.image' or 'observation.image'")
        img = img.float() if img.dtype != torch.float32 else img
        if img.max() > 1.0:
            img = img / 255.0
        state = batch.get('observation.state')
        if state is None:
            raise KeyError("Expected state in batch under 'observation.state'")
        state = state.float()

        # Use a direct forward pass here. Checkpointing in this PPO loop can create
        # hard-to-debug autograd graph reuse issues across repeated minibatch updates.
        features = self.vision_encoder(img)
        state_features = self.state_encoder(state)
        fused_features = torch.cat([features, state_features], dim=-1)
            
        # 2. Compute Value and Delta Action
        value = self.critic(fused_features)
        delta_action_mean = self.actor_residual(fused_features)
        
        return delta_action_mean, value

    def forward(self, batch, deterministic=False):
        """
        Takes a batch of observations, gets the base action, and adds the trainable residual.
        """
        # 1. Get base action from the frozen SmolVLA model
        with torch.no_grad():
            base_action = self.base_policy.select_action(batch)
            
        delta_action_mean, value = self.forward_residual(batch)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(delta_action_mean, action_std)
        
        if not deterministic:
            delta_action = dist.sample()
        else:
            delta_action = delta_action_mean
        log_prob = dist.log_prob(delta_action).sum(dim=-1)
            
        scaled_delta_action = self.residual_scale * delta_action
        final_action = base_action + scaled_delta_action
        
        # Clamp final action to standard normalized bounds
        final_action = torch.clamp(final_action, min=-1.0, max=1.0)
        
        return final_action, base_action, delta_action_mean, value, log_prob, delta_action, scaled_delta_action

    def compute_bc_anchor_loss(self, delta_action_mean):
        """
        Behavior Cloning Anchor Loss (L2 penalty).
        Prevents catastrophic forgetting by penalizing the residual head from drifting 
        too far from the frozen base model's predictions.
        """
        return torch.mean(delta_action_mean ** 2)