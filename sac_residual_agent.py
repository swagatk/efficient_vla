import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

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

class SACResidualVLAPolicy(nn.Module):
    def __init__(
        self,
        base_policy_path,
        action_dim=7,
        state_dim=8,
        residual_scale=0.02,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.residual_scale = residual_scale
        self.action_dim = action_dim
        
        print(f"[SACResidualVLA] Loading frozen base policy from: {base_policy_path}")
        self.base_policy = PreTrainedPolicy.from_pretrained(base_policy_path)
        self.base_policy.to(torch.bfloat16)
        self.base_policy.to(self.device)
        self.base_policy.eval()
        
        for param in self.base_policy.parameters():
            param.requires_grad = False
            
        print("[SACResidualVLA] Initializing trainable residual actor and critics...")
        self.vision_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vision_encoder.fc = nn.Identity()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        fused_dim = 512 + 64
        
        # SAC Actor
        self.actor_net = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Linear(64, action_dim)
        
        # SAC Critics (Twin Q-Networks)
        self.q1 = nn.Sequential(
            nn.Linear(fused_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(fused_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Zero-initialize the final layers to start near 0 residual
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.zeros_(self.actor_log_std.weight)
        nn.init.constant_(self.actor_log_std.bias, -2.0) # Start with a small std dev so behavior doesn't immediately explode

        self.to(device)

    def extract_features(self, batch):
        img = batch.get('observation.images.image', batch.get('observation.image'))
        img = img.float() if img.dtype != torch.float32 else img
        if img.max() > 1.0: img = img / 255.0
        state = batch.get('observation.state').float()

        features = self.vision_encoder(img)
        state_features = self.state_encoder(state)
        return torch.cat([features, state_features], dim=-1)

    def sample_action(self, fused_features, deterministic=False):
        x = self.actor_net(fused_features)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        if deterministic:
            delta_action = mean
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            # In purely residual, we can just use normal and clip later, 
            # or squash if relying on strict bounds.
            # Using Tanh to bound the delta action to [-1, 1] before scaling.
            delta_action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - delta_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return delta_action, log_prob, mean

    def forward(self, batch, deterministic=False):
        with torch.no_grad():
            base_action = self.base_policy.select_action(batch)
            
        fused_features = self.extract_features(batch)
        delta_action, log_prob, mean = self.sample_action(fused_features, deterministic)
        
        scaled_delta_action = self.residual_scale * delta_action
        final_action = base_action + scaled_delta_action
        final_action = torch.clamp(final_action, min=-1.0, max=1.0)
        
        return final_action, base_action, delta_action, scaled_delta_action, log_prob

    def get_trainable_state_dict(self):
        return {
            "vision_encoder": self.vision_encoder.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "actor_net": self.actor_net.state_dict(),
            "actor_mean": self.actor_mean.state_dict(),
            "actor_log_std": self.actor_log_std.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
        }