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
    def __init__(self, base_policy_path, action_dim=7, device="cuda"):
        super().__init__()
        self.device = device
        
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
        
        # 3. Actor: Residual action head
        self.actor_residual = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 4. Critic: Value function for PPO/Advantage computation
        self.critic = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 5. Zero-initialize the final layer of the actor so delta_action starts exactly at 0.
        # This is critical! It ensures step 1 of RL performs identically to the baseline.
        nn.init.zeros_(self.actor_residual[-1].weight)
        nn.init.zeros_(self.actor_residual[-1].bias)
        
        self.to(device)

    def get_trainable_state_dict(self):
        return {
            "vision_encoder": self.vision_encoder.state_dict(),
            "actor_residual": self.actor_residual.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_trainable_state_dict(self, state_dict):
        self.vision_encoder.load_state_dict(state_dict["vision_encoder"])
        self.actor_residual.load_state_dict(state_dict["actor_residual"])
        self.critic.load_state_dict(state_dict["critic"])

    def forward(self, batch, deterministic=False):
        """
        Takes a batch of observations, gets the base action, and adds the trainable residual.
        """
        # 1. Get base action from the frozen SmolVLA model
        with torch.no_grad():
            base_action = self.base_policy.select_action(batch)
            
        import torch.utils.checkpoint as checkpoint
        
        # 2. Extract features for residual head
        img = batch.get('observation.images.image', batch.get('observation.image')) # Expected [B, C, H, W]
        if img is None:
            raise KeyError("Expected image in batch under 'observation.images.image' or 'observation.image'")
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
            
        def run_vision(x):
            return self.vision_encoder(x)
            
        if img.requires_grad:
            features = checkpoint.checkpoint(run_vision, img, use_reentrant=False)
        else:
            # Checkpoint requires at least one input to have requires_grad=True
            img.requires_grad_(True)
            features = checkpoint.checkpoint(run_vision, img, use_reentrant=False)
            
        # 3. Compute Value and Delta Action
        value = self.critic(features)
        delta_action_mean = self.actor_residual(features)
        
        # 4. Sample action for exploration
        if not deterministic:
            action_std = 0.05 # Small fixed standard deviation for exploration
            dist = torch.distributions.Normal(delta_action_mean, action_std)
            delta_action = dist.sample()
            log_prob = dist.log_prob(delta_action).sum(dim=-1)
        else:
            delta_action = delta_action_mean
            log_prob = None
            
        final_action = base_action + delta_action
        
        # Clamp final action to standard normalized bounds
        final_action = torch.clamp(final_action, min=-1.0, max=1.0)
        
        return final_action, delta_action_mean, value, log_prob, delta_action

    def compute_bc_anchor_loss(self, delta_action_mean):
        """
        Behavior Cloning Anchor Loss (L2 penalty).
        Prevents catastrophic forgetting by penalizing the residual head from drifting 
        too far from the frozen base model's predictions.
        """
        return torch.mean(delta_action_mean ** 2)