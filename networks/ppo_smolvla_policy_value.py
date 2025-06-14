import torch
from torch import nn
from torch.distributions import Beta

from .smolvla_backbone import SmolVLABackbone

class PpoSmolvlaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.backbone = SmolVLABackbone()
        rep_dim = self.backbone.hidden_dim
        hidden_dim = 100
        self.value_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        r_seq: torch.Tensor,
        s_seq: torch.Tensor,
        a_seq: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict:
        x = self.backbone.encode(s_seq[:, -1])
        value_x = self.value_enc(x)
        value = self.value_head(value_x)

        policy_x = self.policy_enc(x)
        alpha = self.alpha_head(policy_x).exp() + 1
        beta = self.beta_head(policy_x).exp() + 1

        dist = Beta(alpha, beta)
        if action is None:
            action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=-1)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
        }
