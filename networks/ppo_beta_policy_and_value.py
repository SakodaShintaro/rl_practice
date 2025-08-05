import torch
from torch import nn
from torch.distributions import Beta

from .backbone import AE


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int, seq_len: int) -> None:
        super().__init__()
        self.encoder = AE()
        seq_hidden_dim = self.encoder.output_dim
        rep_dim = 256
        hidden_dim = 100
        self.linear = nn.Linear(seq_hidden_dim, rep_dim)
        self.norm = nn.RMSNorm(rep_dim, elementwise_affine=False)
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
    ) -> tuple:
        x = s_seq[:, -1]  # Use the last time step representation (batch_size, seq_hidden_dim)
        # Get previous reward and action from sequences
        prev_reward = r_seq[:, -1].item() if r_seq.shape[1] > 0 else None
        prev_action = a_seq[:, -2].cpu().numpy() if a_seq.shape[1] > 1 else None
        x, _ = self.encoder(x, reward=prev_reward, prev_action=prev_action)
        x = self.linear(x)  # (batch_size, rep_dim)
        x = self.norm(x)

        value_x = self.value_enc(x)
        value = self.value_head(value_x)

        policy_x = self.policy_enc(x)
        alpha = self.alpha_head(policy_x).exp() + 1
        beta = self.beta_head(policy_x).exp() + 1

        dist = Beta(alpha, beta)
        if action is None:
            action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1, keepdim=True)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
        }
