import torch
from torch import nn
from torch.distributions import Beta

from .sequence_processor import SequenceProcessor


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int, seq_len: int) -> None:
        super().__init__()
        self.sequential_processor = SequenceProcessor(seq_len=seq_len)
        seq_hidden_dim = self.sequential_processor.hidden_dim
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
        # (batch_size, seq_len * 3 - 1, seq_hidden_dim)
        before, after = self.sequential_processor(r_seq, s_seq, a_seq)

        prediction = after[:, :-1]  # (batch_size, seq_len * 3 - 2, seq_hidden_dim)
        prediction[:, 3::3] += before[:, 1:-1:3].detach()

        # (batch_size, seq_len * 3 - 2, seq_hidden_dim)
        error = (prediction - before[:, 1:].detach()) ** 2

        x = before[:, -1]  # Use the last time step representation (batch_size, seq_hidden_dim)
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
        a_logp = dist.log_prob(action).sum(dim=1)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
            "error": error,
            "predicted_s": prediction[:, -1],
        }
