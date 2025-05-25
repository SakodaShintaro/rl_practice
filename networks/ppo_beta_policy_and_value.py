import torch
from torch import nn
from torch.distributions import Beta

from .sequence_processor import SequenceProcessor


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int, seq_len: int) -> None:
        super().__init__()
        self.sequential_compressor = SequenceProcessor(seq_len=seq_len)
        seq_hidden_dim = self.sequential_compressor.hidden_dim
        rep_dim = 256
        hidden_dim = 100
        self.linear = nn.Linear(seq_hidden_dim, rep_dim)
        self.value_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Softplus())

    def forward(self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor) -> tuple:
        # (batch_size, seq_len * 3 - 1, seq_hidden_dim)
        before, after = self.sequential_compressor(r_seq, s_seq, a_seq)

        # (batch_size, seq_len * 3 - 2, seq_hidden_dim)
        error = (after[:, :-1] - before[:, 1:].detach()) ** 2

        x = before[:, -1]  # Use the last time step representation (batch_size, seq_hidden_dim)
        x = self.linear(x)  # (batch_size, rep_dim)

        value_x = self.value_enc(x)
        v = self.value_head(value_x)

        policy_x = self.policy_enc(x)
        alpha = self.alpha_head(policy_x) + 1
        beta = self.beta_head(policy_x) + 1
        return (
            (alpha, beta),
            v,
            {
                "x": x,
                "value_x": value_x,
                "policy_x": policy_x,
                "error": error,
                "predicted_s": after[:, -2],
            },
        )

    def get_action_log_p_and_value(
        self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor, a: torch.Tensor
    ) -> tuple:
        (alpha, beta), v, info = self.forward(r_seq, s_seq, a_seq)
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a).sum(dim=1, keepdim=True)
        return a_logp, v, info

    @torch.inference_mode()
    def get_action_and_value(
        self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> tuple:
        (alpha, beta), v, activation_dict = self.forward(r_seq, s_seq, a_seq)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        return action, a_logp, v, activation_dict
