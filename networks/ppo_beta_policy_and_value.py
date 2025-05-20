import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .sequence_compressor import SequenceCompressor


class ReluSquared(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sign() * F.relu(x) ** 2


class SimBaBlock(nn.Module):
    """https://github.com/SonyResearch/simba/blob/master/scale_rl/networks/layers.py"""

    def __init__(self, input_dim: int):
        super().__init__()
        dim_hidden = input_dim * 4
        self.layer = nn.Sequential(
            nn.RMSNorm(input_dim),
            nn.Linear(input_dim, dim_hidden),
            ReluSquared(),
            nn.Linear(dim_hidden, input_dim),
        )

    def forward(self, x):
        res = x
        x = self.layer(x)
        return x + res


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.sequential_compressor = SequenceCompressor(seq_len=1)
        rep_dim = 256
        self.v = nn.Sequential(SimBaBlock(rep_dim), nn.Linear(rep_dim, 1))
        self.fc = nn.Sequential(SimBaBlock(rep_dim))
        self.alpha_head = nn.Sequential(nn.Linear(rep_dim, action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(rep_dim, action_dim), nn.Softplus())

    def forward(self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor) -> tuple:
        x = self.sequential_compressor(r_seq, s_seq, a_seq)
        x = x.flatten(start_dim=1)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

    def get_action_log_p_and_value(
        self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor, a: torch.Tensor
    ) -> tuple:
        (alpha, beta), v = self.forward(r_seq, s_seq, a_seq)
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a).sum(dim=1, keepdim=True)
        return a_logp, v

    @torch.inference_mode()
    def get_action_and_value(
        self, r_seq: torch.Tensor, s_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> tuple:
        (alpha, beta), v = self.forward(r_seq, s_seq, a_seq)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        return action, a_logp, v
