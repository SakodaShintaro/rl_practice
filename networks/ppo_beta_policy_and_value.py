import torch
from torch import nn
from torch.distributions import Beta

from .backbone import BaseCNN
from .sequence_compressor import SequenceCompressor


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, in_channels: int, action_dim: int) -> None:
        super().__init__()
        self.sequential_compressor = SequenceCompressor(seq_len=1)
        self.cnn_base = BaseCNN(in_channels)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, action_dim), nn.Softplus())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, 3, 96, 96)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        return x

    def head(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

    def forward(self, x: torch.Tensor) -> tuple:
        # x.shape = (batch_size, 3, 96, 96)
        x = self.encode(x)
        return self.head(x)

    def get_action_log_p_and_value(self, s: torch.Tensor, a: torch.Tensor) -> tuple:
        (alpha, beta), v = self.forward(s)
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a).sum(dim=1, keepdim=True)
        return a_logp, v

    @torch.inference_mode()
    def get_action_and_value(self, x: torch.Tensor) -> tuple:
        (alpha, beta), v = self.forward(x)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        return action, a_logp, v
