import torch
import torch.nn.functional as F
from torch import nn

from .backbone import BaseCNN

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PpoTanhPolicyAndValue(nn.Module):
    def __init__(self, in_channels: int, action_dim: int) -> None:
        super().__init__()
        self.cnn_base = BaseCNN(in_channels)
        CNN_OUT_DIM = 256
        hidden_dim_v = 100
        hidden_dim_p = 256
        self.v = nn.Sequential(
            nn.Linear(CNN_OUT_DIM, hidden_dim_v),
            nn.ReLU(),
            nn.Linear(hidden_dim_v, 1),
        )

        # Policy
        self.fc1 = nn.Linear(CNN_OUT_DIM, hidden_dim_p)
        self.fc2 = nn.Linear(hidden_dim_p, hidden_dim_p)
        self.fc_mean = nn.Linear(hidden_dim_p, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim_p, action_dim)

        self.apply(self._weights_init)

        self.action_scale = 0.5
        self.action_bias = 0.5

    @staticmethod
    def _weights_init(m: object) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, 3, 96, 96)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        return x

    def forward(self, x: torch.Tensor) -> tuple:
        # x.shape = (batch_size, 3, 96, 96)
        x = self.encode(x)
        v = self.v(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return (mean, log_std), v

    @torch.inference_mode()
    def get_action_and_value(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mean, log_std), v = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_a = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action_range2 = torch.tanh(raw_a)
        action_range1 = action_range2 * self.action_scale + self.action_bias
        a_logp = normal.log_prob(raw_a)
        a_logp -= torch.log(self.action_scale * (1 - action_range2.pow(2)) + 1e-6)
        a_logp = a_logp.sum(1, keepdim=True)
        return action_range1, a_logp, v

    def get_action_log_p_and_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        (mean, log_std), v = self.forward(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        a_range2 = (a - self.action_bias) / self.action_scale
        a_range2 = torch.clamp(a_range2, -0.999, 0.999)
        raw_a = torch.atanh(a_range2)
        a_logp = normal.log_prob(raw_a)
        a_logp -= torch.log(self.action_scale * (1 - a_range2.pow(2)) + 1e-6)
        a_logp = a_logp.sum(1, keepdim=True)
        return a_logp, v
