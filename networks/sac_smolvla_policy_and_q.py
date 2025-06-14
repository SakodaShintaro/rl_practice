import torch
import torch.nn.functional as F
from torch import nn
from .smolvla_backbone import SmolVLABackbone

from .sac_tanh_policy_and_q import LOG_STD_MAX, LOG_STD_MIN

class SacSmolVLAQ(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.backbone = SmolVLABackbone()
        rep_dim = self.backbone.hidden_dim
        mid_dim = rep_dim + action_dim
        self.fc1 = nn.Linear(mid_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.backbone.encode(s)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SacSmolVLAPolicy(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = SmolVLABackbone()
        rep_dim = self.backbone.hidden_dim
        self.fc1 = nn.Linear(rep_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.encode(s)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return y_t, log_prob, mean
