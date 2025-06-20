import torch
import torch.nn.functional as F
from torch import nn


def weights_init_(m):
    """
    Function to initialize weights.
    When used with apply(fn) recursively applied to every submodule as well as self.

    ## Input:

    - **m** *(nn.Module)*: Checks if the layer is a feedforward layer and initializes using the uniform glorot scheme if True.

    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def orthogonal_weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SacQ(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_normalize: bool = True,
    ) -> None:
        super().__init__()
        mid_dim = in_channels + action_dim
        self.fc1 = nn.Linear(mid_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.use_normalize = use_normalize
        self.apply(weights_init_)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SacTanhPolicy(nn.Module):
    def __init__(
        self, in_channels: int, action_dim: int, hidden_dim: int, use_normalize: bool = True
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.use_normalize = use_normalize
        self.apply(weights_init_)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return y_t, log_prob, mean
