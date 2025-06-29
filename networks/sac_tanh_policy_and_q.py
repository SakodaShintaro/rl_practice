import torch
import torch.nn.functional as F
from torch import nn

from .simba_block import SimbaBlock
from .sparse_utils import apply_one_shot_pruning


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
        sparsity: float = 0.0,
    ) -> None:
        super().__init__()
        mid_dim = in_channels + action_dim
        self.fc_in = nn.Linear(mid_dim, hidden_dim)
        self.fc_mid = SimbaBlock(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.use_normalize = use_normalize
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x = torch.cat([x, a], dim=1)
        x = self.fc_in(x)

        x = self.fc_mid(x)

        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))

        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SacTanhPolicy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hidden_dim: int,
        use_normalize: bool = True,
        sparsity: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.use_normalize = use_normalize
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        result_dict["activation"] = x2

        if self.use_normalize:
            x2 = x2 / torch.norm(x2, dim=1).view((-1, 1))
            result_dict["activation"] = x2

        mean = self.fc_mean(x2)
        log_std = self.fc_logstd(x2)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        result_dict["mean"] = mean
        result_dict["log_std"] = log_std

        return result_dict

    def get_action(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = self(x)
        mean = result_dict["mean"]
        log_std = result_dict["log_std"]
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return y_t, log_prob, mean
