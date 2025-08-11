import torch
import torch.nn.functional as F
from torch import nn

from .blocks import SimbaBlock
from .sparse_utils import apply_one_shot_pruning


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


class SacQ(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        mid_dim = in_channels + action_dim
        self.fc_in = nn.Linear(mid_dim, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x = torch.cat([x, a], dim=1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict
