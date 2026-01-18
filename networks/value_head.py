import torch
from torch import nn

from .blocks import SimbaBlock
from .sparse_utils import apply_one_shot_pruning


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


class StateValueHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_channels, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict


class ActionValueHead(nn.Module):
    """Dueling Architecture: Q(s,a) = V(s) + A(s,a)"""

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

        # Value stream: V(s) - 状態のみに依存
        self.v_fc_in = nn.Linear(in_channels, hidden_dim)
        self.v_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.v_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.v_fc_out = nn.Linear(hidden_dim, num_bins)

        # Advantage stream: A(s,a) - 状態と行動に依存
        self.a_fc_in = nn.Linear(mid_dim, hidden_dim)
        self.a_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.a_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.a_fc_out = nn.Linear(hidden_dim, num_bins)

        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        # Value stream: V(s)
        v = self.v_fc_in(x)
        v = self.v_fc_mid(v)
        v = self.v_norm(v)
        v_out = self.v_fc_out(v)  # (B, num_bins)

        # Advantage stream: A(s,a)
        xa = torch.cat([x, a], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_bins)

        result_dict["activation"] = torch.cat([v, adv], dim=1)

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = v_out + adv_out
        result_dict["output"] = output

        return result_dict
