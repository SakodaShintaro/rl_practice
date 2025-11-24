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
        state_dim: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()

        # Learnable query token
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Cross attention: query from learnable token, key/value from state sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, kdim=state_dim, vdim=state_dim, batch_first=True
        )

        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        # x: (B, S, D)
        bs = x.size(0)

        # Expand query token for batch
        query = self.query_token.expand(bs, -1, -1)  # (B, 1, hidden_dim)

        # Cross attention
        x, _ = self.cross_attn(query, x, x)  # (B, 1, hidden_dim)
        x = x.squeeze(1)  # (B, hidden_dim)

        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict


class ActionValueHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()

        # Project action to hidden_dim
        self.in_proj = nn.Linear(action_dim, hidden_dim)

        # Cross attention: query from action, key/value from state sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, kdim=state_dim, vdim=state_dim, batch_first=True
        )

        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        # x: (B, S, D), a: (B, action_dim)
        query = self.in_proj(a).unsqueeze(1)  # (B, 1, hidden_dim)

        # Cross attention
        x, _ = self.cross_attn(query, x, x)  # (B, 1, hidden_dim)
        x = x.squeeze(1)  # (B, hidden_dim)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict
