# SPDX-License-Identifier: MIT
import torch
from torch import nn

from .blocks import SimbaBlock
from .image_processor import ImageProcessor
from .sparse_utils import apply_one_shot_pruning


def weights_init_(m: nn.Module) -> None:
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
        horizon: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        total_action_dim = action_dim * horizon
        mid_dim = in_channels + total_action_dim

        # Value stream: V(s) - depends only on state
        self.v_fc_in = nn.Linear(in_channels, hidden_dim)
        self.v_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.v_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.v_fc_out = nn.Linear(hidden_dim, num_bins)

        # Advantage stream: A(s,a) - depends on state and action
        self.a_fc_in = nn.Linear(mid_dim, hidden_dim)
        self.a_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.a_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.a_fc_out = nn.Linear(hidden_dim, num_bins)

        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        # Value stream: V(s)
        v = self.v_fc_in(x)
        v = self.v_fc_mid(v)
        v = self.v_norm(v)
        v_out = self.v_fc_out(v)  # (B, num_bins)

        # Advantage stream: A(s,a)
        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_bins)

        result_dict["activation"] = torch.cat([v, adv], dim=1)

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = v_out + adv_out
        result_dict["output"] = output

        return result_dict

    def get_advantage(self, x: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        xa = torch.cat([x, a_flat], dim=1)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        result_dict["activation"] = adv
        adv_out = self.a_fc_out(adv)  # (B, num_bins)
        result_dict["output"] = adv_out

        return result_dict


class SeparateCritic(nn.Module):
    """Separate critic network with its own ImageProcessor and MLP."""

    def __init__(
        self,
        observation_space_shape: tuple[int],
        processor_type: str,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.image_processor = ImageProcessor(observation_space_shape, processor_type)
        output_shape = self.image_processor.output_shape
        flat_dim = output_shape[0] * output_shape[1] * output_shape[2]

        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(flat_dim, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for separate critic.

        Args:
            obs: (B, C, H, W) observation image

        Returns:
            Dictionary with 'output' and 'activation' keys
        """
        result_dict = {}

        x = self.image_processor.encode(obs)
        x = self.flatten(x)
        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict
