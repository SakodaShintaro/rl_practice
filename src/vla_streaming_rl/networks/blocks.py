# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormedLinear(nn.Module):
    """Linear layer with weight projected onto the unit hypersphere in forward.

    Weights are L2-normalized along the input dimension at each forward call,
    so the stored parameters are free to be updated by any optimizer while the
    effective weights always have unit norm per output vector.  No bias is used.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class SimbaV2Block(nn.Module):
    """https://arxiv.org/abs/2502.15280

    Inverted-bottleneck MLP with:
      - L2 normalization instead of LayerNorm
      - NormedLinear (weight on unit hypersphere, no bias)
      - Learnable scaler vector
      - LERP residual connection
    All weight projection is handled inside forward, so no special optimizer
    or external hook is required.

    Args:
        channels: hidden dimension (dh)
    """

    _EXPANSION = 4
    _ALPHA_INIT = 0.1

    def __init__(self, channels: int) -> None:
        super().__init__()
        inner = channels * self._EXPANSION
        self.linear1 = NormedLinear(channels, inner)
        self.linear2 = NormedLinear(inner, channels)

        s_init = math.sqrt(2.0 / channels)
        self.scaler = nn.Parameter(torch.full((inner,), s_init))

        alpha_scale = 1.0 / math.sqrt(channels)
        self.alpha = nn.Parameter(torch.full((channels,), alpha_scale))
        self._alpha_ratio = self._ALPHA_INIT / alpha_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP + L2 Norm  (Eq.11)
        h = self.linear1(x)
        h = F.relu(h)
        h = h * self.scaler
        h = self.linear2(h)
        h = F.normalize(h, dim=-1)

        # LERP + L2 Norm  (Eq.12)
        alpha = self.alpha * self._alpha_ratio
        out = (1.0 - alpha) * x + alpha * h
        return F.normalize(out, dim=-1)


class SimbaBlock(nn.Module):
    """https://arxiv.org/abs/2410.09754"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels, elementwise_affine=False),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BroBlock(nn.Module):
    """https://arxiv.org/abs/2405.16158"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
