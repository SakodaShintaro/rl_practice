import torch
import torch.nn as nn


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
