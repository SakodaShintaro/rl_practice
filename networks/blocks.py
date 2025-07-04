import torch
import torch.nn as nn


class SimbaBlock(nn.Module):
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
