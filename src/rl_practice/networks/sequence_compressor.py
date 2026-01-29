import torch
import torch.nn as nn

from .self_attention import SpatialTransformerBlock


class SequenceCompressor(nn.Module):
    """(B, L_in, D) -> (B, L_out, D) by learnable tokens and Self-Attention"""

    def __init__(self, hidden_dim: int, l_in: int, l_out: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.l_out = l_out

        self.learnable_tokens = nn.Parameter(torch.randn(1, l_out, hidden_dim))
        self.transformer_block = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    hidden_dim=hidden_dim,
                    n_head=1,
                    max_position_embeddings=l_in + l_out,
                )
                for _ in range(2)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, L_in, D)
    ) -> torch.Tensor:
        b, l_in, d = x.shape

        learnable_tokens = self.learnable_tokens.expand(b, -1, -1)  # (B, L_out, D)

        x = torch.cat([x, learnable_tokens], dim=1)  # (B, L_in + L_out, D)

        for block in self.transformer_block:
            x = block(x)  # (B, L_in + L_out, D)

        x = x[:, l_in:, :]  # (B, L_out, D)
        return x
