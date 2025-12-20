import torch
from torch import nn


class RewardProcessor(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bias = bias
        self.encoder = nn.Linear(1, embed_dim, bias=bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        embedded = self.encoder(x)
        return embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            bias_expanded = self.encoder.bias.unsqueeze(0).unsqueeze(0)
            decoded_values = (embedded - bias_expanded) / (self.encoder.weight[:, 0] + 1e-6)
        else:
            decoded_values = embedded / (self.encoder.weight[:, 0] + 1e-6)

        decoded = decoded_values.mean(dim=-1)
        return decoded
