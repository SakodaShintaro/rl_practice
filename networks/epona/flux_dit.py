from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth_double_blocks: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool


class FluxDiT(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.noise_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.cond_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth_double_blocks)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        noise: Tensor,
        timesteps: Tensor,
        cond: Tensor,
        y: Tensor,
    ) -> Tensor:
        B, C, H, W = noise.shape
        # 位置エンコーディングを生成
        pos_id = torch.arange(H * W, device=noise.device)
        noise_ids = pos_id.unsqueeze(0).expand(B, -1).unsqueeze(-1).float()
        cond_ids = noise_ids

        # 3次元に変換
        noise = noise.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        cond = cond.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # running on sequences img
        noise = self.noise_in(noise)
        timesteps = timesteps.flatten()
        vec = self.time_in(timestep_embedding(timesteps, 256)) + self.vector_in(y)
        cond = self.cond_in(cond)

        ids = torch.cat((cond_ids, noise_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            noise, cond = block(img=noise, cond=cond, vec=vec, pe=pe)

        noise = torch.cat((cond, noise), 1)
        for block in self.single_blocks:
            noise = block(noise, vec=vec, pe=pe)
        noise = noise[:, cond.shape[1] :, ...]

        output = self.final_layer(noise, vec)  # (B, H*W, out_channels)
        output = output.permute(0, 2, 1).view(B, self.out_channels, H, W)

        # dummy
        activation = torch.zeros((B, 1))

        return {"output": output, "activation": activation}
