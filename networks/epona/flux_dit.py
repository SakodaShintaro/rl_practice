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


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
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
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
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
                for _ in range(params.depth)
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
        img: Tensor,
        cond: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> dict:
        # imgは常に4次元(B, C, H, W)として処理
        B, C, H, W = img.shape
        # 4次元の画像から2D位置エンコーディングを生成
        y_pos = torch.arange(H, device=img.device).repeat_interleave(W)
        x_pos = torch.arange(W, device=img.device).repeat(H)
        img_ids = torch.stack([y_pos, x_pos], dim=-1).unsqueeze(0).expand(B, -1, -1).float()
        cond_ids = img_ids

        # 3次元に変換
        img = img.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # condを3次元に変換
        cond = cond.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        cond = self.cond_in(cond)

        ids = torch.cat((cond_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, cond = block(img=img, cond=cond, vec=vec, pe=pe)

        img = torch.cat((cond, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, cond.shape[1] :, ...]

        # activationとしてfinal_layerの前の中間表現を保存
        activation = img.flatten(start_dim=1).detach()
        output = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        return {"output": output, "activation": activation}

    def sample(self, state: Tensor, action: Tensor, img_shape: tuple[int, int]) -> Tensor:
        batch_size = state.shape[0]
        H, W = img_shape

        # ランダムノイズから開始 - (B, C, H, W)形状
        img = torch.randn((batch_size, self.in_channels, H, W), device=state.device)

        # stateを4次元に変換してconditionとして使用
        cond = state.view(batch_size, self.in_channels, H, W)

        # サンプリング用のタイムステップを内部で設定
        timesteps = [1.0, 0.0]  # 1から0へ

        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred_dict = self(
                img=img,
                cond=cond,
                y=action,
                timesteps=t_vec,
            )
            pred = pred_dict["output"]  # (B, H*W, C)
            # predを(B, C, H, W)に戻して更新
            pred_4d = pred.permute(0, 2, 1).view(batch_size, self.in_channels, H, W)
            img = img + (t_prev - t_curr) * pred_4d
        return img
