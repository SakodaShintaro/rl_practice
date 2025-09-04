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
    guidance_embed: bool


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
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
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
        guidance: Tensor | None = None,
    ) -> dict:
        if img.ndim != 3 or cond.ndim != 3:
            raise ValueError("Input img and cond tensors must have 3 dimensions.")

        batch_size = img.shape[0]
        cond_seq_len = cond.shape[1]
        img_seq_len = img.shape[1]

        # Generate position IDs internally
        cond_ids = torch.zeros((batch_size, cond_seq_len, 2), device=img.device)
        img_ids = torch.zeros((batch_size, img_seq_len, 2), device=img.device)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
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
        activation = img.detach()
        output = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        return {"output": output, "activation": activation}

    def training_losses(
        self,
        img: Tensor,  # (B, L, C)
        img_ids: Tensor,
        cond: Tensor,
        cond_ids: Tensor,
        t: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        noise: Tensor | None = None,
        return_predict=False,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(img)
        terms = {}

        x_t = t * img + (1.0 - t) * noise
        target = img - noise
        pred = self(
            img=x_t,
            img_ids=img_ids,
            cond=cond,
            cond_ids=cond_ids,
            timesteps=t.reshape(-1),
            y=y,
            guidance=guidance,
        )
        assert pred.shape == target.shape == img.shape
        predict = x_t + pred * (1.0 - t)
        terms["mse"] = mean_flat((target - pred) ** 2)

        terms["loss"] = terms["mse"].mean()
        if return_predict:
            terms["predict"] = predict
        else:
            terms["predict"] = None
        return terms

    def sample(
        self,
        state: Tensor,
        action: Tensor,
    ):
        batch_size = state.shape[0]
        state_dim = state.shape[-1]

        # ランダムノイズから開始
        img = torch.randn((batch_size, 1, state_dim), device=state.device)

        # current_stateとactionを結合してconditionとして使用
        state_action = torch.cat([state, action], dim=-1)
        cond = state_action.unsqueeze(1)  # (B, 1, state_dim + action_dim)

        # vecはactionと同じ
        vec = action

        # サンプリング用のタイムステップを内部で設定
        timesteps = [1.0, 0.0]  # 1から0へ

        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred_dict = self(
                img=img,
                cond=cond,
                y=vec,
                timesteps=t_vec,
            )
            pred = pred_dict["output"]
            img = img + (t_prev - t_curr) * pred
        return img
