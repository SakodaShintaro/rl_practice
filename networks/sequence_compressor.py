from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, MultiheadAttention

from .backbone import AE, BaseCNN
from .diffusion_policy import TimestepEmbedder

"""
報酬 r_t
観測 o_t
行動 a_t
の系列を圧縮するためのTransformerベースのネットワーク
r_t, o_t, a_tはそれぞれ同じ次元(HIDDEN_DIM)に変換される
"""


class TransformerEncoderLayer(Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        device=None,
        dtype=None,
    ) -> None:
        dropout = 0.0
        bias = False
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        layer_norm_eps: float = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation_relu_or_gelu = 1
        self.activation = F.relu

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        coeff = 1e-6
        x = src
        x = x + coeff * self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
        )
        x = x + coeff * self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SequenceCompressor(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = 256

        # 報酬エンコーダー
        self.reward_encoder = TimestepEmbedder(self.hidden_dim)

        # 状態(画像)エンコーダー
        self.state_encoder = BaseCNN(in_channels=3)
        self.state_encoder_linear = nn.Linear(4 * 12 * 12, self.hidden_dim)

        # 行動エンコーダー
        self.action_encoder = nn.Linear(3, self.hidden_dim)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len * 3 - 1, self.hidden_dim))

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 出力層
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self, rewards: torch.Tensor, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rewards (torch.Tensor): 過去の報酬シーケンス (batch_size, seq_len, reward_dim)
            states (torch.Tensor): 過去の状態(画像)シーケンス (batch_size, seq_len, C, H, W)
            actions (torch.Tensor): 過去の行動シーケンス (batch_size, seq_len, action_dim)
        actionの末尾はdummy

        Returns:
            torch.Tensor: 圧縮表現 (batch_size, hidden_dim)
        """
        batch_size = states.shape[0]

        # 報酬をエンコード (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        rewards_embeds = self.reward_encoder(rewards)
        rewards_embeds = rewards_embeds.view(batch_size, self.seq_len, self.hidden_dim)

        # 状態(画像)をエンコード (batch_size * seq_len, C, H, W) -> (batch_size * seq_len, state_embed_dim)
        states_flat = states.reshape(-1, *states.shape[2:])

        # AEを使う場合
        # state_embeds = self.state_encoder.encode(
        #     states_flat
        # )  # (batch_size * seq_len, 4, 12, 12)
        # state_embeds = state_embeds.view(batch_size, self.seq_len, -1)
        # state_embeds = self.state_encoder_linear(state_embeds)

        # BaseCNNを使う場合
        state_embeds = self.state_encoder(states_flat)  # (batch_size * seq_len, 256)
        state_embeds = state_embeds.view(batch_size, self.seq_len, -1)

        # 行動をエンコード (batch_size, seq_len, action_dim) -> (batch_size, seq_len, hidden_dim)
        action_embeds = self.action_encoder(actions)

        # エンコードされた報酬、状態、行動を結合
        x = torch.stack((rewards_embeds, state_embeds, action_embeds), dim=2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.seq_len * 3, self.hidden_dim)
        x = x[:, :-1]  # dummyのactionを削除 (batch_size, seq_len * 3 - 1, hidden_dim)

        # Positional Encodingを追加
        x += self.pos_embedding

        # Transformer Encoderに通す
        transformer_output = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # 最後のトークンの出力を圧縮表現とする
        compressed_representation = transformer_output[:, -1, :]  # (batch_size, hidden_dim)

        # 出力層を通す
        # compressed_representation = self.norm(compressed_representation)
        # compressed_representation = F.relu(compressed_representation)
        # compressed_representation = self.output_layer(compressed_representation)

        return compressed_representation
