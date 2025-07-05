from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, MultiheadAttention

from .backbone import AE, BaseCNN
from .diffusion_policy import TimestepEmbedder
from .sparse_utils import apply_one_shot_pruning

"""
報酬 r_t
観測 o_t
行動 a_t
の系列を処理するためのTransformerベースのネットワーク
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

        self.norm1_prev = LayerNorm(d_model, elementwise_affine=False, **factory_kwargs)
        self.norm1_post = LayerNorm(d_model, elementwise_affine=True, **factory_kwargs)
        self.norm2_prev = LayerNorm(d_model, elementwise_affine=False, **factory_kwargs)
        self.norm2_post = LayerNorm(d_model, elementwise_affine=True, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation_relu_or_gelu = 1
        self.activation = F.relu

        torch.nn.init.zeros_(self.norm2_post.weight)
        torch.nn.init.zeros_(self.norm2_post.bias)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # 因果マスクを作成
        seq_len = src.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=src.device)

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        x = x + self.norm1_post(
            self._sa_block(
                self.norm1_prev(x), attn_mask=causal_mask, key_padding_mask=None, is_causal=True
            )
        )
        x = x + self.norm2_post(self._ff_block(self.norm2_prev(x)))
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


class SequenceProcessor(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, sparsity: float):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Positional Encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len * 3 - 1, self.hidden_dim), requires_grad=True
        )

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.sparse_mask = (
            None
            if sparsity == 0.0
            else apply_one_shot_pruning(self.transformer_encoder, overall_sparsity=sparsity)
        )

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
            torch.Tensor: 変換前の系列表現 (batch_size, seq_len * 3 - 1, hidden_dim)
            torch.Tensor: 変換後の系列表現 (batch_size, seq_len * 3 - 1, hidden_dim)
        """
        batch_size = states.shape[0]

        # 報酬をエンコード (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        rewards_embeds = self.reward_encoder(rewards)
        rewards_embeds = rewards_embeds.view(batch_size, self.seq_len, self.hidden_dim)

        # 状態(画像)をエンコード (batch_size * seq_len, C, H, W) -> (batch_size * seq_len, state_embed_dim)
        states = states.reshape(-1, *states.shape[2:])
        with torch.no_grad():
            # (batch_size * seq_len, 4, 12, 12)
            state_embeds = self.state_encoder.encode(states)
        state_embeds = state_embeds.view(batch_size, self.seq_len, self.hidden_dim)

        # 行動をエンコード (batch_size, seq_len, action_dim) -> (batch_size, seq_len, hidden_dim)
        action_embeds = self.action_encoder(actions)

        # エンコードされた報酬、状態、行動を結合
        # 時系列順に(r1, s1, a1, r2, s2, a2, ..., rn, sn)と並べる
        x_list = []
        for i in range(self.seq_len):
            x_list.append(rewards_embeds[:, i])  # r_i
            x_list.append(state_embeds[:, i])  # s_i
            if i < self.seq_len - 1:  # 最後のアクションはダミーなので除外
                x_list.append(action_embeds[:, i])  # a_i

        # リストをスタックして結合 (batch_size, seq_len * 3 - 1, hidden_dim)
        x = torch.stack(x_list, dim=1)

        # 処理前
        before = x  # (batch_size, seq_len * 3 - 1, hidden_dim)

        # 処理
        x = x + self.pos_embedding
        after = self.transformer_encoder(x)  # (batch_size, seq_len * 3 - 1, hidden_dim)

        return before, after
