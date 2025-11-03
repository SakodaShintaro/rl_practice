# ref. https://github.com/Kevin-thu/Epona/blob/main/models/stt.py
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm


def get_fourier_embeds_from_coordinates(embed_dim: int, coords: torch.Tensor) -> torch.Tensor:
    """
    連続値座標をフーリエ位置埋め込みに変換する

    アクション値などの連続値を高次元ベクトルに変換してTransformerで処理可能にする。
    異なる周波数のsin/cos関数を組み合わせて豊かな表現を作成。

    Args:
        embed_dim: 埋め込み次元数（偶数である必要がある）
        coords: 座標テンソル [B, T, coord_dim] または [B, T]

    Returns:
        torch.Tensor: shape [B, T, coord_dim, embed_dim] のフーリエ埋め込み
    """
    device = coords.device
    dtype = coords.dtype

    # 2次元テンソルの場合は3次元に拡張 [B, T] -> [B, T, 1]
    if coords.dim() == 2:
        coords = coords.unsqueeze(-1)

    batch_size, seq_len, coord_dim = coords.shape

    # 異なる周波数を生成（Transformerの位置埋め込みと同じ原理）
    half_dim = embed_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)

    # ブロードキャスト用に次元を拡張 [half_dim] -> [1, 1, 1, half_dim]
    emb = emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    coords = coords.unsqueeze(-1)  # [B, T, coord_dim] -> [B, T, coord_dim, 1]

    # 各座標値に各周波数を掛ける [B, T, coord_dim, half_dim]
    emb = coords * emb

    # sin/cosを組み合わせて埋め込みベクトルを作成
    # [sin(coord*freq1), cos(coord*freq1), sin(coord*freq2), cos(coord*freq2), ...]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = emb.view(batch_size, seq_len, coord_dim, embed_dim)

    return emb


@dataclass
class Config:
    hidden_dim: int
    n_head: int
    attn_drop_prob: float
    res_drop_prob: float


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.n_head == 0
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.res_drop_prob = nn.Dropout(config.res_drop_prob)
        self.attn_dropout_rate = config.attn_drop_prob
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.hidden_dim)
            self.k_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
        y = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout_rate
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.res_drop_prob(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def forward(self, x, attn_mask, rnn_state):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x, rnn_state


class MambaBlock(nn.Module):
    """
    Mamba状態空間モデルブロック（TransformerBlockと置き換え可能）

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
            - res_drop_prob: residual dropout確率
    """

    def __init__(self, config):
        super().__init__()
        d_ssm = 2 * config.hidden_dim
        self.block = Block(
            dim=config.hidden_dim,
            mixer_cls=lambda dim: Mamba2(d_model=dim, headdim=d_ssm),
            norm_cls=lambda dim: RMSNorm(dim),
            mlp_cls=lambda dim: nn.Sequential(
                nn.Linear(dim, 4 * dim, bias=False),
                nn.GELU(),
                nn.Linear(4 * dim, dim, bias=False),
                nn.Dropout(config.res_drop_prob),
            ),
        )

    def forward(self, x, attn_mask, rnn_state):
        return self.block(x)[0], rnn_state


class GRUBlock(nn.Module):
    """
    GRUブロック（TransformerBlock/MambaBlockと置き換え可能）

    このブロックはインターフェースを統一するため、rnn_stateの形状変換を内部で行います。

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        # num_layers=1 のGRUを使用
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x, attn_mask, rnn_state):
        """
        Args:
            x: [B, T, C]
            attn_mask: 未使用（インターフェース互換性のため）
            rnn_state: [1, B, C] GRUの隠れ状態

        Returns:
            x: [B, T, C]
            rnn_state: [1, B, C] 更新されたGRUの隠れ状態
        """
        x, rnn_state = self.gru(x, rnn_state)
        return x, rnn_state


class SpatialTemporalBlock(nn.Module):
    def __init__(self, config, temporal_model_type):
        super().__init__()
        self.temporal_model_type = temporal_model_type

        if temporal_model_type == "mamba":
            self.tempo_block = MambaBlock(config)
        elif temporal_model_type == "transformer":
            self.tempo_block = TransformerBlock(config)
        elif temporal_model_type == "gru":
            self.tempo_block = GRUBlock(config)
        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")
        self.space_block = TransformerBlock(config)

    def forward(self, x, attn_mask, rnn_state):
        """
        Args:
            x: [B, T, S, C]
            attn_mask: attention mask for temporal dimension
            rnn_state: [1, B*S, C] GRU hidden state (only for temporal_model_type=="gru"), or None

        Returns:
            x: [B, T, S, C]
            rnn_state: [1, B*S, C] updated GRU hidden state (only for temporal_model_type=="gru"), or None
        """
        b, t, s, c = x.shape
        x = rearrange(x, "b t s c -> (b s) t c")
        x, rnn_state = self.tempo_block(x, attn_mask, rnn_state)
        x = rearrange(x, "(b s) t c -> (b t) s c", b=b, s=s, t=t)
        x, _ = self.space_block(x, None, None)
        x = rearrange(x, "(b t) s c -> b t s c", b=b, t=t)
        return x, rnn_state


class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        n_layer: int,
        space_len: int,
        tempo_len: int,
        hidden_dim: int,
        n_head: int,
        attn_drop_prob: float,
        res_drop_prob: float,
        temporal_model_type: str,
    ):
        super().__init__()
        config = Config(
            hidden_dim=hidden_dim,
            n_head=n_head,
            attn_drop_prob=attn_drop_prob,
            res_drop_prob=res_drop_prob,
        )

        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.space_len = space_len
        self.temporal_model_type = temporal_model_type

        self.tempo_emb = nn.Parameter(torch.zeros(1, tempo_len, 1, self.hidden_dim))
        nn.init.normal_(self.tempo_emb.data, mean=0, std=0.02)
        self.space_emb = nn.Parameter(torch.zeros(1, 1, space_len, self.hidden_dim))
        nn.init.normal_(self.space_emb.data, mean=0, std=0.02)

        self.spatial_temporal_blocks = nn.ModuleList(
            [SpatialTemporalBlock(config, temporal_model_type) for _ in range(self.n_layer)]
        )

        self.apply(self._init_weights)

        matrix = torch.tril(torch.ones(tempo_len, tempo_len))
        time_causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        time_causal_mask = torch.where(matrix == 1, 0, time_causal_mask)
        self.register_buffer("mask_time", time_causal_mask.contiguous())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, rnn_state: torch.Tensor) -> tuple:
        """
        Forward pass of the SpatialTemporalTransformer.

        Args:
            x: [B, T, S, C]
            rnn_state: [1, B, n_layer * S * C] (for GRU) or [1, B, 1] (for others)

        Returns:
            out: [B, T, S, C]
            rnn_state: [1, B, n_layer * S * C] (for GRU) or same as input (for others)
        """
        B, T, S, C = x.shape

        tempo_emb = torch.repeat_interleave(self.tempo_emb, S, dim=2)
        space_emb = torch.repeat_interleave(self.space_emb, T, dim=1)

        out = x + tempo_emb + space_emb

        # rnn_stateを各レイヤーの状態に分割
        # GRUの場合: [1, B, n_layer * S * C] -> n_layer個の [1, B*S, C]
        # それ以外: rnn_stateをそのまま使用（各ブロックが無視する）
        if self.temporal_model_type == "gru":
            # [1, B, n_layer * S * C] -> [B, n_layer * S * C]
            rnn_state_squeezed = rnn_state.squeeze(0)
            # [B, n_layer * S * C] -> [B, n_layer, S, C]
            layer_states = rnn_state_squeezed.view(B, self.n_layer, S, C)
            layer_states = [layer_states[:, i, :, :].contiguous() for i in range(self.n_layer)]
            # 各レイヤーの状態を [1, B*S, C] に変形
            layer_states = [s.view(1, B * S, C) for s in layer_states]
        else:
            layer_states = [None] * self.n_layer

        new_layer_states = []
        for i in range(self.n_layer):
            out, new_state = self.spatial_temporal_blocks[i](out, self.mask_time, layer_states[i])
            new_layer_states.append(new_state)

        # 各レイヤーの状態を結合
        # GRUの場合: n_layer個の [1, B*S, C] -> [1, B, n_layer * S * C]
        # それ以外: 入力のrnn_stateをそのまま返す
        if self.temporal_model_type == "gru":
            new_layer_states = [s.view(B, S, C) for s in new_layer_states]
            # [B, n_layer, S, C] -> [B, n_layer * S * C] -> [1, B, n_layer * S * C]
            rnn_state = (
                torch.stack(new_layer_states, dim=1).view(B, self.n_layer * S * C).unsqueeze(0)
            )

        return out, rnn_state
