from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


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
    def __init__(self, config, max_position_embeddings, use_rope):
        super().__init__()
        assert config.hidden_dim % config.n_head == 0
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.res_drop_prob = nn.Dropout(config.res_drop_prob)
        self.attn_dropout_rate = config.attn_drop_prob
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.n_head = config.n_head
        self.head_dim = config.hidden_dim // config.n_head
        self.qk_norm = True
        self.use_rope = use_rope

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.hidden_dim)
            self.k_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        # RoPEを使用する場合のみLlamaRotaryEmbeddingを初期化
        if self.use_rope:
            rope_config = LlamaConfig(
                hidden_size=config.hidden_dim,
                num_attention_heads=config.n_head,
                max_position_embeddings=max_position_embeddings,
                head_dim=self.head_dim,
            )
            self.rotary_emb = LlamaRotaryEmbedding(rope_config)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # use_ropeがTrueの場合のみRoPEを適用
        if self.use_rope:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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


class SpatialTransformerBlock(nn.Module):
    """空間的なTransformerブロック（RoPEなし、maskなし）"""

    def __init__(self, config, max_position_embeddings):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config, max_position_embeddings, use_rope=False)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), attn_mask=None)
        x = x + self.mlp(self.ln2(x))
        return x
