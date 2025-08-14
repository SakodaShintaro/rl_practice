from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 1000.0) -> torch.Tensor:
    """
    2次元回転位置埋め込み（2D Rotary Position Embedding, RoPE）を計算する

    Args:
        dim: 埋め込み次元数（偶数である必要がある）
        end_x: x軸方向の位置数（パッチの幅）
        end_y: y軸方向の位置数（パッチの高さ）
        theta: RoPEの基準周波数（通常1000.0または10000.0）

    Returns:
        torch.Tensor: shape [end_x * end_y, dim] の複素数テンソル（dtype=torch.complex64）
                     各位置(x,y)に対応する回転子を含む。各要素は実部+虚部*jの複素数。
    """
    # 周波数を計算: 低次元ほど低周波数、高次元ほど高周波数
    # dim//2個の異なる周波数を生成（複素数なので実際のdimは2倍になる）
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # x軸、y軸の位置インデックスを作成 [0, 1, 2, ..., end_x-1]
    t_x = torch.arange(end_x, device=freqs_x.device)
    t_y = torch.arange(end_y, device=freqs_y.device)

    # 各位置と各周波数の組み合わせで位相角を計算
    # freqs_x: [end_x, dim//2], freqs_y: [end_y, dim//2]
    freqs_x = torch.outer(t_x, freqs_x).float()
    freqs_y = torch.outer(t_y, freqs_y).float()

    # 複素数として表現: e^(i*θ) = cos(θ) + i*sin(θ)
    # これが回転行列の本質（各次元ペアを回転させる）
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    # 全ての(x,y)位置に対して、x方向とy方向の回転子を連結
    freqs_cis = []
    for i in range(end_x):
        for j in range(end_y):
            # x位置iの回転子とy位置jの回転子を連結してdim次元にする
            freqs_cis.append(torch.cat([freqs_cis_x[i], freqs_cis_y[j]], dim=-1))

    return torch.stack(freqs_cis)


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


class CausalSpaceSelfAttention(nn.Module):
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

    def forward(self, x, attn_mask):
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


class CausalSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


class SpaceSelfAttention(nn.Module):
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

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = (
            F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_rate)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.res_drop_prob(self.proj(y))
        return y


class SpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = SpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


class CausalTimeSelfAttention(nn.Module):
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

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.to(q.dtype), dropout_p=self.attn_dropout_rate
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.res_drop_prob(self.proj(y))
        return y


class CausalTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalTimeSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_time_block = CausalTimeBlock(config)
        self.space_block = SpaceBlock(config)

    def forward(self, x, attn_mask):
        b, f, l, c = x.shape
        x = rearrange(x, "b f l c -> (b l) f c")
        x = self.causal_time_block(x, attn_mask)
        x = rearrange(x, "(b l) f c -> (b f) l c", b=b, l=l, f=f)
        x = self.space_block(x)
        x = rearrange(x, "(b f) l c -> b f l c", b=b, f=f)
        return x


class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        n_layer,
        time_len,
        hidden_dim,
        n_head,
        attn_drop_prob,
        res_drop_prob,
    ):
        super().__init__()
        config = Config(
            hidden_dim=hidden_dim,
            n_head=n_head,
            attn_drop_prob=attn_drop_prob,
            res_drop_prob=res_drop_prob,
        )

        self.hidden_dim = hidden_dim
        self.causal_time_space_num = n_layer

        self.time_emb = nn.Parameter(torch.zeros(50, self.hidden_dim))
        nn.init.normal_(self.time_emb.data, mean=0, std=0.02)

        self.causal_time_space_blocks = nn.Sequential(
            *[CausalTimeSpaceBlock(config) for _ in range(self.causal_time_space_num)]
        )

        self.apply(self._init_weights)

        matrix = torch.tril(torch.ones(time_len, time_len))
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

    def forward(self, feature_embeddings):
        """
        Forward pass of the SpatialTemporalTransformer.

        Args:
            feature_embeddings: [B, T, S, C]

        Returns:
            torch.Tensor: shape [B, T, S, C]
        """
        _, T, S, _ = feature_embeddings.shape

        time_emb_T = self.time_emb[:T, :].unsqueeze(0)
        time_emb_T = torch.repeat_interleave(time_emb_T[:, :, None, :], S, dim=2)

        time_space_token_embeddings = feature_embeddings + time_emb_T

        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](
                time_space_token_embeddings, self.mask_time
            )

        return time_space_token_embeddings
