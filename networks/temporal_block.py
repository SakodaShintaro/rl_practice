import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet
from fla.models.utils import FLACache
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
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


class CausalTransformerBlock(nn.Module):
    """因果的なTransformerブロック（RoPEあり、内部でcausal maskを適用）"""

    def __init__(self, config, max_position_embeddings):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config, max_position_embeddings, use_rope=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

        # Causal maskを登録
        matrix = torch.tril(torch.ones(max_position_embeddings, max_position_embeddings))
        causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        causal_mask = torch.where(matrix == 1, 0, causal_mask)
        self.register_buffer("causal_mask", causal_mask.contiguous())

    def forward(self, x, rnn_state):
        B, T, C = x.size()
        # 現在のシーケンス長に合わせてmaskを切り出す
        current_mask = self.causal_mask[:T, :T]
        x = x + self.attn(self.ln1(x), attn_mask=current_mask)
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


class GdnBlock(nn.Module):
    """
    GatedDeltaNetブロック（TransformerBlock/MambaBlock/GRUBlockと置き換え可能）

    Args:
        hidden_dim: 隠れ次元数
        num_heads: アテンションヘッド数
        layer_idx: レイヤーインデックス（cacheの管理に使用）
    """

    def __init__(self, hidden_dim, num_heads, layer_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.gdn = GatedDeltaNet(
            hidden_size=hidden_dim,
            head_dim=64,
            num_heads=num_heads,
            mode="chunk",
            layer_idx=layer_idx,
        )

        # per-sample cache size estimates (overridden after first cache update)
        self.recurrent_state_shape = (num_heads, hidden_dim, hidden_dim * 2)
        self.recurrent_state_size = math.prod(self.recurrent_state_shape)
        self.conv_state_shapes = [
            (hidden_dim * num_heads, 4),
            (hidden_dim * num_heads, 4),
            (hidden_dim * num_heads * 2, 4),
        ]
        self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
        self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)
        self._initialized = False

    def _flatten_cache(self, cache_dict):
        """FLACache から取得した辞書を1次元テンソルにフラット化"""
        if cache_dict is None:
            return None

        recurrent_state = cache_dict["recurrent_state"]  # [B, ...]
        conv_state = cache_dict["conv_state"]  # tuple of 3 tensors
        batch_size = recurrent_state.shape[0]

        # 初回実行時に cache のサイズ/形状を取得
        if not self._initialized:
            self.recurrent_state_shape = tuple(recurrent_state.shape[1:])
            self.recurrent_state_size = math.prod(self.recurrent_state_shape)
            self.conv_state_shapes = [tuple(c.shape[1:]) for c in conv_state]
            self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
            self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)
            self._initialized = True

        # recurrent_state/conv_state をバッチ毎にフラット化
        recurrent_flat = recurrent_state.reshape(batch_size, -1).to(torch.float32)
        conv_flat = torch.cat(
            [c.reshape(batch_size, -1).to(torch.float32) for c in conv_state], dim=-1
        )

        cache_flat = torch.cat([recurrent_flat, conv_flat], dim=-1)  # [B, cache_size]
        return cache_flat.unsqueeze(0)  # [1, B, cache_size]

    def _unflatten_cache(self, cache_flat, batch_size, device, dtype):
        """1次元テンソルを FLACache 用の辞書に復元"""
        if cache_flat is None:
            return None

        # cache_flat: [1, B, cache_size]
        cache_flat = cache_flat.squeeze(0)  # [B, cache_size]

        current_batch = cache_flat.size(0)
        if current_batch != batch_size:
            if current_batch == 1:
                cache_flat = cache_flat.expand(batch_size, -1)
            elif current_batch > batch_size:
                cache_flat = cache_flat[:batch_size]
            else:
                repeat = math.ceil(batch_size / current_batch)
                cache_flat = cache_flat.repeat(repeat, 1)[:batch_size]

        # ゼロ tensor かチェック（初期状態）
        if torch.all(cache_flat == 0):
            return None

        # recurrent_state を復元
        recurrent_flat = cache_flat[:, : self.recurrent_state_size]
        recurrent_state = recurrent_flat.view(batch_size, *self.recurrent_state_shape).to(dtype)

        # conv_state を復元
        conv_flat = cache_flat[:, self.recurrent_state_size :]
        conv_tensors = []
        offset = 0
        for i, (size, shape) in enumerate(zip(self.conv_state_sizes, self.conv_state_shapes)):
            conv_tensor = conv_flat[:, offset : offset + size].view(batch_size, *shape).to(dtype)
            conv_tensors.append(conv_tensor)
            offset += size

        cache_dict = {
            "recurrent_state": recurrent_state,
            "attn_state": None,
            "conv_state": tuple(conv_tensors),
            "ffn_state": None,
        }

        # FLACache を作成
        fla_cache = FLACache()
        fla_cache.update(cache_dict, self.layer_idx)

        return fla_cache

    def forward(self, x, attn_mask, rnn_state):
        """
        Args:
            x: [B, T, C]
            attn_mask: 未使用（インターフェース互換性のため）
            rnn_state: [1, B, cache_size] フラット化された cache

        Returns:
            x: [B, T, C]
            rnn_state: [1, B, cache_size] 更新されたフラット化された cache
        """
        B, T, C = x.shape

        # GatedDeltaNetはtraining時にseq_len <= 64だとfused_recurrentモードになりエラーが出るため、
        # 必要に応じてパディングして65以上にする
        if T <= 64:
            padding = torch.zeros(B, 65 - T, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
            need_unpad = True
        else:
            x_padded = x
            need_unpad = False

        # rnn_state をデフラット化して FLACache を作成
        past_key_values = self._unflatten_cache(rnn_state, B, x.device, x.dtype)

        # GatedDeltaNet を実行
        output, _, new_cache = self.gdn(x_padded, past_key_values=past_key_values, use_cache=True)

        # パディングを除去
        if need_unpad:
            output = output[:, :T, :]

        # 新しい cache をフラット化
        if new_cache is not None and len(new_cache) > 0:
            new_cache_dict = new_cache[self.layer_idx]
            new_rnn_state = self._flatten_cache(new_cache_dict)
        else:
            new_rnn_state = rnn_state

        return output, new_rnn_state
