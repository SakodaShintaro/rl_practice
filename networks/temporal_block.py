import math

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.models.utils import FLACache
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from .self_attention import Config, SelfAttention


class CausalTransformerBlock(nn.Module):
    """因果的なTransformerブロック（RoPEあり、内部でcausal maskを適用）"""

    def __init__(self, config: Config, max_position_embeddings):
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
