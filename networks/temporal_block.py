import math

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.models.utils import FLACache
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from .self_attention import Config, SelfAttention


class BaseTemporalBlock(nn.Module):
    """
    時系列処理 + MLP の汎用Transformerブロック

    Args:
        config: Configオブジェクト
        temporal_layer: 時系列処理を行うレイヤー
        use_rms_norm: RMSNormを使用するか（False=LayerNorm）
    """

    def __init__(self, config, temporal_layer, use_rms_norm):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # 正規化レイヤーの選択
        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm

        # 第1ブロック: Norm + TemporalLayer + Residual
        self.temporal_norm = norm_cls(config.hidden_dim)
        self.temporal = temporal_layer

        # 第2ブロック: Norm + MLP + Residual
        self.mlp_norm = norm_cls(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False),
            nn.Dropout(config.res_drop_prob),
        )

    def get_rnn_state_size(self):
        """rnn_stateのサイズを返す"""
        return self.temporal.get_rnn_state_size()

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C]
            rnn_state: [1, B, rnn_state_size]

        Returns:
            x: [B, T, C]
            rnn_state: [1, B, rnn_state_size]
        """
        # 第1ブロック: Norm + TemporalLayer + Residual
        temporal_output, new_rnn_state = self.temporal(self.temporal_norm(x), rnn_state)
        x = x + temporal_output

        # 第2ブロック: Norm + MLP + Residual
        x = x + self.mlp(self.mlp_norm(x))

        return x, new_rnn_state


class SelfAttentionLayer(nn.Module):
    """
    Self-Attentionによる時系列処理レイヤー

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
            - n_head: アテンションヘッド数
            - attn_drop_prob: アテンションdropout確率
            - max_position_embeddings: 最大位置埋め込み数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.attn = SelfAttention(config, config.max_position_embeddings, use_rope=True)

        # Causal maskを登録
        max_pos = config.max_position_embeddings
        matrix = torch.tril(torch.ones(max_pos, max_pos))
        causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        causal_mask = torch.where(matrix == 1, 0, causal_mask)
        self.register_buffer("causal_mask", causal_mask.contiguous())

    def get_rnn_state_size(self):
        """状態を使わないのでhidden_dimを返す"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C] 正規化済み入力
            rnn_state: [1, B, C] 未使用

        Returns:
            output: [B, T, C]
            rnn_state: [1, B, C] 未変更
        """
        B, T, C = x.size()
        current_mask = self.causal_mask[:T, :T]
        output = self.attn(x, attn_mask=current_mask)
        return output, rnn_state


class MambaLayer(nn.Module):
    """
    Mamba2による時系列処理レイヤー

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        d_ssm = 2 * config.hidden_dim
        self.mamba = Mamba2(d_model=config.hidden_dim, headdim=d_ssm)

    def get_rnn_state_size(self):
        """状態を使わないのでhidden_dimを返す"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C] 正規化済み入力
            rnn_state: [1, B, C] 未使用

        Returns:
            output: [B, T, C]
            rnn_state: [1, B, C] 未変更
        """
        output = self.mamba(x)
        return output, rnn_state


class GRULayer(nn.Module):
    """
    GRUによる時系列処理レイヤー

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, num_layers=1, batch_first=True)

    def get_rnn_state_size(self):
        """GRUの状態サイズを返す"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C] 正規化済み入力
            rnn_state: [1, B, C] GRUの隠れ状態

        Returns:
            output: [B, T, C]
            rnn_state: [1, B, C] 更新されたGRUの隠れ状態
        """
        # repeat/reshapeで非連続になるケースに備えて contiguous を取る
        rnn_state = rnn_state.contiguous()
        output, new_rnn_state = self.gru(x, rnn_state)
        return output, new_rnn_state


class GatedDeltaNetLayer(nn.Module):
    """
    GatedDeltaNetによる時系列処理レイヤー

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
            - n_head: アテンションヘッド数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.n_head
        # GatedDeltaNetのデフォルト設定: head_dim=64, expand_v=2.0
        self.head_dim = 64
        self.expand_v = 2.0
        self.head_v_dim = int(self.head_dim * self.expand_v)  # 128
        self.key_dim = self.num_heads * self.head_dim  # num_heads * 64
        self.value_dim = self.num_heads * self.head_v_dim  # num_heads * 128

        self.gdn = GatedDeltaNet(
            hidden_size=config.hidden_dim,
            head_dim=self.head_dim,
            num_heads=config.n_head,
            mode="chunk",
        )

        # recurrent_state: [B, num_heads, head_dim, head_v_dim]
        self.recurrent_state_shape = (self.num_heads, self.head_dim, self.head_v_dim)
        self.recurrent_state_size = math.prod(self.recurrent_state_shape)

        # conv_state: 3つのShortConvolution state (q, k, v)
        self.conv_size = 4
        self.conv_state_shapes = [
            (self.key_dim, self.conv_size),  # q_conv1d
            (self.key_dim, self.conv_size),  # k_conv1d
            (self.value_dim, self.conv_size),  # v_conv1d
        ]
        self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
        self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)

    def get_rnn_state_size(self):
        """GatedDeltaNetのキャッシュサイズを返す"""
        return self.cache_size

    def _flatten_cache(self, cache_dict):
        """FLACache から取得した辞書を1次元テンソルにフラット化"""
        if cache_dict is None:
            return None

        recurrent_state = cache_dict["recurrent_state"]  # [B, num_heads, head_dim, head_v_dim]
        conv_state = cache_dict["conv_state"]  # tuple of 3 tensors
        batch_size = recurrent_state.shape[0]

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
            "attn_state": (),
            "conv_state": tuple(conv_tensors),
            "ffn_state": (),
        }

        # FLACache を作成
        fla_cache = FLACache()
        fla_cache.update(cache_dict)

        return fla_cache

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C] 正規化済み入力
            rnn_state: [1, B, cache_size] フラット化されたキャッシュ

        Returns:
            output: [B, T, C]
            rnn_state: [1, B, cache_size] 更新されたキャッシュ
        """
        B, T, C = x.shape

        # GatedDeltaNetはtraining時にseq_len <= 64だとエラーが出るため、
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
        gdn_output, _, new_cache = self.gdn(
            x_padded, past_key_values=past_key_values, use_cache=True
        )

        # パディングを除去
        if need_unpad:
            gdn_output = gdn_output[:, :T, :]

        # 新しい cache をフラット化
        if new_cache is not None and len(new_cache) > 0:
            new_cache_dict = new_cache[0]
            new_rnn_state = self._flatten_cache(new_cache_dict)
        else:
            new_rnn_state = rnn_state

        return gdn_output, new_rnn_state


class IdentityLayer(nn.Module):
    """
    何もしない時系列処理レイヤー（比較用）

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim

    def get_rnn_state_size(self):
        """状態を使わないのでhidden_dimを返す"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C] 正規化済み入力
            rnn_state: [1, B, C] 未使用

        Returns:
            output: [B, T, C] 入力をそのまま返す
            rnn_state: [1, B, C] 未変更
        """
        return x, rnn_state


# 互換性のためのエイリアス
def CausalTransformerBlock(config, max_position_embeddings):
    """因果的なTransformerブロック（RoPEあり、内部でcausal maskを適用）"""
    # configにmax_position_embeddingsを追加
    config.max_position_embeddings = max_position_embeddings
    return BaseTemporalBlock(config, SelfAttentionLayer(config), use_rms_norm=False)


def MambaBlock(config):
    """Mamba状態空間モデルブロック（Transformerブロックと同じ構造）"""
    return BaseTemporalBlock(config, MambaLayer(config), use_rms_norm=True)


def GRUBlock(config):
    """GRUブロック（Transformerブロックと同じ構造）"""
    return BaseTemporalBlock(config, GRULayer(config), use_rms_norm=False)


def GdnBlock(config):
    """GatedDeltaNetブロック（Transformerブロックと同じ構造）"""
    return BaseTemporalBlock(config, GatedDeltaNetLayer(config), use_rms_norm=False)


def IdentityBlock(config):
    """何もしないブロック（比較用）"""
    return BaseTemporalBlock(config, IdentityLayer(config), use_rms_norm=False)


if __name__ == "__main__":
    # networksディレクトリで直接実行できるようにimportパスを調整
    import sys
    from pathlib import Path

    # 親ディレクトリ（rl_practice）をsys.pathに追加
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))

    # 相対importではなく絶対importで再読み込み
    from networks.self_attention import Config, SelfAttention

    # 動作確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, T, C = 2, 10, 16
    x = torch.randn(B, T, C, device=device)

    config = Config(hidden_dim=C, n_head=4, attn_drop_prob=0.1, res_drop_prob=0.1)

    # CausalTransformerBlock
    print("\n=== CausalTransformerBlock ===")
    block = CausalTransformerBlock(config, max_position_embeddings=512).to(device)
    rnn_state = torch.zeros(1, B, block.get_rnn_state_size(), device=device)
    out, new_rnn_state = block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {block.get_rnn_state_size()}")

    # GRUBlock
    print("\n=== GRUBlock ===")
    gru_block = GRUBlock(config).to(device)
    rnn_state = torch.zeros(1, B, gru_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = gru_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {gru_block.get_rnn_state_size()}")

    # MambaBlock
    print("\n=== MambaBlock ===")
    mamba_block = MambaBlock(config).to(device)
    rnn_state = torch.zeros(1, B, mamba_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = mamba_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {mamba_block.get_rnn_state_size()}")

    # GdnBlock
    print("\n=== GdnBlock ===")
    gdn_block = GdnBlock(config).to(device)
    rnn_state = torch.zeros(1, B, gdn_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = gdn_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {gdn_block.get_rnn_state_size()}")

    # IdentityBlock
    print("\n=== IdentityBlock ===")
    identity_block = IdentityBlock(config).to(device)
    rnn_state = torch.zeros(1, B, identity_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = identity_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {identity_block.get_rnn_state_size()}")

    print("\nAll tests passed!")
