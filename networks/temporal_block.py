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
        self.hidden_dim = config.hidden_dim
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

    def get_rnn_state_size(self):
        """rnn_stateのサイズを返す (Transformerは状態を使わないので hidden_dim を返す)"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        B, T, C = x.size()
        # 現在のシーケンス長に合わせてmaskを切り出す
        current_mask = self.causal_mask[:T, :T]
        x = x + self.attn(self.ln1(x), attn_mask=current_mask)
        x = x + self.mlp(self.ln2(x))
        return x, rnn_state


class MambaBlock(nn.Module):
    """
    Mamba状態空間モデルブロック

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
            - res_drop_prob: residual dropout確率
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
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

    def get_rnn_state_size(self):
        """rnn_stateのサイズを返す (Mambaは状態を使わないので hidden_dim を返す)"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        return self.block(x)[0], rnn_state


class GRUBlock(nn.Module):
    """
    GRUブロック

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

    def get_rnn_state_size(self):
        """rnn_stateのサイズを返す"""
        return self.hidden_dim

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C]
            rnn_state: [1, B, C] GRUの隠れ状態

        Returns:
            x: [B, T, C]
            rnn_state: [1, B, C] 更新されたGRUの隠れ状態
        """
        # repeat/reshapeで非連続になるケースに備えて contiguous を取る
        rnn_state = rnn_state.contiguous()
        x, rnn_state = self.gru(x, rnn_state)
        return x, rnn_state


class GdnBlock(nn.Module):
    """
    GatedDeltaNetブロック

    Args:
        config: Configオブジェクト。必要な属性：
            - hidden_dim: 隠れ次元数
            - n_head: アテンションヘッド数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.n_head
        # GdnBlockは単一レイヤーとして動作するので、常にlayer_idx=0を使用
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
            layer_idx=0,
        )

        # recurrent_state: [B, num_heads, head_dim, head_v_dim]
        self.recurrent_state_shape = (self.num_heads, self.head_dim, self.head_v_dim)
        self.recurrent_state_size = math.prod(self.recurrent_state_shape)

        # conv_state: 3つのShortConvolution state (q, k, v)
        # q_conv1d: [B, key_dim, conv_size], k_conv1d: [B, key_dim, conv_size], v_conv1d: [B, value_dim, conv_size]
        self.conv_size = 4
        self.conv_state_shapes = [
            (self.key_dim, self.conv_size),  # q_conv1d
            (self.key_dim, self.conv_size),  # k_conv1d
            (self.value_dim, self.conv_size),  # v_conv1d
        ]
        self.conv_state_sizes = [math.prod(shape) for shape in self.conv_state_shapes]
        self.cache_size = self.recurrent_state_size + sum(self.conv_state_sizes)

    def get_rnn_state_size(self):
        """rnn_stateのサイズを返す"""
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

        # FLACache を作成（単一レイヤーとして常にlayer_idx=0を使用）
        fla_cache = FLACache()
        fla_cache.update(cache_dict, 0)

        return fla_cache

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, C]
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

        # 新しい cache をフラット化（単一レイヤーとして常にlayer_idx=0を使用）
        if new_cache is not None and len(new_cache) > 0:
            new_cache_dict = new_cache[0]
            new_rnn_state = self._flatten_cache(new_cache_dict)
        else:
            new_rnn_state = rnn_state

        return output, new_rnn_state


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
    # ゼロ初期化された状態を使用（初回実行時を想定）
    rnn_state = torch.zeros(1, B, gdn_block.get_rnn_state_size(), device=device)
    out, new_rnn_state = gdn_block(x, rnn_state)
    print(f"output shape: {out.shape}")
    print(f"new_rnn_state shape: {new_rnn_state.shape}")
    print(f"rnn_state_size: {gdn_block.get_rnn_state_size()}")

    print("\nAll tests passed!")
