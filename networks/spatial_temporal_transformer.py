# ref. https://github.com/Kevin-thu/Epona/blob/main/models/stt.py

import torch
import torch.nn as nn
from einops import rearrange

from .temporal_block import (
    CausalTransformerBlock,
    Config,
    GdnBlock,
    GRUBlock,
    MambaBlock,
    SpatialTransformerBlock,
)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, config, temporal_model_type, layer_idx, tempo_len, space_len):
        super().__init__()
        self.temporal_model_type = temporal_model_type

        if temporal_model_type == "mamba":
            self.tempo_block = MambaBlock(config)
        elif temporal_model_type == "transformer":
            self.tempo_block = CausalTransformerBlock(config, tempo_len)
        elif temporal_model_type == "gru":
            self.tempo_block = GRUBlock(config)
        elif temporal_model_type == "gdn":
            self.tempo_block = GdnBlock(config.hidden_dim, config.n_head, layer_idx)
        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")
        self.space_block = SpatialTransformerBlock(config, space_len)

    def forward(self, x, rnn_state):
        """
        Args:
            x: [B, T, S, C]
            rnn_state: [1, B*S, C] GRU hidden state (only for temporal_model_type=="gru"), or None

        Returns:
            x: [B, T, S, C]
            rnn_state: [1, B*S, C] updated GRU hidden state (only for temporal_model_type=="gru"), or None
        """
        b, t, s, c = x.shape
        x = rearrange(x, "b t s c -> (b s) t c")
        x, rnn_state = self.tempo_block(x, rnn_state)
        x = rearrange(x, "(b s) t c -> (b t) s c", b=b, s=s, t=t)
        x = self.space_block(x)
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

        # tempo_emb削除、space_embのみ保持
        self.space_emb = nn.Parameter(torch.zeros(1, 1, space_len, self.hidden_dim))
        nn.init.normal_(self.space_emb.data, mean=0, std=0.02)

        self.spatial_temporal_blocks = nn.ModuleList(
            [
                SpatialTemporalBlock(
                    config,
                    temporal_model_type,
                    layer_idx=i,
                    tempo_len=tempo_len,
                    space_len=space_len,
                )
                for i in range(self.n_layer)
            ]
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
            rnn_state: [1, B, n_layer * S * C] (for GRU),
                [1, B, n_layer * S * cache_size] (for GDN) or [1, B, 1] (others)

        Returns:
            out: [B, T, S, C]
            rnn_state: [1, B, n_layer * S * C] (for GRU) or same as input (for others)
        """
        B, T, S, C = x.shape

        space_emb = torch.repeat_interleave(self.space_emb, T, dim=1)

        out = x + space_emb

        # rnn_stateを各レイヤーの状態に分割
        # GRUの場合: [1, B, n_layer * S * C] -> n_layer個の [1, B*S, C]
        # GDNの場合: [1, B, n_layer * S * cache_size] -> n_layer個の [1, B*S, cache_size]
        # それ以外: rnn_stateをそのまま使用（各ブロックが無視する）
        if self.temporal_model_type == "gru":
            # [1, B, n_layer * S * C] -> [B, n_layer * S * C]
            rnn_state_squeezed = rnn_state.squeeze(0)
            # [B, n_layer * S * C] -> [B, n_layer, S, C]
            layer_states = rnn_state_squeezed.view(B, self.n_layer, S, C)
            layer_states = [layer_states[:, i, :, :].contiguous() for i in range(self.n_layer)]
            # 各レイヤーの状態を [1, B*S, C] に変形
            layer_states = [s.view(1, B * S, C) for s in layer_states]
        elif self.temporal_model_type == "gdn":
            # cache size は GDN の実行後に確定するため、テンソルの実サイズから求める
            rnn_state_squeezed = rnn_state.squeeze(0)
            total_cache_size = rnn_state_squeezed.shape[-1]
            denom = self.n_layer * S
            cache_size = total_cache_size // denom
            if cache_size * denom != total_cache_size:
                raise ValueError(
                    "GDN rnn_state size "
                    f"{total_cache_size} is not divisible by n_layer * space_len={denom}"
                )
            # [B, n_layer * S * cache_size] -> [B, n_layer, S, cache_size]
            layer_states = rnn_state_squeezed.view(B, self.n_layer, S, cache_size)
            layer_states = [
                layer_states[:, i, :, :].reshape(1, B * S, cache_size) for i in range(self.n_layer)
            ]
        else:
            layer_states = [None] * self.n_layer

        new_layer_states = []
        for i in range(self.n_layer):
            out, new_state = self.spatial_temporal_blocks[i](out, layer_states[i])
            new_layer_states.append(new_state)

        # 各レイヤーの状態を結合
        # GRUの場合: n_layer個の [1, B*S, C] -> [1, B, n_layer * S * C]
        # GDNの場合: n_layer個の [1, B*S, cache_size] -> [1, B, n_layer * S * cache_size]
        # それ以外: 入力のrnn_stateをそのまま返す
        if self.temporal_model_type == "gru":
            new_layer_states = [s.view(B, S, C) for s in new_layer_states]
            # [B, n_layer, S, C] -> [B, n_layer * S * C] -> [1, B, n_layer * S * C]
            rnn_state = (
                torch.stack(new_layer_states, dim=1).view(B, self.n_layer * S * C).unsqueeze(0)
            )
        elif self.temporal_model_type == "gdn":
            # [1, B*S, cache_size] -> [B, S, cache_size]
            cache_size = new_layer_states[0].shape[-1]
            new_layer_states = [s.squeeze(0).view(B, S, cache_size) for s in new_layer_states]
            # [B, n_layer, S, cache_size] -> [1, B, n_layer * S * cache_size]
            rnn_state = (
                torch.stack(new_layer_states, dim=1)
                .reshape(B, self.n_layer * S * cache_size)
                .unsqueeze(0)
            )

        return out, rnn_state
