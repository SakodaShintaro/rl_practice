import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BaseCNN
from .diffusion_policy import TimestepEmbedder

"""
観測 o_t
報酬 r_t
行動 a_t
の系列を圧縮するためのTransformerベースのネットワーク
o_t, r_t, a_tはそれぞれ同じ次元(HIDDEN_DIM)に変換される
"""


class SequenceCompressor(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = 256

        # 状態(画像)エンコーダー
        self.state_encoder = BaseCNN(in_channels=3)

        # 報酬エンコーダー
        self.reward_encoder = TimestepEmbedder(self.hidden_dim)

        # 行動エンコーダー
        self.action_encoder = nn.Linear(3, self.hidden_dim)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len * 3, self.hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 出力層（最後のトークンまたはCLSトークンを使用）
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self, states: torch.Tensor, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            states (torch.Tensor): 過去の状態(画像)シーケンス (batch_size, seq_len, C, H, W)
            rewards (torch.Tensor): 過去の報酬シーケンス (batch_size, seq_len, reward_dim)
            actions (torch.Tensor): 過去の行動シーケンス (batch_size, seq_len, action_dim)

        Returns:
            torch.Tensor: 圧縮表現 (batch_size, hidden_dim)
        """
        batch_size = states.shape[0]

        # 状態(画像)をエンコード (batch_size * seq_len, C, H, W) -> (batch_size * seq_len, state_embed_dim)
        states_flat = states.view(-1, *states.shape[2:])
        state_embeds_flat = self.state_encoder(states_flat)
        state_embeds = state_embeds_flat.view(batch_size, self.seq_len, self.hidden_dim)

        # 報酬をエンコード (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        rewards_embeds = self.reward_encoder(rewards)
        rewards_embeds = rewards_embeds.view(batch_size, self.seq_len, self.hidden_dim)

        # 行動をエンコード (batch_size, seq_len, action_dim) -> (batch_size, seq_len, hidden_dim)
        action_embeds = self.action_encoder(actions)

        # エンコードされた状態、報酬、行動を結合
        x = torch.stack((state_embeds, rewards_embeds, action_embeds), dim=2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.seq_len * 3, self.hidden_dim)

        # Positional Encodingを追加
        x += self.pos_embedding[:, : self.seq_len * 3]

        # Transformer Encoderに通す
        transformer_output = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)

        # 最後のトークンの出力を圧縮表現とする
        compressed_representation = transformer_output[:, -1, :]  # (batch_size, hidden_dim)

        # 出力層を通す
        compressed_representation = self.norm(compressed_representation)
        compressed_representation = F.relu(compressed_representation)
        compressed_representation = self.output_layer(compressed_representation)

        return compressed_representation
