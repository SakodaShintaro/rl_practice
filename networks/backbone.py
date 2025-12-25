import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .image_processor import ImageProcessor
from .reward_processor import RewardProcessor
from .self_attention import get_fourier_embeds_from_coordinates
from .spatial_temporal_transformer import SpatialTemporalTransformer
from .temporal_block import CausalTransformerBlock, GdnBlock, GRUBlock, MambaBlock


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SpatialTemporalEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(
        self,
        image_processor: ImageProcessor,
        reward_processor: RewardProcessor,
        seq_len: int,
        n_layer: int,
        action_dim: int,
        temporal_model_type: str,
        use_image_only: bool,
    ):
        super().__init__()

        self.use_image_only = use_image_only
        self.n_layer = n_layer
        self.temporal_model_type = temporal_model_type

        self.image_processor = image_processor
        self.reward_processor = reward_processor
        self.freeze_image_processor = image_processor.processor_type == "ae"

        # image_processor outputs [B, C, H, W] -> treat as [B, H * W, C] (H * W tokens, C channels each)
        self.hidden_image_dim = self.image_processor.output_shape[0]
        self.hidden_h = self.image_processor.output_shape[1]
        self.hidden_w = self.image_processor.output_shape[2]
        self.image_tokens_num = self.hidden_h * self.hidden_w
        action_tokens_num = action_dim
        reward_tokens_num = 1
        register_tokens_num = 1

        self.space_len = (
            self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
        )

        self.spatial_temporal = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=self.space_len,
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            attn_drop_prob=0.0,
            temporal_model_type=temporal_model_type,
        )

        token_num = (
            self.image_tokens_num
            if self.use_image_only
            else (
                self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
            )
        )

        self.output_dim = self.hidden_image_dim * token_num

    def init_state(self) -> torch.Tensor:
        # ブロックのget_rnn_state_size()を使用してstate_sizeを取得
        state_size = self.spatial_temporal.spatial_temporal_blocks[
            0
        ].tempo_block.get_rnn_state_size()
        # [1, space_len, state_size, n_layer] (バッチサイズ1)
        return torch.zeros(1, self.space_len, state_size, self.n_layer)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        obs_z: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        actions: torch.Tensor,  #  (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, space_len, state_size, n_layer)
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (B, space_len, state_size, n_layer)
            action_text: str (always empty string for non-VLM encoders)
        """
        # Encode all frames with AE but preserve spatial structure
        # images: (B, T, C, H, W) -> encode all frames
        B, T = images.shape[:2]

        # 外部形式 [B, space_len, state_size, n_layer] -> 内部形式 [1, B*space_len, state_size, n_layer]
        rnn_state_internal = rnn_state.reshape(1, B * self.space_len, -1, self.n_layer)

        # Use pre-encoded obs_z if using ae, otherwise encode from images
        if self.freeze_image_processor:
            all_latents = obs_z  # [B, T, C', H', W']
        else:
            # Reshape to process all frames: (B*T, C, H, W)
            all_frames = images.reshape(-1, *images.shape[2:])
            # Encode all frames at once
            all_latents = self.image_processor.encode(all_frames)  # [B*T, C', H', W']
            all_latents = all_latents.reshape(B, T, *all_latents.shape[1:])  # [B, T, C', H', W']
        image_embed = all_latents.flatten(3).transpose(2, 3)  # [B, T, S(=H'*W'), C']

        # [B, T, action_dim] -> [B, T, action_dim, C']
        action_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, actions)

        # [B, T, 1] -> [B, T, 1, C']
        reward_embed = self.reward_processor.encode(rewards)  # (B, T, 1, C')

        # [B, T, 1, C']
        register_token = torch.zeros(
            (B, T, 1, self.hidden_image_dim), device=images.device, dtype=images.dtype
        )

        # [B, T, S+action_dim+1+1, C']
        all_embed = torch.cat([image_embed, action_embed, reward_embed, register_token], dim=2)

        # Apply STT to all frames
        spatial_temporal_output, rnn_state_internal = self.spatial_temporal(
            all_embed, rnn_state_internal
        )  # [B, T, S+action_dim+1, C'], [1, B*space_len, state_size, n_layer]

        # 内部形式 [1, B*space_len, state_size, n_layer] -> 外部形式 [B, space_len, state_size, n_layer]
        rnn_state = rnn_state_internal.reshape(B, self.space_len, -1, self.n_layer)

        # Use last timestep's image tokens for final representation
        last_frame_emb = spatial_temporal_output[:, -1, :, :]  # [B, S+action_dim+1, C']

        if self.use_image_only:
            last_frame_emb = last_frame_emb[:, : self.image_tokens_num, :]  # [B, S, C']

        output = last_frame_emb.flatten(start_dim=1)  # [B, token_num * C']

        return output, rnn_state, ""


class TemporalOnlyEncoder(nn.Module):
    """
    統合された時系列エンコーダー
    RecurrentEncoderとSimpleTransformerEncoderを統合し、柔軟な設定を可能にする

    Args:
        image_processor: 画像プロセッサのインスタンス
        reward_processor: 報酬プロセッサのインスタンス
        seq_len: シーケンス長
        n_layer: レイヤー数 (GRUならnum_layers、transformerならブロック数)
        action_dim: アクションの次元 (未使用)
        temporal_model_type: 時系列モデルのタイプ ("gru" or "transformer")
        use_image_only: 画像のみを使うか（Falseならaction, rewardも入れる）
    """

    def __init__(
        self,
        image_processor: ImageProcessor,
        reward_processor: RewardProcessor,
        seq_len: int,
        n_layer: int,
        action_dim: int,
        temporal_model_type: str,
        use_image_only: bool,
    ):
        super().__init__()

        self.n_layer = n_layer
        self.temporal_model_type = temporal_model_type
        self.freeze_image_processor = image_processor.processor_type == "ae"
        self.use_image_only = use_image_only

        # Image processor
        self.image_processor = image_processor
        self.reward_processor = reward_processor

        # 画像特徴量の次元を計算
        image_feature_dim = np.prod(self.image_processor.output_shape)

        # 共通の隠れ層サイズ
        self.output_dim = 64

        # image_processor後の共通線形層
        self.lin_hidden_in = nn.Linear(image_feature_dim, self.output_dim)

        # 時系列ブロックのリストを作成
        max_seq_len = seq_len if use_image_only else seq_len * 3
        hidden_dim = self.output_dim
        n_head = 8
        attn_drop_prob = 0.0

        if temporal_model_type == "gru":
            self.blocks = nn.ModuleList([GRUBlock(hidden_dim) for _ in range(n_layer)])
        elif temporal_model_type == "transformer":
            self.blocks = nn.ModuleList(
                [
                    CausalTransformerBlock(hidden_dim, n_head, attn_drop_prob, max_seq_len)
                    for _ in range(n_layer)
                ]
            )
        elif temporal_model_type == "gdn":
            self.blocks = nn.ModuleList([GdnBlock(hidden_dim, n_head) for _ in range(n_layer)])
        elif temporal_model_type == "mamba":
            self.blocks = nn.ModuleList([MambaBlock(hidden_dim) for _ in range(n_layer)])
        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")

    def init_state(self) -> torch.Tensor:
        # ブロックのget_rnn_state_size()を使用してstate_sizeを取得
        state_size = self.blocks[0].get_rnn_state_size()
        # [1, state_size, n_layer]を返す（バッチサイズ1）
        return torch.zeros(1, state_size, self.n_layer)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        obs_z: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (B, state_size, n_layer)
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (B, state_size, n_layer)
            action_text: str (always empty string for non-VLM encoders)
        """
        B, T = images.shape[:2]

        # Use pre-encoded obs_z if using ae, otherwise encode from images
        if self.freeze_image_processor:
            image_features = obs_z.reshape(B * T, *obs_z.shape[2:])  # (B*T, C', H', W')
        else:
            # 画像を処理
            all_frames = images.reshape(B * T, *images.shape[2:])  # (B*T, C, H, W)
            image_features = self.image_processor.encode(all_frames)

        # Flatten and linear projection
        h = image_features.flatten(start_dim=1)  # (B*T, feature_dim)
        h = self.lin_hidden_in(h)  # (B*T, hidden_size)
        h = h.reshape(B, T, -1)  # (B, T, hidden_size)

        if self.use_image_only:
            # 画像のみ
            sequence = h  # (B, T, d_model)
        else:
            # action, rewardを埋め込み
            action_emb = get_fourier_embeds_from_coordinates(
                self.output_dim, actions
            )  # (B, T, action_dim, d_model)
            action_emb = action_emb.sum(dim=2)  # (B, T, d_model)

            reward_emb = get_fourier_embeds_from_coordinates(
                self.output_dim, rewards
            )  # (B, T, 1, d_model)
            reward_emb = reward_emb.sum(dim=2)  # (B, T, d_model)

            # Interleave: [a_0, r_0, s_0, a_1, r_1, s_1, ...]
            sequence = torch.stack([action_emb, reward_emb, h], dim=2)  # (B, T, 3, d_model)
            sequence = sequence.view(B, T * 3, self.output_dim)  # (B, T*3, d_model)

        # 各レイヤーの状態を分割: [B, state_size, n_layer] -> n_layer個の [B, state_size]
        layer_states = [rnn_state[:, :, i] for i in range(self.n_layer)]

        new_layer_states = []
        for i, block in enumerate(self.blocks):
            # 外部形式 [B, state_size] -> 内部形式 [1, B, state_size]
            layer_state_internal = layer_states[i].unsqueeze(0)
            sequence, layer_state_internal = block(sequence, layer_state_internal)
            # 内部形式 [1, B, state_size] -> 外部形式 [B, state_size]
            new_layer_states.append(layer_state_internal.squeeze(0))

        # 各レイヤーの状態を結合: n_layer個の [B, state_size] -> [B, state_size, n_layer]
        rnn_state = torch.stack(new_layer_states, dim=-1)

        # 最後のトークンの表現
        final_repr = sequence[:, -1, :]  # (B, d_model)

        return final_repr, rnn_state, ""
