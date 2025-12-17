import numpy as np
import torch
from diffusers.models import AutoencoderTiny
from torch import nn
from torch.nn import functional as F

from .spatial_temporal_transformer import (
    Config,
    GdnBlock,
    SpatialTemporalTransformer,
    TransformerBlock,
    get_fourier_embeds_from_coordinates,
)


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


class ImageProcessor(nn.Module):
    def __init__(self, observation_space_shape: tuple[int], processor_type: str) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.processor_type = processor_type
        if processor_type == "ae":
            self.processor = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", cache_dir="./cache"
            )
        elif processor_type == "simple_cnn":
            self.processor = nn.Sequential(
                nn.Conv2d(observation_space_shape[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 0),
                nn.ReLU(),
            )
        self.output_shape = self._get_conv_output_shape(observation_space_shape)

    def _get_conv_output_shape(self, shape: tuple[int]) -> tuple[int]:
        x = torch.zeros(1, *shape)
        if self.processor_type == "ae":
            x = self.processor.encode(x).latents
        elif self.processor_type == "simple_cnn":
            x = self.processor(x)
        return list(x.size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.processor_type == "ae":
            x = self.processor.encode(x).latents
        elif self.processor_type == "simple_cnn":
            x = self.processor(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.processor_type == "ae":
            x = x.view(x.size(0), *self.output_shape)
            x = self.processor.decode(x).sample
        elif self.processor_type == "simple_cnn":
            x = torch.zeros(x.size(0), *self.observation_space_shape, device=x.device)
        return x


class SpatialTemporalEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(
        self,
        observation_space_shape: tuple[int],
        seq_len: int,
        n_layer: int,
        action_dim: int,
        temporal_model_type: str,
        image_processor_type: str,
        use_image_only: bool,
    ):
        super().__init__()

        self.use_image_only = use_image_only
        self.n_layer = n_layer
        self.temporal_model_type = temporal_model_type

        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=image_processor_type
        )
        self.freeze_image_processor = image_processor_type == "ae"

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
            res_drop_prob=0.0,
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
        if self.temporal_model_type == "gdn":
            # GDN の cache サイズ: recurrent_state + conv_state
            # recurrent_state: [num_heads, hidden_dim, hidden_dim*2] (per spatial token)
            # conv_state: 3 * [hidden_dim, 4]
            # これが n_layer * space_len 分必要
            num_heads = 1
            cache_size_per_block = (
                num_heads * self.hidden_image_dim * self.hidden_image_dim * 2
                + 3 * self.hidden_image_dim * 4
            )
            total_cache_size = self.n_layer * self.space_len * cache_size_per_block
            # [1, 1, total_cache_size]
            return torch.zeros(1, 1, total_cache_size)
        else:
            # [1, 1, n_layer * space_len * hidden_image_dim]
            return torch.zeros(1, 1, self.n_layer * self.space_len * self.hidden_image_dim)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        obs_z: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        actions: torch.Tensor,  #  (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (1, B, n_layer * space_len * hidden_image_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (1, B, n_layer * space_len * hidden_image_dim)
            action_text: str (always empty string for non-VLM encoders)
        """
        # Encode all frames with AE but preserve spatial structure
        # images: (B, T, C, H, W) -> encode all frames
        B, T = images.shape[:2]

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
        reward_embed = get_fourier_embeds_from_coordinates(self.hidden_image_dim, rewards)

        # [B, T, 1, C']
        register_token = torch.zeros(
            (B, T, 1, self.hidden_image_dim), device=images.device, dtype=images.dtype
        )

        # [B, T, S+action_dim+1+1, C']
        all_embed = torch.cat([image_embed, action_embed, reward_embed, register_token], dim=2)

        # Apply STT to all frames
        spatial_temporal_output, rnn_state = self.spatial_temporal(
            all_embed, rnn_state
        )  # [B, T, S+action_dim+1, C']

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
        observation_space_shape: 観測空間の形状 [C, H, W]
        seq_len: シーケンス長
        n_layer: レイヤー数 (GRUならnum_layers、transformerならブロック数)
        action_dim: アクションの次元 (未使用)
        temporal_model_type: 時系列モデルのタイプ ("gru" or "transformer")
        image_processor_type: 画像プロセッサのタイプ ("ae" or "simple_cnn")
        use_image_only: 画像のみを使うか（Falseならaction, rewardも入れる）
    """

    def __init__(
        self,
        observation_space_shape: tuple[int],
        seq_len: int,
        n_layer: int,
        action_dim: int,
        temporal_model_type: str,
        image_processor_type: str,
        use_image_only: bool,
    ):
        super().__init__()

        self.temporal_model_type = temporal_model_type
        self.freeze_image_processor = image_processor_type == "ae"
        self.use_image_only = use_image_only

        # Image processor
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=image_processor_type
        )

        # 画像特徴量の次元を計算
        image_feature_dim = np.prod(self.image_processor.output_shape)

        # 共通の隠れ層サイズ
        self.output_dim = 256

        # image_processor後の共通線形層
        self.lin_hidden_in = nn.Linear(image_feature_dim, self.output_dim)

        # 時系列モデルのタイプによって構造を変える
        if temporal_model_type == "gru":
            # GRUベースの実装
            self.recurrent_layer = nn.GRU(
                self.output_dim, self.output_dim, num_layers=n_layer, batch_first=True
            )

        elif temporal_model_type == "transformer":
            # Transformerベースの実装 (SimpleTransformerEncoderと同様)
            # シーケンス長の計算 (action, rewardを含む場合は3倍)
            max_seq_len = seq_len if use_image_only else seq_len * 3

            n_heads = 8

            # Positional encoding
            self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, self.output_dim))

            # TransformerBlocks
            config = Config(
                hidden_dim=self.output_dim,
                n_head=n_heads,
                attn_drop_prob=0.0,
                res_drop_prob=0.0,
            )
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_layer)])

            # Causal mask
            matrix = torch.tril(torch.ones(max_seq_len, max_seq_len))
            self.causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
            self.causal_mask = torch.where(matrix == 1, 0, self.causal_mask)

        elif temporal_model_type == "gdn":
            self.gdn_blocks = nn.ModuleList(
                [GdnBlock(self.output_dim, num_heads=8, layer_idx=i) for i in range(n_layer)]
            )

        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")

    def init_state(self) -> torch.Tensor:
        if self.temporal_model_type == "gdn":
            # GDN の cache サイズ: recurrent_state + conv_state
            # recurrent_state: [num_heads, hidden_dim, hidden_dim*2]
            # conv_state: 3 * [hidden_dim, 4]
            num_heads = 8
            cache_size = num_heads * self.output_dim * self.output_dim * 2 + 3 * self.output_dim * 4
            return torch.zeros(1, 1, cache_size)
        else:
            return torch.zeros(1, 1, self.output_dim)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        obs_z: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (1, B, hidden_size) for GRU, unused for Transformer
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (1, B, hidden_size) for GRU, same as input for Transformer
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
        h = F.relu(self.lin_hidden_in(h))  # (B*T, hidden_size)
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

            # Interleave: [s_0, a_0, r_0, s_1, a_1, r_1, ...]
            sequence = torch.stack([h, action_emb, reward_emb], dim=2)  # (B, T, 3, d_model)
            sequence = sequence.view(B, T * 3, self.output_dim)  # (B, T*3, d_model)

        # 時系列モデルによって処理を分岐
        if self.temporal_model_type == "gru":
            sequence, rnn_state = self.recurrent_layer(
                sequence, rnn_state
            )  # (B, T, hidden_size), (1, B, hidden_size)

        elif self.temporal_model_type == "transformer":
            # Positional encoding
            actual_seq_len = sequence.size(1)
            sequence = sequence + self.pos_encoding[:actual_seq_len].unsqueeze(0)

            # Apply Transformer blocks
            current_mask = self.causal_mask[:actual_seq_len, :actual_seq_len].to(sequence.device)
            for block in self.blocks:
                sequence, _ = block(sequence, current_mask, None)

        elif self.temporal_model_type == "gdn":
            for gdn_block in self.gdn_blocks:
                sequence, _ = gdn_block(sequence, None, None)

        # 最後のトークンの表現
        final_repr = sequence[:, -1, :]  # (B, d_model)

        return final_repr, rnn_state, ""
