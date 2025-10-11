import numpy as np
import torch
from diffusers.models import AutoencoderTiny
from torch import nn
from torch.nn import functional as F

from .spatial_temporal_transformer import (
    Config,
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
    def __init__(self, observation_space_shape: list[int], processor_type: str) -> None:
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

    def _get_conv_output_shape(self, shape: list[int]) -> list[int]:
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
            x = torch.zeros(self.observation_space_shape, device=x.device)
        return x


class RecurrentEncoder(nn.Module):
    """
    CNN + GRU
    recurrent-ppo-truncated-bpttのActorCriticModelのエンコーダー部分を統合
    https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
    """

    def __init__(self, observation_space_shape: list[int]) -> None:
        super().__init__()
        self.image_h = observation_space_shape[1]
        self.image_w = observation_space_shape[2]
        hidden_size = 256

        # CNN部分
        self.image_processor = ImageProcessor(observation_space_shape, processor_type="simple_cnn")
        conv_output_size = np.prod(self.image_processor.output_shape)

        # CNN出力をRNN入力に変換する線形層
        self.lin_hidden_in = nn.Linear(conv_output_size, hidden_size)

        # RNN層（GRU）
        self.recurrent_layer = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # 出力次元
        self.output_dim = hidden_size

    def init_state(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.output_dim)

    def forward(
        self,
        images: torch.Tensor,  # (B, T, 3, H, W)
        actions: torch.Tensor,  # (B, T, action_dim)
        rewards: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # (1, B, hidden_size)
    ) -> torch.Tensor:
        """
        Returns:
            encoded features: (B, output_dim)
            rnn_state: (1, B, hidden_size)
        """
        B, T = images.shape[:2]

        # 各フレームをCNNで処理
        all_frames = images.reshape(B * T, *images.shape[2:])  # (B*T, C, H, W)

        # CNN forward pass
        h = self.image_processor.encode(all_frames)

        # Flatten and linear projection
        h = h.flatten(start_dim=1)  # (B*T, conv_output_size)
        h = F.relu(self.lin_hidden_in(h))  # (B*T, hidden_size)

        # RNN forward pass - 系列を処理
        h = h.reshape(B, T, self.output_dim)
        h, rnn_state = self.recurrent_layer(h, rnn_state)  # (B, T, hidden_size)
        h = h[:, -1]  # (B, hidden_size) - 最後の時刻の出力を使用

        return h, rnn_state


class STTEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(
        self,
        observation_space_shape: list[int],
        seq_len: int,
        n_layer: int,
        tempo_block_type: str,
        action_dim: int,
    ):
        super().__init__()

        self.image_processor = ImageProcessor(observation_space_shape, processor_type="ae")

        # image_processor outputs [B, C, H, W] -> treat as [B, H * W, C] (H * W tokens, C channels each)
        self.hidden_image_dim = self.image_processor.output_shape[0]
        self.hidden_h = self.image_processor.output_shape[1]
        self.hidden_w = self.image_processor.output_shape[2]
        self.image_tokens_num = self.hidden_h * self.hidden_w
        action_tokens_num = action_dim
        reward_tokens_num = 1
        register_tokens_num = 1

        self.stt = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=(
                self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
            ),
            tempo_len=seq_len,
            hidden_dim=self.hidden_image_dim,
            n_head=1,
            attn_drop_prob=0.0,
            res_drop_prob=0.0,
            tempo_block_type=tempo_block_type,
        )

        self.use_image_only = True

        token_num = (
            self.image_tokens_num
            if self.use_image_only
            else (
                self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
            )
        )

        self.output_dim = self.hidden_image_dim * token_num

    def forward(
        self, images: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, T, 3, H, W)
            actions: Tensor of shape (B, T, action_dim)
            rewards: Tensor of shape (B, T, 1)

        Returns:
            encoded features: (B, output_dim)
        """
        # Encode all frames with AE but preserve spatial structure
        # images: (B, T, C, H, W) -> encode all frames
        B, T = images.shape[:2]

        # Reshape to process all frames: (B*T, C, H, W)
        all_frames = images.reshape(-1, *images.shape[2:])

        with torch.no_grad():
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
        stt_output = self.stt(all_embed)  # [B, T, S+action_dim+1, C']

        # Use last timestep's image tokens for final representation
        last_frame_emb = stt_output[:, -1, :, :]  # [B, S+action_dim+1, C']

        if self.use_image_only:
            last_frame_emb = last_frame_emb[:, : self.image_tokens_num, :]  # [B, S, C']

        output = last_frame_emb.flatten(start_dim=1)  # [B, token_num * C']

        return output


class SimpleTransformerEncoder(nn.Module):
    """Sequence encoder using simple Transformer"""

    def __init__(self, observation_space_shape: list[int], seq_len: int):
        super().__init__()

        max_seq_len = seq_len * 3  # (obs, action, reward) per timestep

        self.image_processor = ImageProcessor(observation_space_shape, processor_type="ae")
        self.out_channels = self.image_processor.output_shape[0]
        self.hidden_h = self.image_processor.output_shape[1]
        self.hidden_w = self.image_processor.output_shape[2]
        self.output_dim = self.out_channels * self.hidden_h * self.hidden_w
        n_heads = 8  # Fixed number of attention heads
        n_blocks = 2  # Number of CausalTimeBlocks

        # Positional encoding (for max possible sequence length with rewards)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, self.output_dim))

        # CausalTimeBlocks using existing implementation
        config = Config(
            hidden_dim=self.output_dim,
            n_head=n_heads,
            attn_drop_prob=0.0,
            res_drop_prob=0.0,
        )

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_blocks)])

        # Create causal mask for sequence length (considering max possible length with rewards)
        matrix = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        self.causal_mask = torch.where(matrix == 1, 0, self.causal_mask)

        # Final projection
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)

    def forward(
        self, images: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, T, 3, H, W)
            actions: Tensor of shape (B, T, action_dim)
            rewards: Tensor of shape (B, T, 1)

        Returns:
            encoded features: (B, output_dim)
        """
        # Encode images using AE to get states
        B, T = images.shape[:2]

        # Process all images through AE
        all_images = images.reshape(B * T, *images.shape[2:])  # (B*T, C, H, W)
        with torch.no_grad():
            states = self.image_processor.encode(all_images)  # (B*T, C_h, H_h, W_h)
            states = states.view(B, T, -1)  # (B, T, hidden_dim)

        # Use fourier embeddings for actions (same as STTEncoder)
        action_emb = get_fourier_embeds_from_coordinates(
            self.output_dim, actions
        )  # (B, T, action_dim, d_model)

        # Sum over action dimensions to get (B, T, d_model)
        action_emb = action_emb.sum(dim=2)  # (B, T, d_model)

        # Add rewards
        reward_emb = get_fourier_embeds_from_coordinates(
            self.output_dim, rewards
        )  # (B, T, 1, d_model)
        reward_emb = reward_emb.sum(dim=2)  # (B, T, d_model)

        # Interleave states, actions, and rewards: [s_0, a_0, r_0, s_1, a_1, r_1, ...]
        sequence = torch.stack([states, action_emb, reward_emb], dim=2)  # (B, T, 3, d_model)
        sequence = sequence.view(B, T * 3, self.output_dim)  # (B, T*3, d_model)

        # Add positional encoding (adjust for actual sequence length)
        actual_seq_len = sequence.size(1)
        sequence = sequence + self.pos_encoding[:actual_seq_len].unsqueeze(0)

        # Apply CausalTimeBlocks with adjusted mask
        current_mask = self.causal_mask[:actual_seq_len, :actual_seq_len].to(sequence.device)
        for block in self.blocks:
            sequence = block(sequence, current_mask)

        # Take the last token's representation
        final_repr = sequence[:, -1, :]  # (B, d_model)

        # Final projection
        output = self.output_proj(final_repr)  # (B, d_model)

        return output
