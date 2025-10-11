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


class SingleFrameEncoder(nn.Module):
    def __init__(self, observation_space_shape: list[int]) -> None:
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")

        # self.ae.apply(init_weights)
        self.out_channels = 4
        self.hidden_h = observation_space_shape[1] // 8
        self.hidden_w = observation_space_shape[2] // 8

        self.output_dim = self.out_channels * self.hidden_h * self.hidden_w
        self.norm = nn.LayerNorm(self.output_dim, elementwise_affine=False)

    @torch.no_grad()
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
        x = images[:, -1]  # (B, C, H, W)
        x = self.ae.encode(x).latents.flatten(1)
        x = self.norm(x)
        return x

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), self.out_channels, self.hidden_h, self.hidden_w)
        return self.ae.decode(x).sample


class RecurrentEncoder(nn.Module):
    """
    CNN + GRU
    recurrent-ppo-truncated-bpttのActorCriticModelのエンコーダー部分を統合
    https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
    """

    def __init__(self, observation_space_shape: list[int]) -> None:
        super().__init__()
        input_channels = observation_space_shape[0]
        self.image_h = observation_space_shape[1]
        self.image_w = observation_space_shape[2]
        hidden_size = 256

        # CNN部分
        self.encoder_type = "simple_cnn"
        if self.encoder_type == "ae":
            self.ae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", cache_dir="./cache", device_map="cpu"
            )
        elif self.encoder_type == "simple_cnn":
            self.conv1 = nn.Conv2d(input_channels, 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        else:
            raise ValueError("Invalid encoder_type")

        # CNN出力サイズ
        conv_output_size = self._get_conv_output(observation_space_shape)

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
        if self.encoder_type == "ae":
            with torch.no_grad():
                h = self.ae.encode(all_frames).latents
        elif self.encoder_type == "simple_cnn":
            h = F.relu(self.conv1(all_frames))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
        else:
            raise ValueError("Invalid encoder_type")

        # Flatten and linear projection
        h = h.flatten(start_dim=1)  # (B*T, conv_output_size)
        h = F.relu(self.lin_hidden_in(h))  # (B*T, hidden_size)

        # RNN forward pass - 系列を処理
        h = h.reshape(B, T, self.output_dim)
        h, rnn_state = self.recurrent_layer(h, rnn_state)  # (B, T, hidden_size)
        h = h[:, -1]  # (B, hidden_size) - 最後の時刻の出力を使用

        return h, rnn_state

    def decode(self, x):
        # RecurrentEncoderは画像再構成機能がないため、ダミーを返す
        # 互換性のためのメソッド
        return torch.zeros(x.size(0), 3, self.image_h, self.image_w, device=x.device)

    def _get_conv_output(self, observation_space_shape: list[int]) -> int:
        o = torch.zeros(1, *observation_space_shape)
        if self.encoder_type == "simple_cnn":
            o = self.conv1(o)
            o = self.conv2(o)
            o = self.conv3(o)
        elif self.encoder_type == "ae":
            o = self.ae.encode(o).latents
        return int(np.prod(o.size()))


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

        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")

        # AE encoder outputs [B, 4, H, W] -> treat as [B, H * W, 4] (H * W tokens, 4 channels each)
        self.vae_dim = 4
        self.hidden_h = observation_space_shape[1] // 8
        self.hidden_w = observation_space_shape[2] // 8
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
            hidden_dim=self.vae_dim,
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

        self.output_dim = self.vae_dim * token_num

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
            all_latents = self.ae.encode(all_frames).latents  # [B*T, C', H', W']
            # Reshape back to sequence: [B, T, C', H', W']
            all_latents = all_latents.reshape(B, T, 4, 12, 12)
            # Convert to tokens: [B, T, S(=H'*W'), C']
            image_embed = all_latents.reshape(B, T, 4, -1).transpose(2, 3)

        # [B, T, action_dim] -> [B, T, action_dim, C']
        action_embed = get_fourier_embeds_from_coordinates(self.vae_dim, actions)

        # [B, T, 1] -> [B, T, 1, C']
        reward_embed = get_fourier_embeds_from_coordinates(self.vae_dim, rewards)

        # [B, T, 1, C']
        register_token = torch.zeros(
            (B, T, 1, self.vae_dim), device=images.device, dtype=images.dtype
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

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), self.vae_dim, self.hidden_h, self.hidden_w)
        return self.ae.decode(x).sample


class SimpleTransformerEncoder(nn.Module):
    """Sequence encoder using simple Transformer"""

    def __init__(self, observation_space_shape: list[int], seq_len: int):
        super().__init__()

        max_seq_len = seq_len * 3  # (obs, action, reward) per timestep

        # d_model is determined by AE output
        self.out_channels = 4
        self.hidden_h = observation_space_shape[1] // 8
        self.hidden_w = observation_space_shape[2] // 8
        self.output_dim = self.out_channels * self.hidden_h * self.hidden_w
        n_heads = 8  # Fixed number of attention heads
        n_blocks = 2  # Number of CausalTimeBlocks

        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")

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
            states = self.ae.encode(all_images).latents  # (B*T, 4, 12, 12)
            states = states.view(B, T, -1)  # (B, T, 576) - flatten AE output

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

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample
