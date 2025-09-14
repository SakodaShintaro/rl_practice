import torch
from diffusers.models import AutoencoderTiny
from torch import nn

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
    def __init__(self, seq_len: int, device: str) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.device = device

        self.ae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="./cache", device_map=device
        )

        # self.ae.apply(init_weights)

        self.output_dim = 576
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
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample


class STTEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(self, seq_len: int, device: str, n_layer: int, tempo_block_type: str, action_dim: int):
        super().__init__()

        self.seq_len = seq_len

        self.device = device

        self.ae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="./cache", device_map=device
        )

        # AE encoder outputs [B, 4, 12, 12] -> treat as [B, 144, 4] (144 patches of 4 dims each)
        self.vae_dim = 4
        self.image_tokens_num = 144  # 12 * 12 patches
        action_tokens_num = action_dim
        reward_tokens_num = 1
        register_tokens_num = 1

        self.stt = SpatialTemporalTransformer(
            n_layer=n_layer,
            space_len=(
                self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
            ),
            tempo_len=self.seq_len,
            hidden_dim=self.vae_dim,
            n_head=1,
            attn_drop_prob=0.0,
            res_drop_prob=0.0,
            tempo_block_type=tempo_block_type,
        ).to(self.device)

        self.output_dim = self.vae_dim * (
            self.image_tokens_num + action_tokens_num + reward_tokens_num + register_tokens_num
        )

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
        B, T, C, H, W = images.shape

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

        output = last_frame_emb.flatten(start_dim=1)  # [B, (S+action_dim+1)*C']

        return output

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample


class SimpleTransformerEncoder(nn.Module):
    """Sequence encoder using simple Transformer"""

    def __init__(self, seq_len: int, device: str):
        super().__init__()

        self.seq_len = seq_len
        self.device = device

        # d_model is determined by AE output: 4 * 12 * 12 = 576
        self.d_model = 4 * 12 * 12  # 576
        n_heads = 8  # Fixed number of attention heads
        n_blocks = 2  # Number of CausalTimeBlocks

        self.ae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="./cache", device_map=device
        )

        # Positional encoding (for max possible sequence length with rewards)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len * 3, self.d_model))

        # CausalTimeBlocks using existing implementation
        config = Config(
            hidden_dim=self.d_model,
            n_head=n_heads,
            attn_drop_prob=0.0,
            res_drop_prob=0.0,
        )

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_blocks)])

        # Create causal mask for sequence length (considering max possible length with rewards)
        max_seq_len = seq_len * 3  # Maximum length when rewards are included
        matrix = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        self.causal_mask = torch.where(matrix == 1, 0, self.causal_mask)

        # Final projection
        self.output_proj = nn.Linear(self.d_model, self.d_model)

        # Move all parameters to the specified device
        self.to(device)

        self.output_dim = self.d_model

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
            self.d_model, actions
        )  # (B, T, action_dim, d_model)

        # Sum over action dimensions to get (B, T, d_model)
        action_emb = action_emb.sum(dim=2)  # (B, T, d_model)

        # Add rewards
        reward_emb = get_fourier_embeds_from_coordinates(
            self.d_model, rewards
        )  # (B, T, 1, d_model)
        reward_emb = reward_emb.sum(dim=2)  # (B, T, d_model)

        # Interleave states, actions, and rewards: [s_0, a_0, r_0, s_1, a_1, r_1, ...]
        sequence = torch.stack([states, action_emb, reward_emb], dim=2)  # (B, T, 3, d_model)
        sequence = sequence.view(B, T * 3, self.d_model)  # (B, T*3, d_model)

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
