import torch
from diffusers.models import AutoencoderTiny
from torch import nn

from .spatial_temporal_transformer import (
    SpatialTemporalTransformer,
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


class AE(nn.Module):
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

    def forward(
        self, images: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Args:
            images: Tensor of shape (B, T, 3, H, W)
            actions: Tensor of shape (B, T, action_dim)

        Returns:
            Tuple: (encoded features, str)
        """
        batch_size = images.shape[0]
        x = images[:, -1]  # (B, C, H, W)
        x = self.ae.encode(x).latents.flatten(1)
        x = self.norm(x)
        return x, [""] * batch_size

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample

    def reset_inference_params(self):
        pass


class STTEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(self, seq_len: int, device: str):
        super().__init__()

        self.seq_len = seq_len

        self.device = device

        self.ae_encoder = AE(self.seq_len, self.device)

        # AE encoder outputs [B, 4, 12, 12] -> treat as [B, 144, 4] (144 patches of 4 dims each)
        self.vae_dim = 4
        self.image_tokens_num = 144  # 12 * 12 patches

        self.stt = SpatialTemporalTransformer(
            n_layer=1,
            time_len=self.seq_len,
            hidden_dim=self.vae_dim,
            n_head=1,
            attn_drop_prob=0.0,
            res_drop_prob=0.0,
        ).to(self.device)

        self.output_dim = self.vae_dim * self.image_tokens_num

    def reset_inference_params(self):
        pass

    def forward(
        self, images: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Args:
            images: Tensor of shape (B, T, 3, H, W)
            actions: Tensor of shape (B, T, action_dim)

        Returns:
            Tuple: (encoded features, str)
        """
        # Encode all frames with AE but preserve spatial structure
        # images: (B, T, C, H, W) -> encode all frames
        B, T, C, H, W = images.shape

        # Reshape to process all frames: (B*T, C, H, W)
        all_frames = images.reshape(-1, *images.shape[2:])

        with torch.no_grad():
            # Encode all frames at once
            all_latents = self.ae_encoder.ae.encode(all_frames).latents  # [B*T, C', H', W']
            # Reshape back to sequence: [B, T, C', H', W']
            all_latents = all_latents.reshape(B, T, 4, 12, 12)
            # Convert to tokens: [B, T, S(=H'*W'), C']
            image_embed = all_latents.reshape(B, T, 4, -1).transpose(2, 3)

        # [B, T, action_dim] -> [B, T, action_dim, C']
        action_embed = get_fourier_embeds_from_coordinates(self.vae_dim, actions)

        # [B, T, S+action_dim, C']
        all_embed = torch.cat([image_embed, action_embed], dim=2)

        # Apply STT to all frames
        stt_output = self.stt(all_embed)  # [B, T, S+action_dim, C']

        # Use last timestep's image tokens for final representation
        last_frame_emb = stt_output[:, -1, :, :]  # [B, S+action_dim, C']

        last_frame_emb = last_frame_emb[:, : self.image_tokens_num]  # [B, S, C']

        output = last_frame_emb.flatten(start_dim=1)  # [B, S*C']

        return output, [""] * B
