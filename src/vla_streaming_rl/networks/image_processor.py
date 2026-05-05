# SPDX-License-Identifier: MIT
import torch
from diffusers.models import AutoencoderTiny
from torch import nn

from vla_streaming_rl.self_forcing.utils.wan_wrapper import WanVAEWrapper


class ImageProcessor(nn.Module):
    def __init__(self, observation_space_shape: tuple[int], processor_type: str) -> None:
        super().__init__()
        self.observation_space_shape = observation_space_shape
        self.processor_type = processor_type
        if processor_type == "ae":
            self.processor = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        elif processor_type == "wan_vae":
            self.processor = WanVAEWrapper().eval().requires_grad_(False)
        self.output_shape = self._get_conv_output_shape(observation_space_shape)

    def _get_conv_output_shape(self, shape: tuple[int]) -> tuple[int]:
        x = torch.zeros(1, *shape)
        if self.processor_type == "ae":
            x = self.processor.encode(x).latents
        elif self.processor_type == "wan_vae":
            x = self._wan_encode(x)
        return list(x.size())[1:]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.processor_type == "ae":
            x = self.processor.encode(x).latents
        elif self.processor_type == "wan_vae":
            x = self._wan_encode(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.processor_type == "ae":
            x = x.view(x.size(0), *self.output_shape)
            x = self.processor.decode(x).sample
        elif self.processor_type == "wan_vae":
            x = x.view(x.size(0), *self.output_shape)
            x = self._wan_decode(x)
        return x

    def _wan_encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1] -> Wan expects (B, 3, T=1, H, W) in [-1, 1]
        z = self.processor.encode_to_latent((x * 2.0 - 1.0).unsqueeze(2))
        self.processor.model.clear_cache()
        return z.squeeze(1)  # (B, 16, H/8, W/8)

    def _wan_decode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 16, H_lat, W_lat) -> (B, 3, H, W) in [0, 1]
        pix = self.processor.decode_to_pixel(x.unsqueeze(1)).squeeze(1)
        self.processor.model.clear_cache()
        return (pix * 0.5 + 0.5).clamp(0, 1)
