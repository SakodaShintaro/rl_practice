# SPDX-License-Identifier: MIT
import torch
from diffusers.models import AutoencoderTiny
from torch import nn


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
