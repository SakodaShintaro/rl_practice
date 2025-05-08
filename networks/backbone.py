from diffusers.models import AutoencoderKL, AutoencoderTiny
from torch import nn


class BaseCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # input shape: (batch_size, in_channels, 96, 96)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=2),  # -> (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # -> (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # -> (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # -> (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # -> (128, 3, 3)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # -> (256, 1, 1)
            nn.ReLU(),
            nn.Flatten(),  # -> (256,)
        )

    def forward(self, x):
        return self.features(x)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./pretrained")

    def encode(self, x):
        return self.ae.encode(x).latents

    def decode(self, x):
        return self.ae.decode(x).sample


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema", cache_dir="./pretrained"
        )
        self.scale = 0.18215

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scale)

    def decode(self, x):
        return self.vae.decode(x / self.scale).sample


if __name__ == "__main__":
    import torch

    model_ae = AE()
    x = torch.randn(1, 3, 96, 96)
    output_enc = model_ae.encode(x)
    print(output_enc.shape)  # (1, 4, 12, 12)
    output_dec = model_ae.decode(output_enc)
    print(output_dec.shape)  # (1, 3, 96, 96)

    model_vae = VAE()
    x = torch.randn(1, 3, 96, 96)
    output_enc = model_vae.encode(x)
    print(output_enc.shape)  # (1, 4, 12, 12)
    output_dec = model_vae.decode(output_enc)
    print(output_dec.shape)  # (1, 3, 96, 96)
