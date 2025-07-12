import torch
from diffusers.models import AutoencoderKL, AutoencoderTiny
from torch import nn
from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor


class BaseCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # input shape: (batch_size, in_channels, 96, 96)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=2),  # -> (8, 47, 47)
            nn.LayerNorm([8, 47, 47], elementwise_affine=False),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # -> (16, 23, 23)
            nn.LayerNorm([16, 23, 23], elementwise_affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # -> (32, 11, 11)
            nn.LayerNorm([32, 11, 11], elementwise_affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # -> (64, 5, 5)
            nn.LayerNorm([64, 5, 5], elementwise_affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # -> (128, 3, 3)
            nn.LayerNorm([128, 3, 3], elementwise_affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # -> (256, 1, 1)
            nn.LayerNorm([256, 1, 1], elementwise_affine=False),
            nn.ReLU(),
            nn.Flatten(),  # -> (256,)
        )

    def encode(self, x):
        return self.features(x)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")
        self.output_dim = 576

    @torch.no_grad()
    def encode(self, x):
        return self.ae.encode(x).latents.flatten(1)

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir="./cache")
        self.scale = 0.18215

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scale)

    def decode(self, x):
        return self.vae.decode(x / self.scale).sample


class SmolVLMEncoder(nn.Module):
    def __init__(self, device=None) -> None:
        super().__init__()
        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.prompt = "<image> Please drive in the lane."
        self.output_dim = 576

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        batch_size = images.shape[0]
        images_np = images.cpu().numpy()
        model_inputs = (
            self.processor(
                text=[self.prompt] * batch_size,
                images=[[img] for img in images_np],
                return_tensors="pt",
                do_rescale=False,
                padding=True,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = model_inputs["input_ids"].shape[-1]
        outputs = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = outputs["hidden_states"][-1]
        x = hidden[:, input_len - 1]
        x = x.to(torch.float32)
        return x


class SmolVLABackbone(nn.Module):
    def __init__(self, device=None) -> None:
        super().__init__()
        model_id = "HuggingFaceTB/SmolVLM-256M-Base"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.prompt = "<image> Please drive in the lane."
        self.output_dim = 576

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        batch_size = images.shape[0]
        images_list = [images[i] for i in range(batch_size)]
        model_inputs = (
            self.processor(
                text=[self.prompt] * batch_size,
                images=images_list,
                return_tensors="pt",
                do_rescale=False,
                padding=True,
            )
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = model_inputs["input_ids"].shape[-1]
        outputs = self.model.forward(
            **model_inputs,
            output_hidden_states=True,
        )
        hidden = outputs["hidden_states"][-1]
        x = hidden[:, input_len - 1]
        x = x.to(torch.float32)
        return x


if __name__ == "__main__":
    import torch

    def parameter_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = torch.device("cuda")
    x = torch.rand(1, 3, 96, 96, device=device)

    model_ae = AE().to(device)
    print(f"AE parameter count: {parameter_count(model_ae):,}")
    output_enc = model_ae.encode(x)
    print(output_enc.shape)  # (1, 4, 12, 12)
    output_dec = model_ae.decode(output_enc)
    print(output_dec.shape)  # (1, 3, 96, 96)

    model_vae = VAE().to(device)
    print(f"VAE parameter count: {parameter_count(model_vae):,}")
    output_enc = model_vae.encode(x)
    print(output_enc.shape)  # (1, 4, 12, 12)
    output_dec = model_vae.decode(output_enc)
    print(output_dec.shape)  # (1, 3, 96, 96)

    model_cnn = BaseCNN(in_channels=3).to(device)
    print(f"CNN parameter count: {parameter_count(model_cnn):,}")
    output = model_cnn(x)
    print(output.shape)  # (1, 256)

    model_smolvlm = SmolVLMEncoder(device=device)
    print(f"SmolVLMEncoder parameter count: {parameter_count(model_smolvlm):,}")
    output = model_smolvlm.encode(x)
    print(output.shape)  # (1, 256)

    model_smolvla = SmolVLABackbone(device=device)
    print(f"SmolVLABackbone parameter count: {parameter_count(model_smolvla):,}")
    output = model_smolvla.encode(x)
    print(output.shape)  # (1, 256)
