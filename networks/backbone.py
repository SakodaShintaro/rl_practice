import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderKL, AutoencoderTiny
from mamba_ssm.utils.generation import InferenceParams
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


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

    def forward(self, x):
        return self.encode(x)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")
        self.output_dim = 576

    @torch.no_grad()
    def encode(self, x):
        return self.ae.encode(x).latents.flatten(1)

    def forward(self, x):
        return self.encode(x)

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

    def forward(self, x):
        return self.encode(x)

    def decode(self, x):
        return self.vae.decode(x / self.scale).sample


class BaseSmolEncoder(nn.Module):
    def __init__(self, model_id: str, model_class, device=None) -> None:
        super().__init__()

        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model_class.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.prompt = "Drive the red car along the gray road. Do not leave the road or touch the green areas. <image>"
        self.output_dim = 576

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        batch_size = images.shape[0]
        model_inputs = (
            self.processor(
                text=[self.prompt] * batch_size,
                images=[[img] for img in images],
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

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def describe(self, images: torch.Tensor, custom_prompt: str) -> list[str]:
        """Generate descriptions for cumulative image sequences.

        1st call: describes image 1
        2nd call: describes images 1-2
        3rd call: describes images 1-3
        etc.

        Args:
            images: Tensor of images with shape (sequence_length, 3, height, width)
            custom_prompt: Optional custom prompt for description generation

        Returns:
            List of generated descriptions for each cumulative sequence
        """
        device = images.device
        sequence_length = images.shape[0]

        # Convert images to PIL format for processor
        from PIL import Image

        pil_images = []
        for img in images:
            # Convert tensor to PIL Image
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype("uint8")
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)

        generated_texts = []

        # Process cumulative sequences: 1 image, then 1-2 images, then 1-3 images, etc.
        for i in range(sequence_length):
            current_images = pil_images[: i + 1]

            # Create messages in the correct chat format
            content = []
            for img in current_images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": custom_prompt})

            messages = [{"role": "user", "content": content}]

            # Use apply_chat_template for proper formatting
            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move all inputs to the same device as the model and convert to bfloat16
            model_inputs = {
                k: v.to(device).to(torch.bfloat16) if v.dtype == torch.float32 else v.to(device)
                for k, v in model_inputs.items()
            }

            # Generate text
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

            # Decode the generated text
            input_length = model_inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

            generated_text = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts


class SmolVLMEncoder(BaseSmolEncoder):
    def __init__(self, device=None) -> None:
        super().__init__(
            model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            model_class=AutoModelForImageTextToText,
            device=device,
        )


class MMMambaEncoder:
    """
    https://huggingface.co/hustvl/mmMamba-linear/blob/main/modeling_mmMamba_chat.py
    """

    def __init__(self, device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "hustvl/mmMamba-linear"

        # fused_norm_gate.pyが見える必要があるためpathに追加
        fused_norm_gate_path = Path(__file__).parent / "for_mmmamba"
        sys.path.append(str(fused_norm_gate_path))

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir="./cache",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model = self.model.to(device)
        # type(self.model)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.modeling_mmMamba_chat.mmMambaChatModel'>
        # print(f"{type(self.model)=}")

        # type(self.model.language_model)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.modeling_mmMamba.mmMambaForCausalLM'>
        # print(f"{type(self.model.language_model)=}")  # AutoModel

        # print(f"{self.model.config.embedding_config.img_context_token_id=}")  # 92546=IMG_CONTEXT

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        # type(self.tokenizer)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.tokenization_internlm2.InternLM2Tokenizer'>
        # print(f"{type(self.tokenizer)=}")

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self.input_size = 448

        self.transform = T.Compose(
            [
                T.Resize(
                    (self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        batch_size = images.shape[0]
        inference_params = InferenceParams(max_seqlen=1024, max_batch_size=4)
        images = self.transform(images).to(device).to(torch.bfloat16)
        model_inputs = self.tokenizer(
            text=["Please describe" + "<IMG_CONTEXT>" * 256] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        input_ids = model_inputs["input_ids"].to(device)
        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=images,
            inference_params=inference_params,
            output_hidden_states=True,
        )
        x = outputs["hidden_states"][-1][:, 0]
        x = x.to(torch.float32)
        return x

    def forward(self, x):
        return self.encode(x)

    @torch.no_grad()
    def describe(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        batch_size = images.shape[0]
        inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)
        images = self.transform(images).to(device).to(torch.bfloat16)
        model_inputs = self.tokenizer(
            text=["Please describe" + "<IMG_CONTEXT>" * 256] * batch_size,
            return_tensors="pt",
            padding=True,
        )
        print(f"Before")
        for k, v in inference_params.key_value_memory_dict.items():
            print(f"{k}={v.shape} on {v.device}")
        input_ids = model_inputs["input_ids"].to(device)
        print(f"{input_ids.shape=}")

        for itr in range(5):
            outputs = self.model.forward(
                input_ids=input_ids,
                pixel_values=images,
                inference_params=inference_params,
                output_hidden_states=True,
            )  # CausalLMOutputWithPast (outputs.keys()=odict_keys(['logits', 'hidden_states']))
            print(f"After {itr}")
            print(outputs["logits"].sum())
            for k, v in inference_params.key_value_memory_dict.items():
                print(f"{k}={type(v)}")
                for tensor in v:
                    print(f"  {tensor.shape} on {tensor.device} {torch.sum(tensor)}")
                    tensor = tensor[:]
            # inference_params.seqlen_offset += input_ids.shape[1]
            inference_params.seqlen_offset += 1

        x = outputs["hidden_states"][-1][:, 0]
        x = x.to(torch.float32)
        return x


if __name__ == "__main__":
    import torch

    def parameter_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = torch.device("cuda")
    x = torch.rand(1, 3, 96, 96, device=device)

    # model_ae = AE().to(device)
    # print(f"AE parameter count: {parameter_count(model_ae):,}")
    # output_enc = model_ae.encode(x)
    # print(output_enc.shape)  # (1, 4, 12, 12)
    # output_dec = model_ae.decode(output_enc)
    # print(output_dec.shape)  # (1, 3, 96, 96)

    # model_vae = VAE().to(device)
    # print(f"VAE parameter count: {parameter_count(model_vae):,}")
    # output_enc = model_vae.encode(x)
    # print(output_enc.shape)  # (1, 4, 12, 12)
    # output_dec = model_vae.decode(output_enc)
    # print(output_dec.shape)  # (1, 3, 96, 96)

    # model_cnn = BaseCNN(in_channels=3).to(device)
    # print(f"CNN parameter count: {parameter_count(model_cnn):,}")
    # output = model_cnn(x)
    # print(output.shape)  # (1, 256)

    # model_smolvlm = SmolVLMEncoder(device=device)
    # print(f"SmolVLMEncoder parameter count: {parameter_count(model_smolvlm):,}")
    # output = model_smolvlm.encode(x)
    # print(output.shape)  # (1, 576)

    model_mmmamba = MMMambaEncoder()
    print(f"MMMambaEncoder parameter count: {parameter_count(model_mmmamba.model):,}")
    output = model_mmmamba.encode(x)
    print(output.shape)  # (1, 576)

    descriptions = model_mmmamba.describe(x)
