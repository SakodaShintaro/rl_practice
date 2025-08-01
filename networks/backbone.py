import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderKL, AutoencoderTiny
from mamba_ssm.utils.generation import InferenceParams
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel


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
        self.action_prompt = (
            "You are an AI driving assistant. Based on the video frames (from oldest to newest), determine the appropriate steering, gas, and braking values. "
            "Action space: steering (-1 to +1), gas (0 to 1), braking (0 to 1). "
            "Stay on the gray road and avoid green areas. "
            "Respond in format: 'Action: steering=X.X, gas=X.X, braking=X.X' where X.X are decimal values."
        )
        self.output_dim = 576
        self.device = device
        self.max_frames = 100
        self.frame_buffer = []

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        # Add current frame to buffer
        self.frame_buffer.append(images[0])  # (3, H, W)

        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)

        # Build prompt with image placeholders
        prompt_text = f"{self.action_prompt} " + "<image>" * len(self.frame_buffer)

        # Prepare video frames
        video_frames = torch.stack(self.frame_buffer, dim=0)  # (num_frames, 3, H, W)

        model_inputs = (
            self.processor(
                text=[prompt_text],
                images=[video_frames],  # Pass as a list containing one video tensor
                return_tensors="pt",
                do_rescale=False,
                padding=True,
            )
            .to(torch.bfloat16)
            .to(self.device)
        )

        # Get hidden state from forward pass
        input_len = model_inputs["input_ids"].shape[-1]
        outputs = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = outputs["hidden_states"][-1]
        representation = hidden[:, input_len - 1].to(torch.float32)

        # Generate action text
        action_text = self._generate_action_text(outputs.logits, model_inputs)

        return representation, action_text

    def _generate_action_text(self, logits, model_inputs):
        """Generate action text from logits without using model.generate()"""
        generated_ids = []
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        # Get the next token from the initial logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)

        for _ in range(50):  # max_new_tokens
            if next_token.item() in stop_token_ids:
                break
            generated_ids.append(next_token.item())

            # Manually update attention mask
            if "attention_mask" in model_inputs:
                attention_mask = model_inputs["attention_mask"]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device)],
                    dim=-1,
                )
                model_inputs["attention_mask"] = attention_mask

            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=model_inputs.get("attention_mask"),
                pixel_values=model_inputs.get("pixel_values"),
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        action_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return action_text

    def forward(self, x):
        return self.encode(x)

    def reset_inference_params(self):
        self.frame_buffer = []


class MMMambaEncoder(nn.Module):
    """
    https://huggingface.co/hustvl/mmMamba-linear/blob/main/modeling_mmMamba_chat.py
    """

    def __init__(self, device=None) -> None:
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "hustvl/mmMamba-linear"

        self.model = mmMambaChatModel.from_pretrained(
            model_id,
            cache_dir="./cache",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model = self.model.to(device)

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

        self.input_size = 96
        self.image_token_num = (self.input_size // 14 // 2) ** 2

        self.transform = T.Compose(
            [
                T.Resize(
                    (self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)
        self.output_dim = 2048

    def reset_inference_params(self):
        """Reset inference parameters to default values."""
        self.inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        batch_size = image.shape[0]
        assert batch_size == 1, "Batch size must be 1 for stepwise inference"
        image = self.transform(image).to(device).to(torch.bfloat16)
        model_inputs = self.tokenizer(
            text=[
                "Please describe the following image: <|im_start|>"
                + "<IMG_CONTEXT>" * self.image_token_num
                + "<|im_end|>"
            ]
            * batch_size,
            return_tensors="pt",
            padding=True,
        )
        input_ids = model_inputs["input_ids"].to(device)

        output_ids = []
        stop_token_ids = [
            2,
            1163,
            92543,
            92542,
        ]

        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=image,
            inference_params=self.inference_params,
            output_hidden_states=True,
        )  # CausalLMOutputWithPast (outputs.keys()=odict_keys(['logits', 'hidden_states']))
        logits = outputs["logits"]
        # print(f"{logits.shape=}")  # logits.shape=torch.Size([1, len, vocab_size=92553])
        last_logit = logits[:, -1, :]  # shape: (batch_size, vocab_size)
        token = torch.argmax(last_logit, dim=-1)  # shape: (batch_size,)
        output_ids.append(token)
        self.inference_params.seqlen_offset += input_ids.shape[1]
        input_ids = token.unsqueeze(1)  # Append the new token

        x = outputs["hidden_states"][-1][:, 0]
        x = x.to(torch.float32)
        return x


class QwenVLEncoder(nn.Module):
    def __init__(self, device=None) -> None:
        super().__init__()
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

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
        self.action_prompt = (
            "You are an AI driving assistant. Analyze the driving scene from the video frames (from oldest to newest) and provide the next action. "
            "Action space: steering (-1 to +1, where -1 is full left, +1 is full right), "
            "gas (0 to 1), braking (0 to 1). "
            "Stay on the gray road and avoid green areas. "
            "Respond in format: 'Action: steering=X.X, gas=X.X, braking=X.X' where X.X are decimal values."
        )
        self.output_dim = 1536
        self.device = device
        self.max_frames = 100
        self.frame_buffer = []

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        from PIL import Image

        # Convert tensor to PIL Image and add to buffer
        img_tensor = images[0]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)
        self.frame_buffer.append(pil_img)

        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)

        # Prepare content for chat template
        content = []
        for frame in self.frame_buffer:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": self.action_prompt})

        messages = [{"role": "user", "content": content}]

        # Prepare model inputs
        model_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Get hidden state from forward pass
        input_len = model_inputs["input_ids"].shape[-1]
        output = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][-1]
        x = hidden[:, input_len - 1].to(torch.float32)

        # Generate action text using logits from forward pass
        action_text = self._generate_action_text(output.logits, model_inputs)

        return x, action_text

    def _generate_action_text(self, logits, model_inputs) -> str:
        """Generate action text from logits without using model.generate()"""
        generated_ids = []
        # The stop tokens for Qwen2.5-VL-Instruct are from the conversation template
        stop_token_ids = [
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]

        # Get the next token from the initial logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)

        for _ in range(50):  # max_new_tokens
            if next_token.item() in stop_token_ids:
                break
            generated_ids.append(next_token.item())

            # Manually update attention mask and position ids if they exist
            if "attention_mask" in model_inputs:
                attention_mask = model_inputs["attention_mask"]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device)],
                    dim=-1,
                )
                model_inputs["attention_mask"] = attention_mask

            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=model_inputs.get("attention_mask"),
                output_hidden_states=False,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        action_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return action_text

    def forward(self, x):
        return self.encode(x)

    def reset_inference_params(self):
        self.frame_buffer = []
