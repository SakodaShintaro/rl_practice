import re

import numpy as np
import torch
import torchvision.transforms as T
from mamba_ssm.utils.generation import InferenceParams
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel

# Unified action prompt for all VLM encoders
ACTION_PROMPT = (
    "You are an AI driving assistant. Analyze the driving scene from the video frames (from oldest to newest) and provide the next action. "
    "Action space: steering (-1 to +1, where -1 is full left, +1 is full right), "
    "gas (0 to 1), braking (0 to 1). "
    "Stay on the gray road and avoid green areas. "
    "Respond in format: 'Action: steering=X.X, gas=X.X, braking=X.X' where X.X are decimal values."
)


def parse_action_text(action_text: str) -> np.ndarray:
    """Parse action text and extract steering, gas, braking values.

    Args:
        action_text: Text in format 'Action: steering=X.X, gas=X.X, braking=X.X'

    Returns:
        np.ndarray of shape (3,) containing [steering, gas, braking]

    Example:
        >>> parse_action_text("Action: steering=0.5, gas=0.3, braking=0.0")
        array([0.5, 0.3, 0.0])
    """
    # Default values in case parsing fails
    steering, gas, braking = 0.0, 0.0, 0.0

    try:
        # Extract steering value
        steering_match = re.search(r"steering=([+-]?\d*\.?\d+)", action_text)
        if steering_match:
            steering = float(steering_match.group(1))

        # Extract gas value
        gas_match = re.search(r"gas=([+-]?\d*\.?\d+)", action_text)
        if gas_match:
            gas = float(gas_match.group(1))

        # Extract braking value
        braking_match = re.search(r"braking=([+-]?\d*\.?\d+)", action_text)
        if braking_match:
            braking = float(braking_match.group(1))

    except (ValueError, AttributeError):
        # Return default values if parsing fails
        pass

    # Create array and clamp values using numpy.clip
    action_array = np.array([steering, gas, braking], dtype=np.float32)
    action_array[0] = np.clip(action_array[0], -1.0, 1.0)  # steering
    action_array[1] = np.clip(action_array[1], 0.0, 1.0)  # gas
    action_array[2] = np.clip(action_array[2], 0.0, 1.0)  # braking

    return action_array


class DummyImageProcessor:
    """Dummy image processor for VLM encoders to maintain interface compatibility"""

    def __init__(self, output_shape):
        self.output_shape = output_shape


class VLMEncoderBase(nn.Module):
    """Base class for Vision-Language Model encoders"""

    def __init__(self, model_id: str, output_dim: int, device=None):
        super().__init__()

        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.output_dim = output_dim
        self.device = device
        # Dummy image processor for interface compatibility
        self.image_processor = DummyImageProcessor((output_dim,))

    def init_state(self) -> torch.Tensor:
        """Return dummy state for compatibility with standard encoders"""
        return torch.zeros(1, 1, 1)

    def get_stop_tokens(self):
        """Get stop tokens for text generation - to be overridden by subclasses"""
        return [
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, None, str]:
        # images: (B, T, C, H, W)
        # obs_z, actions, rewards, rnn_state are ignored for VLM encoders

        batch_size = images.shape[0]
        seq_len = images.shape[1]

        # Convert all batch samples to PIL images
        all_frames = []
        texts = []
        for b in range(batch_size):
            batch_frames = []
            for t in range(seq_len):
                img_tensor = images[b, t].to(torch.float32)  # (C, H, W)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                batch_frames.append(pil_img)
            all_frames.append(batch_frames)
            # Add image tokens for each frame in the sequence
            image_tokens = "<image>" * seq_len
            texts.append(image_tokens + ACTION_PROMPT)

        # Use processor directly to batch process all samples
        model_inputs = self.processor(
            images=all_frames,
            text=texts,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 384},
        )
        model_inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in model_inputs.items()
        }

        # Get hidden state from forward pass (batch)
        output = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][-1]  # (B, seq_len, hidden_dim)
        # Use the last token's hidden state for each batch
        x = hidden[:, -1, :].to(torch.float32)  # (B, hidden_dim)

        # Generate action text for the first batch only
        action_text = self._generate_action_text_batch(output.logits, model_inputs)

        return x, None, action_text

    def _generate_action_text_batch(self, logits, model_inputs) -> str:
        """Generate action text from logits for batched input (returns first batch's text)"""
        generated_ids = []
        stop_token_ids = self.get_stop_tokens()
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # Process only the first batch
        next_token_logits = logits[0:1, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([model_inputs["input_ids"][0:1], next_token], dim=-1)

        for _ in range(50):
            if next_token.item() in stop_token_ids:
                break
            generated_ids.append(next_token.item())

            if "attention_mask" in model_inputs:
                attention_mask = model_inputs["attention_mask"][0:1]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)],
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

    def _generate_action_text(self, logits, model_inputs) -> str:
        """Generate action text from logits without using model.generate()"""
        generated_ids = []
        stop_token_ids = self.get_stop_tokens()
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

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


class SmolVLMEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        output_dim = 576
        super().__init__(model_id, output_dim, device)


class QwenVLEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
        output_dim = 1536
        super().__init__(model_id, output_dim, device)


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
            dtype=torch.bfloat16,
        ).eval()
        self.model = self.model.to(device)

        # type(self.model.language_model)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.modeling_mmMamba.mmMambaForCausalLM'>
        # print(f"{type(self.model.language_model)=}")  # AutoModel

        # print(f"{self.model.config.embedding_config.img_context_token_id=}")  # 92546=IMG_CONTEXT

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir="./cache", trust_remote_code=True, use_fast=False
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
        # Dummy image processor for interface compatibility
        self.image_processor = DummyImageProcessor((self.output_dim,))

    def init_state(self) -> torch.Tensor:
        """Return dummy state for compatibility with standard encoders"""
        return torch.zeros(1, 1, 1)

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, None, str]:
        # images: (B, T, C, H, W)
        # obs_z, actions, rewards, rnn_state are ignored for VLM encoders

        batch_size = images.shape[0]
        device = images.device

        # MMMamba processes the last frame from each batch sample
        # Stack all last frames into a batch: (B, C, H, W)
        last_frames = images[:, -1, :, :, :]  # (B, C, H, W)

        # Transform all frames at once
        batch_images = torch.stack([self.transform(last_frames[b]) for b in range(batch_size)])
        batch_images = batch_images.to(device).to(torch.bfloat16)  # (B, C, H, W)

        # Create tokenized input (same for all batch samples)
        prompt = (
            ACTION_PROMPT + " <|im_start|>" + "<IMG_CONTEXT>" * self.image_token_num + "<|im_end|>"
        )
        messages = [{"role": "user", "content": prompt}]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Repeat input_ids for batch
        input_ids = model_inputs["input_ids"].repeat(batch_size, 1).to(device)

        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # Note: inference_params with batch > 1 may not be supported
        # Process first batch only for now due to inference_params limitation
        self.inference_params.seqlen_offset = 0

        outputs = self.model.forward(
            input_ids=input_ids[0:1],
            pixel_values=batch_images[0:1],
            inference_params=self.inference_params,
            output_hidden_states=True,
        )

        # Get representation from first batch
        x_first = outputs["hidden_states"][-1][:, 0].to(torch.float32)

        # For other batches, process without inference_params
        batch_outputs = [x_first]
        for b in range(1, batch_size):
            outputs_b = self.model.forward(
                input_ids=input_ids[b : b + 1],
                pixel_values=batch_images[b : b + 1],
                inference_params=None,
                output_hidden_states=True,
            )
            x_b = outputs_b["hidden_states"][-1][:, 0].to(torch.float32)
            batch_outputs.append(x_b)

        batch_output = torch.cat(batch_outputs, dim=0)  # (B, output_dim)

        # Generate action text for first batch only
        output_ids = []
        logits = outputs["logits"]
        last_logit = logits[:, -1, :]
        token = torch.argmax(last_logit, dim=-1)

        self.inference_params.seqlen_offset += input_ids.shape[1]
        input_ids_gen = token.unsqueeze(1)

        for _ in range(50):
            if token.item() in stop_token_ids:
                break
            output_ids.append(token.item())

            outputs = self.model.forward(
                input_ids=input_ids_gen,
                inference_params=self.inference_params,
            )
            logits = outputs["logits"]
            last_logit = logits[:, -1, :]
            token = torch.argmax(last_logit, dim=-1)
            input_ids_gen = token.unsqueeze(1)

        action_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return batch_output, None, action_text
