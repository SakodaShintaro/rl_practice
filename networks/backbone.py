import base64
import io
import re

import numpy as np
import openai
import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderTiny
from google import genai
from google.genai import types
from mamba_ssm.utils.generation import InferenceParams
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel

# Unified action prompt for all VLM encoders
GAS_LIMIT = 0.25
ACTION_PROMPT = (
    "You are an AI driving assistant. Analyze the driving scene from the images (from oldest to newest) and provide the next action. "
    "The input images show a top-down view of the racing environment. "
    "The red car is the ego vehicle. "
    "Keep the speed low. "
    "Action space: steering (-1 to +1, where -1 is full left, +1 is full right), "
    f"gas (0 to {GAS_LIMIT}), braking (0 to 1). "
    "Pay attention to the state of ego vehicle's motion compared to the previous image. "
    'Determine whether your current movement state is "stationary, slowing down, turning left, or turning right."'
    "The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track."
    "First, analyze the situation and plan your strategy inside <think></think> tags. "
    "After closing the </think> tag, you MUST provide the high-level action and detailed action in this exact format: "
    "High-level action: [Maintain/Accelerate/Decelerate/Turn left/Turn right] "
    "'Action: steering=X.X, gas=X.X, braking=X.X' where X.X are decimal values. "
    "Example format:\n"
    "<think>The road curves left, I need to steer left while maintaining speed.</think>\n"
    "High-level action: Turn left\n"
    "Action: steering=-0.3, gas=0.5, braking=0.0"
)


def parse_action_text(action_text: str) -> np.ndarray:
    """Parse action text and extract steering, gas, braking values.

    Args:
        action_text: Text that may contain <think></think> tags and 'Action: steering=X.X, gas=X.X, braking=X.X'

    Returns:
        np.ndarray of shape (3,) containing [steering, gas, braking]

    Example:
        >>> parse_action_text("<think>I need to turn left</think>Action: steering=0.5, gas=0.3, braking=0.0")
        array([0.5, 0.3, 0.0])
    """
    # Default values in case parsing fails
    steering, gas, braking = 0.0, 0.0, 0.0

    try:
        # Remove <think></think> content first (optional, but more explicit)
        clean_text = re.sub(r"<think>.*?</think>", "", action_text, flags=re.DOTALL)

        # Extract steering value
        steering_match = re.search(r"steering=([+-]?\d*\.?\d+)", clean_text)
        if steering_match:
            steering = float(steering_match.group(1))

        # Extract gas value
        gas_match = re.search(r"gas=([+-]?\d*\.?\d+)", clean_text)
        if gas_match:
            gas = float(gas_match.group(1))

        # Extract braking value
        braking_match = re.search(r"braking=([+-]?\d*\.?\d+)", clean_text)
        if braking_match:
            braking = float(braking_match.group(1))

    except (ValueError, AttributeError):
        # Return default values if parsing fails
        pass

    # Create array and clamp values using numpy.clip
    action_array = np.array([steering, gas, braking], dtype=np.float32)
    action_array[0] = np.clip(action_array[0], -1.0, 1.0)  # steering
    action_array[1] = np.clip(action_array[1], 0.0, GAS_LIMIT)  # gas
    action_array[2] = np.clip(action_array[2], 0.0, 1.0)  # braking

    return action_array


class AE(nn.Module):
    def __init__(self, device=None) -> None:
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="./cache", device_map=device
        )
        self.output_dim = 576

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, reward: float | None, prev_action: np.ndarray | None
    ) -> tuple[torch.Tensor, str]:
        return self.ae.encode(x).latents.flatten(1), ""

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample

    def reset_inference_params(self):
        pass


class VLMEncoderBase(nn.Module):
    """Base class for Vision-Language Model encoders"""

    def __init__(self, model_id: str, output_dim: int, device=None):
        super().__init__()

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
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.output_dim = output_dim
        self.device = device
        self.max_images = 100
        self.step_buffer = []  # List of (image, reward, action) tuples

    def get_stop_tokens(self):
        """Get stop tokens for text generation - to be overridden by subclasses"""
        return [
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, reward: float | None, prev_action: np.ndarray | None
    ) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        # Convert tensor to PIL Image
        img_tensor = images[0].to(torch.float32)  # (3, H, W) - ensure float32 for PIL conversion
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Add current step (image, reward, action) tuple to buffer
        self.step_buffer.append((pil_img, reward, prev_action))

        if len(self.step_buffer) > self.max_images:
            self.step_buffer.pop(0)

        # Create chat template messages with interleaved images and temporal context
        content = [{"type": "text", "text": ACTION_PROMPT}]

        # Add steps with their temporal context
        for image, step_reward, step_action in self.step_buffer:
            content.append({"type": "image", "image": image})

            # Build reward string
            if step_reward is None:
                reward_str = "start"
            else:
                reward_str = f"{step_reward:.2f}"

            # Build action string
            if step_action is None:
                action_str = "start"
            else:
                action_str = f"(steering={step_action[0]:.2f}, gas={step_action[1]:.2f}, braking={step_action[2]:.2f})"

            # Add temporal context for this step
            context_text = f"reward={reward_str}, action={action_str}"
            content.append({"type": "text", "text": context_text})

        messages = [{"role": "user", "content": content}]

        # Prepare model inputs
        model_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in model_inputs.items()
        }

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
        stop_token_ids = self.get_stop_tokens()
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # Get the next token from the initial logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)

        for _ in range(150):  # max_new_tokens
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

    def reset_inference_params(self):
        self.step_buffer = []


class SmolVLMEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        output_dim = 576
        super().__init__(model_id, output_dim, device)


class QwenVLEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
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
            torch_dtype=torch.bfloat16,
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
        self.step_buffer = []  # List of (reward, action) tuples for MMMamba (single image)
        self.max_images = 100

    def reset_inference_params(self):
        """Reset inference parameters to default values."""
        self.inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)
        self.step_buffer = []

    @torch.inference_mode()
    def forward(
        self, image: torch.Tensor, reward: float | None, prev_action: np.ndarray | None
    ) -> tuple[torch.Tensor, str]:
        device = image.device
        batch_size = image.shape[0]
        assert batch_size == 1, "Batch size must be 1 for stepwise inference"
        image = self.transform(image).to(device).to(torch.bfloat16)

        # Add previous time step's reward and action to buffer BEFORE making current decision
        self.step_buffer.append((reward, prev_action))

        if len(self.step_buffer) > self.max_images:
            self.step_buffer.pop(0)

        # Use basic action prompt
        action_prompt = ACTION_PROMPT

        messages = [
            {
                "role": "user",
                "content": action_prompt
                + " <|im_start|>"
                + "<IMG_CONTEXT>" * self.image_token_num
                + "<|im_end|>",
            }
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"].to(device)

        output_ids = []
        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=image,
            inference_params=self.inference_params,
            output_hidden_states=True,
        )  # CausalLMOutputWithPast (outputs.keys()=odict_keys(['logits', 'hidden_states']))

        # Get representation from hidden states
        x = outputs["hidden_states"][-1][:, 0]
        x = x.to(torch.float32)

        # Generate action text using multiple tokens
        logits = outputs["logits"]
        last_logit = logits[:, -1, :]  # shape: (batch_size, vocab_size)
        token = torch.argmax(last_logit, dim=-1)  # shape: (batch_size,)

        self.inference_params.seqlen_offset += input_ids.shape[1]
        input_ids = token.unsqueeze(1)  # Start with first generated token

        # Generate more tokens for action text
        for _ in range(150):  # max_new_tokens
            if token.item() in stop_token_ids:
                break
            output_ids.append(token.item())

            outputs = self.model.forward(
                input_ids=input_ids,
                inference_params=self.inference_params,
            )
            logits = outputs["logits"]
            last_logit = logits[:, -1, :]
            token = torch.argmax(last_logit, dim=-1)
            input_ids = token.unsqueeze(1)

        # Decode generated tokens to action text
        action_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return x, action_text


class GeminiEncoder(nn.Module):
    """Gemini API based encoder for action text generation"""

    def __init__(self, model_name="gemini-2.5-flash", device=None) -> None:
        super().__init__()

        self.model_name = model_name
        self.output_dim = 512  # Dummy dimension
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.step_buffer = []  # List of (image, reward, action) tuples
        self.max_images = 20  # Smaller buffer for API calls
        self.client = genai.Client()

    def _convert_image_to_bytes(self, image: torch.Tensor) -> tuple[bytes, str]:
        """Convert torch tensor image to bytes for Gemini API"""
        # Convert tensor to PIL Image
        img_tensor = image.to(torch.float32)  # (3, H, W)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Convert to bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return img_bytes, "image/png"

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, reward: float | None, prev_action: np.ndarray | None
    ) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        # Add current step to buffer
        img_bytes, mime_type = self._convert_image_to_bytes(images[0])
        self.step_buffer.append((img_bytes, mime_type, reward, prev_action))

        if len(self.step_buffer) > self.max_images:
            self.step_buffer.pop(0)

        # Create content for Gemini API
        content = [ACTION_PROMPT]

        # Add images with context
        for img_bytes, mime_type, step_reward, step_action in self.step_buffer:
            content.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

            # Build reward string
            if step_reward is None:
                reward_str = "start"
            else:
                reward_str = f"{step_reward:.2f}"

            # Build action string
            if step_action is None:
                action_str = "start"
            else:
                action_str = f"(steering={step_action[0]:.2f}, gas={step_action[1]:.2f}, braking={step_action[2]:.2f})"

            # Add temporal context for this step
            context_text = f"reward={reward_str}, action={action_str}"
            content.append(context_text)

        # Call Gemini API
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=content)
            action_text = response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            action_text = "Action: steering=0.0, gas=0.0, braking=0.0"

        # Create dummy representation (random vector)
        dummy_representation = torch.randn(
            1, self.output_dim, device=self.device, dtype=torch.float32
        )

        return dummy_representation, action_text

    def reset_inference_params(self):
        self.step_buffer = []


class OpenAIEncoder(nn.Module):
    """OpenAI API based encoder for action text generation"""

    def __init__(self, model_name="gpt-4o", device=None) -> None:
        super().__init__()

        self.model_name = model_name
        self.output_dim = 512  # Dummy dimension
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.step_buffer = []  # List of (image, reward, action) tuples
        self.max_images = 10  # Smaller buffer for API calls

    def _encode_image_to_base64(self, image: torch.Tensor) -> str:
        """Convert torch tensor image to base64 string"""
        # Convert tensor to PIL Image
        img_tensor = image.to(torch.float32)  # (3, H, W)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, reward: float | None, prev_action: np.ndarray | None
    ) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        # Add current step to buffer
        image_b64 = self._encode_image_to_base64(images[0])
        self.step_buffer.append((image_b64, reward, prev_action))

        if len(self.step_buffer) > self.max_images:
            self.step_buffer.pop(0)

        # Create content for OpenAI API
        content = [{"type": "text", "text": ACTION_PROMPT}]

        # Add images with context
        for img_b64, step_reward, step_action in self.step_buffer:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            )

            # Build reward string
            if step_reward is None:
                reward_str = "start"
            else:
                reward_str = f"{step_reward:.2f}"

            # Build action string
            if step_action is None:
                action_str = "start"
            else:
                action_str = f"(steering={step_action[0]:.2f}, gas={step_action[1]:.2f}, braking={step_action[2]:.2f})"

            # Add temporal context for this step
            context_text = f"reward={reward_str}, action={action_str}"
            content.append({"type": "text", "text": context_text})

        # Call OpenAI API
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0.7,
            )
            action_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            action_text = "Action: steering=0.0, gas=0.0, braking=0.0"

        # Create dummy representation (random vector)
        dummy_representation = torch.randn(
            1, self.output_dim, device=self.device, dtype=torch.float32
        )

        return dummy_representation, action_text

    def reset_inference_params(self):
        self.step_buffer = []
