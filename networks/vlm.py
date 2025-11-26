import re

import numpy as np
import torch
import torchvision.transforms as T
from mamba_ssm.utils.generation import InferenceParams
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel

ACTION_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
    "Action space: steering [-1, +1] where -1 is full left and +1 is full right; gas [0, 1]; braking [0, 1]. "
    "Pick exactly one High Level Action: 'Turn Left', 'Turn Right', 'Go Straight', or 'Slow Down'. Prefer small steering changes (|steering| <= 0.3) and modest gas; use brake only for sharp turns or when drifting off the road. If unsure, choose 'Slow Down'. "
    "Typical mappings: Turn Left -> steering=-0.20, gas=0.00, braking=0.00; Turn Right -> steering=0.20, gas=0.00, braking=0.00; Go Straight -> steering=0.00, gas=0.01, braking=0.00; Slow Down -> steering=0.00, gas=0.00, braking=0.10. "
    "Respond in the exact format: <think>Write your thinking</think>'High Level Action: <command>, Action: steering=X.XX, gas=X.XX, braking=X.XX' using decimal values within range."
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return zeros with the expected output shape"""
        batch_size = x.shape[0]
        device = x.device
        return torch.zeros(batch_size, *self.output_shape, device=device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy decode - not used for VLM encoders"""
        return x


class QwenVLEncoder(nn.Module):
    def __init__(self, output_text: bool) -> None:
        super().__init__()

        self.output_text = output_text

        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        # model_id = "Qwen/Qwen3-VL-2B-Thinking"

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with quantization
        # Use device_map with explicit device to avoid multi-GPU distribution issues
        device_map_config = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device_map_config,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            use_dora=True,
            init_lora_weights="gaussian",
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        self.processor = AutoProcessor.from_pretrained(model_id)
        out_dim = 4
        self.use_pixel_values = False
        if self.use_pixel_values:
            self.out_proj = nn.Linear(1536, out_dim)
            self.output_dim = out_dim * 256
        else:
            self.out_proj = nn.Linear(2048, out_dim)
            self.output_dim = out_dim * 74
        self.device = device
        self.out_proj = self.out_proj.to(device)
        self.video_fps = 50 / 8
        self.image_processor = DummyImageProcessor((self.output_dim,))
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def _build_messages(self, images: torch.Tensor, rewards: torch.Tensor) -> list[list[dict]]:
        batch_size = images.shape[0]
        seq_len = images.shape[1]
        messages = []

        for b in range(batch_size):
            frames = []
            for t in range(seq_len):
                img_tensor = images[b, t].to(torch.float32)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                frames.append(Image.fromarray(img_np))

            # content = [{"type": "video", "video": frames, "fps": self.video_fps}]
            content = [{"type": "image", "image": frame} for frame in frames]
            # content.append(
            #     {"type": "text", "text": f"The previous reward is {rewards[b, -1].item():.3f}."}
            # )
            # content.append({"type": "text", "text": ACTION_PROMPT})
            messages.append([{"role": "user", "content": content}])

        return messages

    def _prepare_inputs(self, messages: list[list[dict]]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in inputs.items()
        }
        return inputs

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        messages = self._build_messages(images, rewards)
        model_inputs = self._prepare_inputs(messages)

        if self.use_pixel_values:
            hidden = model_inputs["pixel_values"]
            B = images.size(0)
            token_num = hidden.size(0) // B
            hidden = hidden.view(B, token_num, -1)
        else:
            output = self.model.forward(**model_inputs, output_hidden_states=True)
            hidden = output["hidden_states"][-1]
        x = hidden.to(torch.float32)
        x = self.out_proj(x)
        x = x.flatten(start_dim=1)

        action_text = self._generate_action_text(messages[0]) if self.output_text else ""

        return x, rnn_state, action_text

    def _generate_action_text(self, conversation) -> str:
        model_inputs = self._prepare_inputs([conversation])

        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""


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
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        # images: (B, T, C, H, W)
        # obs_z, actions, rewards are ignored for VLM encoders
        # rnn_state is passed through unchanged (VLM encoders are stateless)

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

        return batch_output, rnn_state, action_text
