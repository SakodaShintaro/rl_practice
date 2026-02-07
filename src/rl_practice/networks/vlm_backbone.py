# SPDX-License-Identifier: MIT
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


def get_action_prompt(horizon: int) -> str:
    """Generate action prompt based on horizon."""
    base = (
        "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
        "Action space: steer [-1, +1] where -1 is full left and +1 is full right; accel [-1, +1] where positive is gas and negative is brake. "
        "Typical actions: Turn Left -> steer=-0.20, accel=0.00; Turn Right -> steer=0.20, accel=0.00; Go Straight -> steer=0.00, accel=0.10; Slow Down -> steer=0.00, accel=-0.10. "
    )
    example_actions = "; ".join([f"t{i}: steer=0.00, accel=0.10" for i in range(horizon)])
    return (
        base + f"Respond with {horizon} sequential actions in format: 'Actions: {example_actions}'"
    )


def load_model(
    model_id: str, use_quantization: bool, use_lora: bool, device: torch.device
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen-VL model and processor."""
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        cache_dir="./cache",
        device_map=device,
    )
    if use_lora:
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="./cache",
        device_map=device,
    )

    return model, processor


def prepare_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    rewards: torch.Tensor,
    task_prompt: str,
) -> dict[str, torch.Tensor]:
    """Build VLM messages and prepare model inputs.

    Args:
        processor: AutoProcessor instance
        images: (B, T, C, H, W) tensor
        rewards: (B, T, 1) tensor
        task_prompt: Task prompt string (can be empty)

    Returns:
        Dictionary of model inputs
    """
    device = images.device
    batch_size, seq_len = images.shape[:2]
    messages = []
    for b in range(batch_size):
        content: list[dict] = []
        if task_prompt:
            content.append({"type": "text", "text": task_prompt})
        for t in range(seq_len):
            img_tensor = images[b, t].to(torch.float32)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            content.append({"type": "image", "image": Image.fromarray(img_np)})
            reward_text = f"reward {float(rewards[b, t, 0]):.2f}"
            content.append({"type": "text", "text": reward_text})
        messages.append([{"role": "user", "content": content}])

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    proc_images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        videos = None
        video_metadata = None

    inputs = processor(
        text=text,
        images=proc_images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )
    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }
    return inputs


def parse_action_text(action_text: str, horizon: int) -> tuple[np.ndarray, bool]:
    """Parse action text and extract steer, accel values for multiple timesteps.

    Args:
        action_text: Text in format 'Actions: t0: steer=X.XX, accel=X.XX; t1: steer=X.XX, accel=X.XX; ...'
        horizon: Number of timesteps to parse

    Returns:
        tuple of (action_array, success)
        - action_array: np.ndarray of shape (horizon, 2) containing [steer, accel] for each timestep
        - success: True if all timesteps were successfully parsed
    """
    action_array = np.zeros((horizon, 2), dtype=np.float32)
    success = True

    # Find all steer=X and accel=X pairs with optional t prefix
    pattern = r"(?:t\d+:\s*)?steer=([+-]?\d*\.?\d+),\s*accel=([+-]?\d*\.?\d+)"
    matches = re.findall(pattern, action_text)

    parsed_count = min(len(matches), horizon)
    success = parsed_count == horizon

    for i in range(parsed_count):
        steer = float(matches[i][0])
        accel = float(matches[i][1])
        action_array[i, 0] = np.clip(steer, -1.0, 1.0)
        action_array[i, 1] = np.clip(accel, -1.0, 1.0)

    return action_array, success


class QwenVLEncoder(nn.Module):
    def __init__(
        self,
        use_quantization: bool,
        use_lora: bool,
        target_layer_idx: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.use_lora = use_lora
        self.target_layer_idx = target_layer_idx
        self.seq_len = seq_len

        device = torch.device("cuda")

        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        self.model, self.processor = load_model(model_id, use_quantization, use_lora, device)

        out_dim = 4
        self.out_proj = nn.Linear(2048, out_dim)
        self.device = device
        self.out_proj = self.out_proj.to(device)
        self.video_fps = 50 / 8
        self._dummy_state = torch.zeros(1, 1, 1)

        # Compute target sequence length and output_dim via dummy forward pass
        self._target_seq_len, self.output_dim = self._compute_output_dim()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_output_dim(self) -> tuple[int, int]:
        """Compute target sequence length and output dimension by running a dummy forward pass."""
        dummy_images = torch.zeros(1, self.seq_len, 3, 96, 96, device=self.device)
        dummy_rewards = torch.zeros(1, self.seq_len, 1, device=self.device)

        model_inputs = prepare_vlm_inputs(self.processor, dummy_images, dummy_rewards, "")

        output = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][self.target_layer_idx]

        target_seq_len = hidden.shape[1]
        output_dim = target_seq_len * self.out_proj.out_features
        return target_seq_len, output_dim

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        with torch.enable_grad() if self.use_lora else torch.no_grad():
            model_inputs = prepare_vlm_inputs(self.processor, images, rewards, "")

            output = self.model.forward(**model_inputs, output_hidden_states=True)
            hidden = output["hidden_states"][self.target_layer_idx]

        x = hidden.to(torch.float32)
        x = self.out_proj(x)  # (B, seq_len, out_dim)
        seq_len = x.shape[1]
        if seq_len > self._target_seq_len:
            x = x[:, seq_len - self._target_seq_len :, :]
        elif seq_len < self._target_seq_len:
            pad = torch.zeros(
                x.shape[0], self._target_seq_len - seq_len, x.shape[2], device=x.device
            )
            x = torch.cat([pad, x], dim=1)
        x = x.flatten(start_dim=1)

        return x, rnn_state
