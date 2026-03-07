# SPDX-License-Identifier: MIT
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


def _is_qwen35(model_id: str) -> bool:
    return "Qwen3.5" in model_id


def load_model(
    model_id: str, use_quantization: bool, use_lora: bool, device: torch.device
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen-VL or Qwen3.5 model and processor."""
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_impl = "sdpa" if _is_qwen35(model_id) else "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
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

    processor = AutoProcessor.from_pretrained(model_id, cache_dir="./cache", device_map=device)

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
