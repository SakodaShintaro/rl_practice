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


def is_qwen35(model_id: str) -> bool:
    return "Qwen3.5" in model_id


def load_model(
    model_id: str, use_lora: bool, device: torch.device
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen-VL or Qwen3.5 model and processor."""

    # quantization has a negative effect on performance, so we disable it by default for now
    # True:4.30 steps/sec, False 5.40 steps/sec
    use_quantization = False

    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_impl = "sdpa" if is_qwen35(model_id) else "flash_attention_2"
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
    is_qwen35: bool,
) -> dict[str, torch.Tensor]:
    """Build VLM messages and prepare model inputs.

    Args:
        processor: AutoProcessor instance
        images: (B, T, C, H, W) tensor
        rewards: (B, T, 1) tensor
        task_prompt: Task prompt string (can be empty)
        is_qwen35: Whether the model is Qwen3.5

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

    if is_qwen35:
        proc_images, _ = process_vision_info(messages)
    else:
        proc_images, _ = process_vision_info(messages, image_patch_size=16)

    inputs = processor(
        text=text,
        images=proc_images,
        return_tensors="pt",
        padding=True,
    )

    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }
    return inputs
