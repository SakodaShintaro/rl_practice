# SPDX-License-Identifier: MIT
"""Build VLM model inputs without re-tokenizing every step.

The HF AutoProcessor.__call__ runs apply_chat_template + tokenizer +
placeholder-replacement + BatchFeature construction every call, allocating
thousands of small Python objects per step. In long Bench2Drive routes the
heap state churned by scenario runtime amplifies that allocation cost
(observed: 2 ms -> 17 ms over a single run on the same input shape).

In our setup task prompt and image dimensions are fixed, so the text-side
outputs are identical every call. Build them once at init and only run
image preprocessing per call. The original `prepare_vlm_inputs` also
roundtripped tensor -> CPU -> numpy -> PIL just so the HF processor could
do PIL -> tensor; we skip that and pass tensors straight to the
image_processor.
"""
from collections.abc import Sequence

import torch
from PIL import Image
from transformers import AutoProcessor


class VLMInputCache:
    """Caches text-side outputs and computes pixel_values on the fly.

    Output dict matches what the rest of the network consumes from
    `prepare_vlm_inputs`:
        - input_ids, attention_mask, image_grid_thw  (cached)
        - all_pixel_values, all_image_grid_thw       (per-call, only the
          first one actually depends on image content)
        - seq_len                                    (constant)

    `pixel_values` is intentionally omitted because nothing downstream
    reads it (the visual encoder uses all_pixel_values; the language
    forward uses inputs_embeds built externally from all_pixel_values).
    """

    def __init__(
        self,
        processor: AutoProcessor,
        observation_shape: Sequence[int],
        task_prompt: str,
        seq_len: int,
        device: torch.device,
    ) -> None:
        if len(observation_shape) != 3:
            raise ValueError(
                f"observation_shape must be (C, H, W); got {tuple(observation_shape)}"
            )
        self.image_processor = processor.image_processor
        self.observation_shape = tuple(observation_shape)
        self.seq_len = seq_len
        self.device = device

        _, H, W = self.observation_shape
        dummy_pil = Image.new("RGB", (W, H))

        content: list[dict] = []
        if task_prompt:
            content.append({"type": "text", "text": task_prompt})
        content.append({"type": "image", "image": dummy_pil})
        messages = [[{"role": "user", "content": content}]]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ref = processor(
            text=text, images=[dummy_pil], return_tensors="pt", padding=True
        )

        self.cached_input_ids: torch.Tensor = ref["input_ids"].to(device)
        self.cached_attention_mask: torch.Tensor = ref["attention_mask"].to(device)
        # image_grid_thw: (1, 3) — fixed for fixed input dimensions
        self.cached_image_grid_thw: torch.Tensor = ref["image_grid_thw"].to(device)

    def __call__(self, images: torch.Tensor) -> dict:
        """images: (B, T, C, H, W) float in [0, 1]. Returns input dict."""
        if images.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got shape {tuple(images.shape)}")
        B, T = images.shape[:2]
        if T != self.seq_len:
            raise ValueError(f"T mismatch: cache expects seq_len={self.seq_len}, got T={T}")

        device = images.device

        # Pass tensors straight to image_processor — it accepts torch input
        # without going through PIL. Input is in [0, 1] so we disable the
        # processor's internal /255 rescale (image_mean/std are in [0, 1]
        # scale and applied directly after).
        flat = images.reshape(B * T, *images.shape[2:])
        proc_out = self.image_processor(
            images=[flat[i] for i in range(B * T)],
            return_tensors="pt",
            do_rescale=False,
        )

        all_pixel_values = proc_out["pixel_values"].to(device).to(torch.bfloat16)
        # Each frame has the same shape so every row of image_grid_thw is
        # identical. Use the cached single-row value to avoid a fresh CPU
        # tensor allocation per step.
        all_image_grid_thw = self.cached_image_grid_thw.expand(B * T, -1)

        input_ids = self.cached_input_ids.expand(B, -1)
        attention_mask = self.cached_attention_mask.expand(B, -1)
        image_grid_thw = self.cached_image_grid_thw.expand(B, -1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "all_pixel_values": all_pixel_values,
            "all_image_grid_thw": all_image_grid_thw,
            "seq_len": T,
        }
