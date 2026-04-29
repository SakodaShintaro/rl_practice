# SPDX-License-Identifier: MIT
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from vla_streaming_rl.networks.vlm_backbone import is_qwen35, load_model


class ZeroShotVLMAgent:
    """Zero-shot VLM controller. Each step the current observation is sent to a chat-tuned
    VLM as a user turn; the model replies with an action string that is parsed via the
    env-supplied regex. The most recent ``seq_len`` (image, assistant_response) turns from
    the current episode are kept in a FIFO and prepended to the chat as in-context history.
    """

    def __init__(
        self,
        *,
        model_id: str,
        parse_action_text: Callable[[str], tuple[np.ndarray, bool]],
        action_dim: int,
        format_hint: str,
        seq_len: int,
        max_new_tokens: int,
    ) -> None:
        self.device = "cuda"
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()
        self.is_qwen35 = is_qwen35(model_id)

        self.parse_action_text = parse_action_text
        self.action_dim = action_dim
        self.format_hint = format_hint
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens

        self.task_prompt = ""
        self.history: deque[tuple[Image.Image, str]] = deque(maxlen=seq_len)
        self.last_action = np.zeros(action_dim, dtype=np.float32)

    def reset(self, task_prompt: str) -> None:
        self.task_prompt = task_prompt
        self.history.clear()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, prev_reward: float) -> tuple[np.ndarray, dict]:
        del prev_reward
        current_image = self._obs_to_pil(obs)
        messages = self._build_messages(current_image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.is_qwen35:
            images, _ = process_vision_info(messages)
        else:
            images, _ = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(self.device)
            for k, v in inputs.items()
        }

        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        response_text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        action_array, parse_success = self.parse_action_text(response_text)
        if parse_success and len(action_array) > 0:
            action = action_array[0].astype(np.float32)
        else:
            action = self.last_action

        if self.seq_len > 0:
            self.history.append((current_image, response_text))
        self.last_action = action

        return action, {"text": response_text, "parse_success": parse_success}

    def _build_messages(self, current_image: Image.Image) -> list[dict]:
        system_prompt = "\n\n".join(p for p in (self.task_prompt, self.format_hint) if p)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for past_image, past_response in self.history:
            messages.append(
                {"role": "user", "content": [{"type": "image", "image": past_image}]}
            )
            messages.append({"role": "assistant", "content": past_response})
        messages.append(
            {"role": "user", "content": [{"type": "image", "image": current_image}]}
        )
        return messages

    @staticmethod
    def _obs_to_pil(obs: np.ndarray) -> Image.Image:
        img = (obs.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(img)
