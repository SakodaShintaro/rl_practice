# SPDX-License-Identifier: MIT
import re
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from vla_streaming_rl.networks.vlm_backbone import is_qwen35, load_model

_NUMBER_RE = re.compile(r"[+-]?\d*\.\d+|[+-]?\d+")


def _fallback_parse(text: str, action_dim: int) -> tuple[np.ndarray, bool]:
    """Pick the first ``action_dim`` numeric tokens from ``text``. Used when the env's
    label-strict regex fails — small chat models routinely drop the field names and
    emit something like "0.00, 0.10" instead of "steer=0.00, accel=0.10"."""
    matches = _NUMBER_RE.findall(text)
    if len(matches) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    values = np.array([float(m) for m in matches[:action_dim]], dtype=np.float32)
    return np.clip(values, -1.0, 1.0).reshape(1, action_dim), True


def _parse_bin_indices(
    text: str, action_dim: int, n_bins: int, bin_centers: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Extract ``action_dim`` integer bin indices from ``text`` and decode them via
    ``bin_centers``. Indices outside [0, n_bins) are clipped."""
    matches = _NUMBER_RE.findall(text)
    if len(matches) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    indices = np.round(np.array([float(m) for m in matches[:action_dim]])).astype(np.int64)
    indices = np.clip(indices, 0, n_bins - 1)
    return bin_centers[indices].astype(np.float32), True


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
        parse_action_text: Callable[[str], tuple[np.ndarray, bool]] | None,
        action_dim: int,
        format_hint: str,
        seq_len: int,
        max_new_tokens: int,
        action_format: str,
        n_bins: int,
    ) -> None:
        if action_format not in {"continuous", "bin_index"}:
            raise ValueError(f"Unknown action_format: {action_format}")
        if action_format == "continuous" and parse_action_text is None:
            raise ValueError("parse_action_text is required when action_format='continuous'")

        self.device = "cuda"
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()
        self.is_qwen35 = is_qwen35(model_id)

        self.parse_action_text = parse_action_text
        self.action_dim = action_dim
        self.format_hint = format_hint
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens
        self.action_format = action_format
        self.n_bins = n_bins
        bins = np.linspace(-1.0, 1.0, n_bins + 1)
        self.bin_centers = ((bins[:-1] + bins[1:]) / 2.0).astype(np.float32)

        self.task_prompt = ""
        # Each entry stores (observation image, assistant response, reward observed
        # AFTER that response). The reward slot is None until the next select_action()
        # call back-fills it.
        self.history: deque[tuple[Image.Image, str, float | None]] = deque(maxlen=seq_len)
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self._action_count = 0

    def reset(self, task_prompt: str) -> None:
        self.task_prompt = task_prompt
        self.history.clear()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._action_count = 0

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, prev_reward: float) -> tuple[np.ndarray, dict]:
        has_prev_reward = self._action_count > 0
        # Back-fill the reward of the previous response now that we know what it earned.
        if has_prev_reward and self.history:
            past_image, past_response, _ = self.history[-1]
            self.history[-1] = (past_image, past_response, prev_reward)

        current_image = self._obs_to_pil(obs)
        messages = self._build_messages(
            current_image,
            current_reward=prev_reward if has_prev_reward else None,
        )

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

        if self.action_format == "bin_index":
            decoded, parse_ok = _parse_bin_indices(
                response_text, self.action_dim, self.n_bins, self.bin_centers
            )
            if parse_ok:
                action = decoded
                parse_used = "bin"
            else:
                action = self.last_action
                parse_used = "failed"
        else:
            action_array, parse_strict = self.parse_action_text(response_text)
            if parse_strict and len(action_array) > 0:
                action = action_array[0].astype(np.float32)
                parse_used = "strict"
            else:
                action_array, parse_loose = _fallback_parse(response_text, self.action_dim)
                if parse_loose:
                    action = action_array[0].astype(np.float32)
                    parse_used = "fallback"
                else:
                    action = self.last_action
                    parse_used = "failed"

        if self.seq_len > 0:
            self.history.append((current_image, response_text, None))
        self.last_action = action
        self._action_count += 1

        return action, {"text": response_text, "parse_used": parse_used}

    def _build_messages(
        self, current_image: Image.Image, current_reward: float | None
    ) -> list[dict]:
        system_prompt = "\n\n".join(p for p in (self.task_prompt, self.format_hint) if p)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # The reward shown before user turn i is the one stored on history[i-1] —
        # i.e. the reward observed AFTER history[i-1]'s action. The oldest entry
        # in the FIFO has no preceding entry we still remember, so its reward prefix
        # is dropped.
        for i in range(len(self.history)):
            past_image, past_response, _ = self.history[i]
            reward_prefix = self.history[i - 1][2] if i > 0 else None
            messages.append(
                {"role": "user", "content": self._build_user_content(past_image, reward_prefix)}
            )
            messages.append({"role": "assistant", "content": past_response})
        messages.append(
            {"role": "user", "content": self._build_user_content(current_image, current_reward)}
        )
        return messages

    @staticmethod
    def _build_user_content(image: Image.Image, reward: float | None) -> list[dict]:
        content: list[dict] = []
        if reward is not None:
            content.append({"type": "text", "text": f"Previous reward: {reward:+.3f}"})
        content.append({"type": "image", "image": image})
        return content

    @staticmethod
    def _obs_to_pil(obs: np.ndarray) -> Image.Image:
        img = (obs.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(img)
