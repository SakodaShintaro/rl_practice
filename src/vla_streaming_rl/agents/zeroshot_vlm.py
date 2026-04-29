# SPDX-License-Identifier: MIT
import re
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from vla_streaming_rl.networks.vlm_backbone import is_qwen35, load_model

THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>"

_NUMBER_RE = re.compile(r"[+-]?\d*\.\d+|[+-]?\d+")


class _StopOnTokenSequence(StoppingCriteria):
    """Stop when the most recently generated tokens match a fixed target sequence —
    used to halt generation as soon as the model emits ``</think>``."""

    def __init__(self, target_ids: list[int]) -> None:
        self.target_ids = torch.as_tensor(target_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = self.target_ids.shape[0]
        if input_ids.shape[1] < n:
            return False
        tail = input_ids[:, -n:]
        target = self.target_ids.to(tail.device)
        return bool((tail == target).all(dim=1).all().item())


class _BinTokenOnly(LogitsProcessor):
    """Restrict logits to the contiguous range of token ids reserved for bin indices.
    With this in effect each generated token IS a bin selection — no separators, no
    EOS, no chatter."""

    def __init__(self, bin_token_start: int, n_bins: int) -> None:
        self.bin_token_start = bin_token_start
        self.n_bins = n_bins

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.bin_token_start : self.bin_token_start + self.n_bins] = 0.0
        return scores + mask


def _extract_post_think(text: str) -> str:
    """Return the segment after the last ``</think>`` so the action parser is not
    poisoned by stray numbers inside the reasoning."""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1]
    return text


def _fallback_parse(text: str, action_dim: int) -> tuple[np.ndarray, bool]:
    """Pick the LAST ``action_dim`` numeric tokens from ``text``. Used by the
    continuous-action path when the env's label-strict regex fails."""
    matches = _NUMBER_RE.findall(text)
    if len(matches) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    values = np.array([float(m) for m in matches[-action_dim:]], dtype=np.float32)
    return np.clip(values, -1.0, 1.0).reshape(1, action_dim), True


class ZeroShotVLMAgent:
    """Zero-shot VLM controller. Each step the current observation is sent to a
    chat-tuned VLM as a user turn; the model replies with an action that is decoded
    either from emitted bin tokens (``action_format='bin_index'``) or from parsed
    text (``action_format='continuous'``). The most recent ``seq_len`` turns from
    the current episode (image, assistant response, reward observed AFTER that
    response) are kept in a FIFO and prepended to the chat as in-context history."""

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
        use_thinking: bool,
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
        self.use_thinking = use_thinking
        self._eos_token_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id
        self._pad_token_id = pad_id if pad_id is not None else self._eos_token_id
        self._think_close_ids = self.processor.tokenizer.encode(
            THINK_CLOSE, add_special_tokens=False
        )

        # Reserve ``action_dim`` tokens for the action emission stage (one token per
        # bin in bin_index mode); the remainder of max_new_tokens goes to thinking.
        self._action_reserve = action_dim if action_format == "bin_index" else action_dim * 4 + 12
        self._think_budget = max(
            1, max_new_tokens - self._action_reserve - len(self._think_close_ids)
        )

        bins = np.linspace(-1.0, 1.0, n_bins + 1)
        self.bin_centers = ((bins[:-1] + bins[1:]) / 2.0).astype(np.float32)

        if action_format == "bin_index":
            # Reserve the last ``n_bins`` token ids of the model's BASE vocabulary as
            # dedicated bin tokens (one token = one bin value). In zero-shot these
            # tokens carry no learned semantics; we rely on in-context learning over
            # the FIFO history to bind them to actions.
            #
            # Use ``tokenizer.vocab_size`` rather than the embedding row count or
            # ``len(tokenizer)`` because:
            #   - The embedding matrix is often padded for hardware alignment, and
            #     the trailing rows correspond to no real token (decode → "").
            #   - ``len(tokenizer)`` includes added special tokens (im_start, im_end,
            #     eos, ...) that we definitely don't want to treat as bins.
            # ``tokenizer.vocab_size`` is the count of the base BPE vocabulary
            # before any special tokens were appended.
            self._bin_token_start = self.processor.tokenizer.vocab_size - n_bins
            tok = self.processor.tokenizer
            for i in range(n_bins):
                tid = self._bin_token_start + i
                text = tok.decode([tid], skip_special_tokens=False)
                re_ids = tok.encode(text, add_special_tokens=False)
                if re_ids != [tid]:
                    raise RuntimeError(
                        f"Bin token {tid} (bin {i}) does not round-trip cleanly: "
                        f"decoded={text!r}, re-encoded={re_ids}. Pick a different "
                        f"vocabulary slice for the bin tokens."
                    )
        else:
            self._bin_token_start = -1

        self.task_prompt = ""
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
        if self.use_thinking:
            text = text + THINK_OPEN
        if self.is_qwen35:
            images, _ = process_vision_info(messages)
        else:
            images, _ = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(self.device)
            for k, v in inputs.items()
        }

        prompt_len = inputs["input_ids"].shape[1]

        if self.use_thinking:
            stage1_out = self._run_thinking(inputs, prompt_len)
            thinking_text = self.processor.tokenizer.decode(
                stage1_out[0, prompt_len:], skip_special_tokens=False
            )
            action_input_ids = stage1_out
        else:
            thinking_text = ""
            action_input_ids = inputs["input_ids"]

        if self.action_format == "bin_index":
            bin_token_ids, action_text = self._emit_bin_action(inputs, action_input_ids)
            action = self._decode_bin_action(bin_token_ids)
            parse_used = "bin"
            bin_indices = [tid - self._bin_token_start for tid in bin_token_ids]
            display_action = "[" + ", ".join(str(b) for b in bin_indices) + "]"
        else:
            action_text = self._emit_text_action(inputs, action_input_ids)
            action, parse_used = self._parse_continuous_action(action_text)
            display_action = action_text

        # ``response_text`` is fed back into the chat history and MUST be the actual
        # decoded text of the model's output (so re-tokenizing it on the next turn
        # restores the same token ids — preserving the in-context property).
        # ``display_text`` is for the human-readable log only and substitutes the
        # bin-token characters with the bin indices they decode to.
        response_text = (thinking_text + action_text).strip()
        display_inner = (thinking_text + display_action).strip()
        full_response = (THINK_OPEN + response_text) if self.use_thinking else response_text
        display_text = (THINK_OPEN + display_inner) if self.use_thinking else display_inner

        if self.seq_len > 0:
            self.history.append((current_image, full_response, None))
        self.last_action = action
        self._action_count += 1

        return action, {"text": display_text, "parse_used": parse_used}

    # ------------------------------------------------------------------
    # Generation stages
    # ------------------------------------------------------------------

    def _run_thinking(self, inputs: dict, prompt_len: int) -> torch.LongTensor:
        """Stage 1: emit reasoning until ``</think>`` is observed (or budget is hit,
        in which case the closing tag is force-injected). Returns the full
        ``input_ids`` tensor extended with the thinking + closure."""
        stop = StoppingCriteriaList([_StopOnTokenSequence(self._think_close_ids)])
        stage1_out = self.model.generate(
            **inputs,
            max_new_tokens=self._think_budget,
            do_sample=False,
            num_beams=1,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            stopping_criteria=stop,
        )
        stage1_text = self.processor.tokenizer.decode(
            stage1_out[0, prompt_len:], skip_special_tokens=False
        )
        if THINK_CLOSE not in stage1_text:
            closure_ids = torch.tensor(
                [self._think_close_ids], device=self.device, dtype=stage1_out.dtype
            )
            stage1_out = torch.cat([stage1_out, closure_ids], dim=-1)
        return stage1_out

    def _emit_bin_action(
        self, inputs: dict, action_input_ids: torch.LongTensor
    ) -> tuple[list[int], str]:
        """Stage 2 for bin_index mode: emit exactly ``action_dim`` tokens drawn from
        the dedicated bin-token range. Returns ``(bin_token_ids, action_text)`` where
        ``action_text`` is the textual representation of those tokens (for FIFO
        history)."""
        proc = LogitsProcessorList([_BinTokenOnly(self._bin_token_start, self.n_bins)])
        stage2_inputs = {**inputs}
        stage2_inputs["input_ids"] = action_input_ids
        stage2_inputs["attention_mask"] = torch.ones_like(action_input_ids)
        out = self.model.generate(
            **stage2_inputs,
            max_new_tokens=self.action_dim,
            min_new_tokens=self.action_dim,
            do_sample=False,
            num_beams=1,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            logits_processor=proc,
        )
        bin_token_ids = out[0, action_input_ids.shape[1] :].tolist()
        action_text = self.processor.tokenizer.decode(bin_token_ids, skip_special_tokens=False)
        return bin_token_ids, action_text

    def _emit_text_action(
        self, inputs: dict, action_input_ids: torch.LongTensor
    ) -> str:
        """Stage 2 for continuous mode: free-form text generation that the env's
        ``parse_action_text`` callback then interprets."""
        stage2_inputs = {**inputs}
        stage2_inputs["input_ids"] = action_input_ids
        stage2_inputs["attention_mask"] = torch.ones_like(action_input_ids)
        out = self.model.generate(
            **stage2_inputs,
            max_new_tokens=self._action_reserve,
            do_sample=False,
            num_beams=1,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
        )
        new_tokens = out[0, action_input_ids.shape[1] :]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_bin_action(self, bin_token_ids: list[int]) -> np.ndarray:
        indices = np.array(
            [tid - self._bin_token_start for tid in bin_token_ids], dtype=np.int64
        )
        indices = np.clip(indices, 0, self.n_bins - 1)
        return self.bin_centers[indices].astype(np.float32)

    def _parse_continuous_action(self, action_text: str) -> tuple[np.ndarray, str]:
        action_array, parse_strict = self.parse_action_text(action_text)
        if parse_strict and len(action_array) > 0:
            return action_array[0].astype(np.float32), "strict"
        action_array, parse_loose = _fallback_parse(action_text, self.action_dim)
        if parse_loose:
            return action_array[0].astype(np.float32), "fallback"
        return self.last_action, "failed"

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

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
