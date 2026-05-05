# SPDX-License-Identifier: MIT
"""Action-independent goal-image predictor backed by the trained Self-Forcing
world model.

External interface:
- ``__init__(...)``: configure
- ``step(obs)`` -> goal frame for this step
- ``reset()``: drop rolling buffer state (e.g. on episode boundary)

Everything else is private.

When ``enabled=False``, the heavy world-model pipeline is not loaded and every
``step`` call returns a pre-allocated black image, so callers can use the
predictor unconditionally without branching on whether the world model is
configured.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from vla_streaming_rl.self_forcing.model.inference_model import CausalInferencePipeline
from vla_streaming_rl.self_forcing.utils.misc import (
    CachedTextEncoder,
    load_generator_state_dict,
    resolve_checkpoint_path,
)

_INFERENCE_KEY_ORDER = ("generator_ema", "generator", "model")
_WAN_H, _WAN_W = 480, 832  # Wan T2V-1.3B native resolution
_WAN_FPS = 10  # Bench2Drive sampling rate; one predicted latent covers 4 pixel frames


class WorldModelGoalPredictor:
    def __init__(
        self,
        *,
        enabled: bool,
        config_path: str,
        checkpoint_path: str | None,
        device: torch.device,
        num_context_blocks: int,
        seconds_ahead: float,
        predict_interval: int,
    ) -> None:
        self._enabled = bool(enabled)
        self._device = device
        self._predict_interval = int(predict_interval)

        config = OmegaConf.load(config_path)
        self._fpb = int(config.num_frame_per_block)
        self._fixed_caption = config.b2d_caption
        K_lat = int(num_context_blocks) * self._fpb
        # Causal Wan VAE: first latent <- 1 pixel; later latents <- 4 pixels each.
        self._ctx_pix_needed = 1 + (K_lat - 1) * 4
        self._target_h = _WAN_H
        self._target_w = _WAN_W

        # Map seconds_ahead onto a single frame index in the predicted block.
        block_pix = self._fpb * 4  # =12 for fpb=3
        self._goal_frame_idx = max(
            0, min(round(seconds_ahead * _WAN_FPS) - 1, block_pix - 1)
        )

        self._step_counter: int = 0
        self._buffer: list[torch.Tensor] = []
        self._latest = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)

        self._pipeline: CausalInferencePipeline | None = None
        if self._enabled:
            self._pipeline = self._build_pipeline(config, checkpoint_path)

    def reset(self) -> None:
        """Drop the rolling pixel buffer and step counter (e.g. on episode end)."""
        self._step_counter = 0
        self._buffer.clear()
        self._latest = np.zeros_like(self._latest)

    @torch.inference_mode()
    def step(self, obs: np.ndarray) -> np.ndarray:
        """Push one observation frame and return the current goal frame.

        Args:
            obs: (3, H, W) float in [0, 1] — the current env observation.

        Returns:
            (H, W, 3) uint8 RGB image at Wan native resolution. Black until the
            buffer first fills and the model produces a real prediction; or
            always black when ``enabled=False``.
        """
        if not self._enabled:
            return self._latest

        self._push(obs)
        if (
            len(self._buffer) >= self._ctx_pix_needed
            and self._step_counter % self._predict_interval == 0
        ):
            block = self._run_inference()  # (fpb*4, H, W, 3) uint8
            self._latest = block[self._goal_frame_idx]
        self._step_counter += 1
        return self._latest

    def _build_pipeline(
        self, config, checkpoint_path: str | None
    ) -> CausalInferencePipeline:
        from peft import LoraConfig, get_peft_model

        pipeline = CausalInferencePipeline(
            timestep_shift=config.timestep_shift,
            num_frame_per_block=self._fpb,
            context_noise=config.context_noise,
        )
        base_ckpt_path = resolve_checkpoint_path(config.generator_ckpt)
        pipeline.generator.load_state_dict(
            load_generator_state_dict(base_ckpt_path, prefer_keys=_INFERENCE_KEY_ORDER),
            strict=True,
        )

        lora_cfg = config.lora
        if not bool(lora_cfg.enabled):
            raise ValueError("config.lora.enabled must be true.")
        pipeline.generator.model.requires_grad_(False)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            bias="none",
        )
        pipeline.generator.model = get_peft_model(pipeline.generator.model, peft_cfg)

        if checkpoint_path:
            sd = load_generator_state_dict(
                resolve_checkpoint_path(checkpoint_path), prefer_keys=_INFERENCE_KEY_ORDER
            )
            missing, unexpected = pipeline.generator.load_state_dict(sd, strict=False)
            print(
                f"[WorldModelGoalPredictor] finetune load: "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

        pipeline = pipeline.to(dtype=torch.bfloat16)
        pipeline.generator.to(device=self._device)
        pipeline.vae.to(device=self._device)
        pipeline.text_encoder.to(device=self._device)
        cached_text = pipeline.text_encoder([self._fixed_caption])
        pipeline.text_encoder = CachedTextEncoder(cached_text, device=self._device)
        torch.cuda.empty_cache()
        return pipeline

    def _push(self, obs: np.ndarray) -> None:
        t = torch.from_numpy(obs).to(device=self._device, dtype=torch.float32)
        if t.shape[1:] != (self._target_h, self._target_w):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self._target_h, self._target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        t = (t * 2.0 - 1.0).to(dtype=torch.bfloat16)
        self._buffer.append(t)
        if len(self._buffer) > self._ctx_pix_needed:
            del self._buffer[: len(self._buffer) - self._ctx_pix_needed]

    @torch.inference_mode()
    def _run_inference(self) -> np.ndarray:
        ctx_pix = torch.stack(self._buffer[-self._ctx_pix_needed :], dim=1).unsqueeze(0)
        ctx_latent = self._pipeline.vae.encode_to_latent(ctx_pix).to(dtype=torch.bfloat16)
        self._pipeline.vae.model.clear_cache()
        noise = torch.randn(
            (1, self._fpb, 16, ctx_latent.shape[-2], ctx_latent.shape[-1]),
            device=self._device,
            dtype=torch.bfloat16,
        )
        _, all_lat = self._pipeline.inference(
            noise=noise,
            text_prompts=[self._fixed_caption],
            initial_latent=ctx_latent,
        )
        # Decode (ctx + pred) together so the predicted frames see the same VAE
        # cache prefix used at inference time.
        decode_seq = torch.cat([ctx_latent, all_lat[:, -self._fpb :]], dim=1)
        video = self._pipeline.vae.decode_to_pixel(decode_seq)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self._pipeline.vae.model.clear_cache()
        out = (video[0].permute(0, 2, 3, 1).cpu().float() * 255).clamp(0, 255).to(torch.uint8)
        return out.numpy()[-self._fpb * 4 :]
