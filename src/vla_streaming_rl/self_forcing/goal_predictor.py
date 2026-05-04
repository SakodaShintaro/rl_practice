# SPDX-License-Identifier: MIT
"""Action-independent goal-image predictor backed by the trained Self-Forcing
world model. Owned by the streaming RL agent; called every env step.

When `enabled=False`, the heavy world-model pipeline is not loaded and every
call returns a pre-allocated black image, so callers can use the predictor
unconditionally without branching on whether the world model is configured.
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


class WorldModelGoalPredictor:
    def __init__(
        self,
        *,
        enabled: bool,
        config_path: str,
        checkpoint_path: str | None,
        device: torch.device,
        num_context_blocks: int,
        predict_interval: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.config = OmegaConf.load(config_path)
        self.fpb = int(self.config.num_frame_per_block)
        self.K = int(num_context_blocks)
        self.K_lat = self.K * self.fpb
        # Causal Wan VAE: first latent <- 1 pixel frame; subsequent latents <- 4 pixels each.
        self.ctx_pix_needed = 1 + (self.K_lat - 1) * 4
        self.target_h = _WAN_H
        self.target_w = _WAN_W
        self.predict_interval = int(predict_interval)
        self.fixed_caption = self.config.b2d_caption

        self._step_counter: int = 0
        self._buffer: list[torch.Tensor] = []
        self._latest = np.zeros(
            (self.fpb * 4, self.target_h, self.target_w, 3), dtype=np.uint8
        )

        self.pipeline: CausalInferencePipeline | None = None
        if not self.enabled:
            return

        pipeline = CausalInferencePipeline(
            timestep_shift=self.config.timestep_shift,
            num_frame_per_block=self.fpb,
            context_noise=self.config.context_noise,
        )
        base_ckpt_path = resolve_checkpoint_path(self.config.generator_ckpt)
        pipeline.generator.load_state_dict(
            load_generator_state_dict(base_ckpt_path, prefer_keys=_INFERENCE_KEY_ORDER),
            strict=True,
        )

        lora_cfg = self.config.lora
        if not bool(lora_cfg.enabled):
            raise ValueError("WorldModelGoalPredictor requires `lora.enabled: true`.")
        from peft import LoraConfig, get_peft_model

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
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
        pipeline.text_encoder.to(device=device)
        cached_text = pipeline.text_encoder([self.fixed_caption])
        pipeline.text_encoder = CachedTextEncoder(cached_text, device=device)
        torch.cuda.empty_cache()

        self.pipeline = pipeline

    def reset(self) -> None:
        self._step_counter = 0
        self._buffer.clear()
        self._latest = np.zeros_like(self._latest)

    def get_latest(self) -> np.ndarray:
        return self._latest

    @torch.inference_mode()
    def step(self, obs_chw_float01: np.ndarray | torch.Tensor) -> np.ndarray:
        """Push a new (3, H, W) frame in [0, 1] and, every `predict_interval`
        steps once the buffer is full, run a fresh inference. Returns the
        latest goal block (black image until the first prediction, or always
        black when disabled)."""
        if not self.enabled:
            return self._latest

        if isinstance(obs_chw_float01, np.ndarray):
            t = torch.from_numpy(obs_chw_float01)
        else:
            t = obs_chw_float01
        t = t.to(device=self.device, dtype=torch.float32)
        if t.shape[1:] != (self.target_h, self.target_w):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        t = (t * 2.0 - 1.0).to(dtype=torch.bfloat16)
        self._buffer.append(t)
        if len(self._buffer) > self.ctx_pix_needed:
            del self._buffer[: len(self._buffer) - self.ctx_pix_needed]

        if (
            len(self._buffer) >= self.ctx_pix_needed
            and self._step_counter % self.predict_interval == 0
        ):
            self._latest = self._run_inference()
        self._step_counter += 1
        return self._latest

    @torch.inference_mode()
    def _run_inference(self) -> np.ndarray:
        ctx_pix = torch.stack(self._buffer[-self.ctx_pix_needed :], dim=1).unsqueeze(0)
        ctx_latent = self.pipeline.vae.encode_to_latent(ctx_pix).to(dtype=torch.bfloat16)
        self.pipeline.vae.model.clear_cache()
        noise = torch.randn(
            (1, self.fpb, 16, ctx_latent.shape[-2], ctx_latent.shape[-1]),
            device=self.device,
            dtype=torch.bfloat16,
        )
        _, all_lat = self.pipeline.inference(
            noise=noise,
            text_prompts=[self.fixed_caption],
            initial_latent=ctx_latent,
        )
        # Decode (ctx + pred) together so the predicted frames see the same VAE
        # cache prefix used at inference time; mirrors infer_valid.py.
        decode_seq = torch.cat([ctx_latent, all_lat[:, -self.fpb :]], dim=1)
        video = self.pipeline.vae.decode_to_pixel(decode_seq)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.pipeline.vae.model.clear_cache()
        out = (video[0].permute(0, 2, 3, 1).cpu().float() * 255).clamp(0, 255).to(torch.uint8)
        return out.numpy()[-self.fpb * 4 :]
