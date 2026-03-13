# SPDX-License-Identifier: MIT
import argparse
import math

import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .diffusion_utils import compute_actor_loss_with_dacer, euler_denoise
from .image_processor import ImageProcessor
from .prediction_head import StatePredictionHead
from .reward_processor import RewardProcessor
from .value_head import ActionValueHead, maybe_update_hl_gauss_range
from .video_encoder import VideoEncoder
from .vlm_backbone import is_qwen35, load_model, prepare_vlm_inputs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. q/k: (B, heads, seq, head_dim), cos/sin: (B, seq, rope_dim).

    Supports partial rotary embedding: if rope_dim < head_dim, RoPE is applied
    only to the first rope_dim dimensions and the rest pass through unchanged.
    """
    cos = cos.unsqueeze(1)  # (B, 1, seq, rope_dim)
    sin = sin.unsqueeze(1)
    rope_dim = cos.shape[-1]
    if rope_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dim: int, min_period: float, max_period: float
) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timestep. time: (B,), returns (B, dim)."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, device=time.device, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(time.dtype)


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with adaptive modulation (scale, shift, gate) from conditioning signal."""

    def __init__(self, dim: int, eps: float, cond_dim: int) -> None:
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        modulation = self.dense(cond)
        # cond is (B, cond_dim), expand for (B, seq, dim)
        modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed.float() * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class ActionExpertLayer(nn.Module):
    """Single transformer layer: combined self+cross attention to VLM, then MLP.

    Uses GQA matching VLM config. Cross-attention uses VLM's own k/v projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_kv_heads
        intermediate_size = hidden_size * 4

        # Pre-attention norm (adaptive)
        self.input_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        # Self-attention projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # QK-norm (like Qwen3)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        # Post-attention norm (adaptive)
        self.post_attn_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        # MLP (SwiGLU like Qwen3)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA. (B, num_kv_heads, S, head_dim) -> (B, num_heads, S, head_dim)"""
        return kv.repeat_interleave(self.num_kv_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, action_len, hidden_size)
        vlm_k: torch.Tensor,  # (B, num_kv_heads, vlm_len, head_dim) with RoPE
        vlm_v: torch.Tensor,  # (B, num_kv_heads, vlm_len, head_dim)
        cos: torch.Tensor,  # (B, action_len, head_dim)
        sin: torch.Tensor,  # (B, action_len, head_dim)
        adarms_cond: torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:
        residual = hidden_states
        normed, gate1 = self.input_layernorm(hidden_states, adarms_cond)

        B, action_len, _ = normed.shape

        # Expert Q/K/V
        q = self.q_proj(normed).view(B, action_len, self.num_heads, self.head_dim)
        k_self = self.k_proj(normed).view(B, action_len, self.num_kv_heads, self.head_dim)
        v_self = self.v_proj(normed).view(B, action_len, self.num_kv_heads, self.head_dim)

        # QK-norm
        q = self.q_norm(q)
        k_self = self.k_norm(k_self)

        q = q.transpose(1, 2)  # (B, num_heads, action_len, head_dim)
        k_self = k_self.transpose(1, 2)  # (B, num_kv_heads, action_len, head_dim)
        v_self = v_self.transpose(1, 2)  # (B, num_kv_heads, action_len, head_dim)

        # RoPE on Q and self K to match VLM K
        q, k_self = apply_rotary_pos_emb(q, k_self, cos, sin)

        # Concat VLM K/V with self K/V: [vlm_tokens, action_tokens]
        k = torch.cat([vlm_k, k_self], dim=2)  # (B, num_kv_heads, vlm_len+action_len, head_dim)
        v = torch.cat([vlm_v, v_self], dim=2)

        # GQA: expand KV heads
        k = self._repeat_kv(k)  # (B, num_heads, vlm_len+action_len, head_dim)
        v = self._repeat_kv(v)  # (B, num_heads, vlm_len+action_len, head_dim)

        # Scaled dot-product attention (no mask: action tokens attend to everything)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, action_len, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, action_len, -1)

        # Output projection + gated residual
        hidden_states = residual + gate1 * self.o_proj(attn_out)  # (B, action_len, hidden_size)

        # MLP
        residual2 = hidden_states
        normed2, gate2 = self.post_attn_layernorm(hidden_states, adarms_cond)
        hidden_states = residual2 + gate2 * self.down_proj(
            F.silu(self.gate_proj(normed2)) * self.up_proj(normed2)
        )

        return hidden_states


class ActionExpert(nn.Module):
    """Stack of ActionExpertLayers that cross-attend to VLM hidden states."""

    def __init__(
        self,
        num_layers: int,
        expert_hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rotary_emb: nn.Module,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(
            [
                ActionExpertLayer(
                    expert_hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(expert_hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_seq_len: int,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        B, action_len, _ = hidden_states.shape

        # RoPE positions for action tokens: right after VLM sequence
        action_pos = torch.arange(
            vlm_seq_len, vlm_seq_len + action_len, device=hidden_states.device
        )
        action_pos_ids = action_pos.unsqueeze(0).expand(B, -1)  # (B, action_len)
        cos, sin = self.rotary_emb(hidden_states, position_ids=action_pos_ids)

        for j in range(self.num_layers):
            k, v = vlm_kv_list[j]
            hidden_states = self.layers[j](hidden_states, k, v, cos, sin, adarms_cond)
        return self.norm(hidden_states)


class VLMActorCriticWithActionValue(nn.Module):
    """VLM backbone + Action Expert (flow matching) + Action Value critic.

    Architecture (Knowledge Insulation):
    - VLM (Qwen3-VL, frozen): processes images + text, provides intermediate representations
    - Action Expert: transformer that cross-attends to VLM at each layer, denoises actions
    - Critic: Q(state, action) with dueling architecture
    - Stop-gradient: VLM hidden states are detached before Expert uses them
    """

    def __init__(
        self,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.gamma = args.gamma
        self.num_bins = args.num_bins
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = args.critic_loss_weight
        self.denoising_steps = args.denoising_steps
        self.denoising_time = args.denoising_time
        self.dacer_loss_weight = args.dacer_loss_weight
        self.text_q_margin = args.text_q_margin
        self.text_action_mode = args.text_action_mode

        self.predictor_step_num = args.predictor_step_num
        self.disable_state_predictor = args.disable_state_predictor
        self.detach_predictor = args.detach_predictor

        # Image processor (for replay buffer obs_z encoding)
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        # Load VLM
        device = "cuda"
        self.use_lora = bool(args.use_lora)
        self.vlm_model, self.processor = load_model(
            args.vlm_model_id,
            use_lora=self.use_lora,
            device=device,
        )
        self.device = device

        # VLM config
        self.is_qwen35 = is_qwen35(args.vlm_model_id)
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size
        num_layers = vlm_cfg.num_hidden_layers
        self.num_layers = num_layers
        self.vlm_num_kv_heads = vlm_cfg.num_key_value_heads
        self.vlm_head_dim = vlm_cfg.head_dim
        self.target_layer_idx = args.target_layer_idx
        self.task_prompt = ""
        self.parse_action_text = args.parse_action_text
        self.text_action_prompt = args.get_action_prompt(args.horizon)
        self.high_level_prompt = args.get_action_prompt(args.horizon)
        self.max_new_tokens = args.max_new_tokens

        # Determine which VLM layers have KV cache for cross-attention
        if self.is_qwen35:
            layer_types = vlm_cfg.layer_types
            self.attn_layer_indices = [
                i for i, lt in enumerate(layer_types) if lt == "full_attention"
            ]
        else:
            self.attn_layer_indices = list(range(num_layers))
        num_expert_layers = len(self.attn_layer_indices)

        # State projection
        state_out_dim = 4
        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        self.video_encoder = VideoEncoder()
        self._target_seq_len, state_dim = self._compute_state_dim()
        torch.cuda.empty_cache()

        # Action Expert
        expert_hidden = args.expert_hidden_size
        # PEFT wraps the model with an extra .model level
        if self.use_lora:
            rotary_emb = self.vlm_model.model.model.language_model.rotary_emb
        else:
            rotary_emb = self.vlm_model.model.language_model.rotary_emb
        self.action_expert = ActionExpert(
            num_layers=num_expert_layers,
            expert_hidden_size=expert_hidden,
            num_attention_heads=vlm_cfg.num_attention_heads,
            num_kv_heads=vlm_cfg.num_key_value_heads,
            head_dim=vlm_cfg.head_dim,
            rms_norm_eps=vlm_cfg.rms_norm_eps,
            rotary_emb=rotary_emb,
        )

        # Action in/out projections
        self.action_in_proj = nn.Linear(self.action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, self.action_dim)

        # Time MLP for adaRMS conditioning
        self.time_mlp_in = nn.Linear(expert_hidden, expert_hidden)
        self.time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        # Critic (state from VLM layer output, projected and flattened)
        self.value_head = ActionValueHead(
            in_channels=state_dim,
            action_dim=self.action_dim,
            horizon=args.horizon,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=args.num_bins,
            sparsity=args.sparsity,
        )

        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
        )
        # Project VLM state to match FluxDiT context_in_dim
        self.state_to_predictor_proj = nn.Linear(state_out_dim, hidden_image_dim)

        self.value_range = 1.0
        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-self.value_range,
                max_value=+self.value_range,
                num_bins=args.num_bins,
                clamp_to_range=True,
            )

        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    @torch.inference_mode()
    def infer(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> dict:
        state, action, q_value = self._infer(s_seq, r_seq)

        next_image, next_reward = self.prediction_head.predict_next_state(
            self._state_for_predictor(state),
            action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        return {
            "action": action,
            "a_logp": torch.zeros(s_seq.shape[0], 1, device=self.device),
            "value": q_value.item(),
            "x": state,
            "rnn_state": rnn_state,
            "next_image": next_image,
            "next_reward": next_reward,
            "action_token_ids": [],
            "parse_success": True,
        }

    def compute_loss(self, data) -> tuple[torch.Tensor, dict, dict]:
        _, _, next_q = self._infer(
            data.observations[:, self.horizon :], data.rewards[:, self.horizon :]
        )
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        state, vlm_past_kv = self._vlm_forward(curr_obs, curr_rewards)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state, vlm_past_kv)

        # Sequence (state prediction) loss
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }
        info_dict = {**critic_info, **actor_info, **seq_info}

        return total_loss, activations_dict, info_dict

    def infer_and_compute_loss(self, data) -> tuple[dict, torch.Tensor, dict, dict]:
        _, next_action, next_q = self._infer(
            data.observations[:, self.horizon :], data.rewards[:, self.horizon :]
        )
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        state, vlm_past_kv = self._vlm_forward(curr_obs, curr_rewards)
        action_chunk = data.actions[:, -self.horizon :]

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state, vlm_past_kv)

        # Sequence (state prediction) loss
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        # Actor-only loss (no critic component)
        actor_entropy_loss = actor_loss + seq_loss

        # -Q(s,a) for eligibility trace backward (detached from encoder)
        et_critic_dict = self.value_head(state.detach(), action_chunk.detach())
        if self.num_bins > 1:
            neg_value_detached = -self.hl_gauss_loss(et_critic_dict["output"]).mean()
        else:
            neg_value_detached = -et_critic_dict["output"].mean()

        next_image, next_reward = self.prediction_head.predict_next_state(
            self._state_for_predictor(state),
            next_action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        infer_dict = {
            "action": next_action,
            "value": next_q.item(),
            "rnn_state": self._dummy_state.clone(),
            "next_image": next_image,
            "next_reward": next_reward,
        }

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }
        info_dict = {**critic_info, **actor_info, **seq_info}

        et_info = {
            "actor_entropy_loss": actor_entropy_loss,
            "neg_value": neg_value_detached,
            "delta": critic_info["delta"],
        }

        return infer_dict, total_loss, activations_dict, info_dict, et_info

    ####################
    # Internal methods #
    ####################

    @torch.no_grad()
    def _compute_state_dim(self) -> tuple[int, int]:
        """Compute target sequence length and state dimension via dummy forward pass."""
        dummy_images = torch.zeros(
            1, self.seq_len, *self.observation_space_shape, device=self.device
        )
        dummy_rewards = torch.zeros(1, self.seq_len, 1, device=self.device)
        inputs = prepare_vlm_inputs(
            self.processor, dummy_images, dummy_rewards, self.task_prompt, self.is_qwen35
        )
        inputs_embeds = self._build_inputs_embeds(inputs)
        output = self._vlm_language_forward(inputs, inputs_embeds)
        hidden = output["hidden_states"][self.target_layer_idx]
        target_seq_len = hidden.shape[1]
        state_dim = target_seq_len * self.state_out_proj.out_features
        return target_seq_len, state_dim

    def _get_visual(self) -> nn.Module:
        """Get the visual encoder from the VLM model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model.visual
        return self.vlm_model.model.visual

    def _get_vlm_model_inner(self) -> nn.Module:
        """Get the inner Qwen3_5Model (handles PEFT wrapping)."""
        if self.use_lora:
            return self.vlm_model.model.model
        return self.vlm_model.model

    def _build_inputs_embeds(self, inputs: dict) -> torch.Tensor:
        """Build inputs_embeds by running video encoder on all frames and injecting last-frame embeddings.

        1. Embed input_ids to get inputs_embeds (with <image_pad> as placeholder)
        2. Run all frames through ViT → extract last frame embeddings
        3. masked_scatter last-frame embeddings into <image_pad> positions
        """
        vlm_inner = self._get_vlm_model_inner()
        inputs_embeds = vlm_inner.get_input_embeddings()(inputs["input_ids"])

        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["seq_len"]

        # Run video encoder: all frames through ViT, keep last frame
        last_frame_embeds = self.video_encoder(
            self._get_visual(),
            inputs["all_pixel_values"],
            inputs["all_image_grid_thw"],
            batch_size,
            seq_len,
        )
        last_frame_embeds = last_frame_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # masked_scatter into <image_pad> positions
        image_token_id = vlm_inner.config.image_token_id
        image_mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, last_frame_embeds)

        return inputs_embeds

    def _vlm_language_forward(self, inputs: dict, inputs_embeds: torch.Tensor):
        """Run the VLM language model with pre-built inputs_embeds (no pixel_values)."""
        vlm_inner = self._get_vlm_model_inner()

        # Compute 3D position_ids (needed for image token positions)
        position_ids = vlm_inner.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
        )

        forward_kwargs = dict(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

        # language_model forward via the outer model (handles lm_head, cache wrapping)
        return self.vlm_model.forward(**forward_kwargs)

    def _vlm_forward(self, images: torch.Tensor, rewards: torch.Tensor):
        """Run VLM forward and extract state + past_key_values (with RoPE)."""
        inputs = prepare_vlm_inputs(
            self.processor, images, rewards, self.task_prompt, self.is_qwen35
        )

        inputs_embeds = self._build_inputs_embeds(inputs)

        if self.use_lora:
            outputs = self._vlm_language_forward(inputs, inputs_embeds)
        else:
            with torch.no_grad():
                outputs = self._vlm_language_forward(inputs, inputs_embeds)

        all_hidden_states = outputs.hidden_states

        # State: target_layer_idx → out_proj → pad/truncate → flatten
        hidden = all_hidden_states[self.target_layer_idx].to(torch.float32).detach()
        state = self.state_out_proj(hidden)  # (B, vlm_seq_len, state_out_dim)
        seq_len = state.shape[1]
        if seq_len > self._target_seq_len:
            state = state[:, seq_len - self._target_seq_len :, :]
        elif seq_len < self._target_seq_len:
            pad = torch.zeros(
                state.shape[0], self._target_seq_len - seq_len, state.shape[2], device=state.device
            )
            state = torch.cat([pad, state], dim=1)
        state = state.flatten(start_dim=1)  # (B, state_dim)

        return state, outputs.past_key_values

    def _extract_kv(self, vlm_past_kv) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        """Extract (K, V) pairs for the attention layers used by ActionExpert.

        Returns:
            vlm_kv_list: list of (K, V) tuples, one per expert layer
            vlm_seq_len: sequence length of the VLM KV cache
        """
        kv_list = []
        if self.is_qwen35:
            for idx in self.attn_layer_indices:
                kv_list.append((vlm_past_kv.key_cache[idx], vlm_past_kv.value_cache[idx]))
            seq_len = vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[2]
        else:
            for j in range(self.num_layers):
                layer = vlm_past_kv.layers[j]
                kv_list.append((layer.keys, layer.values))
            seq_len = vlm_past_kv.layers[0].keys.shape[2]
        return kv_list, seq_len

    def _denoise(
        self,
        noisy_actions: torch.Tensor,
        vlm_past_kv,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run one denoising step through the Action Expert."""
        expert_hidden = self.time_mlp_in.in_features
        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_hidden, min_period=4e-3, max_period=4.0
        )
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        action_embs = self.action_in_proj(noisy_actions)  # (B, horizon, expert_hidden)
        vlm_kv_list, vlm_seq_len = self._extract_kv(vlm_past_kv)
        expert_out = self.action_expert(action_embs, vlm_kv_list, vlm_seq_len, adarms_cond)
        return self.action_out_proj(expert_out.to(torch.float32))

    def _generate_action(
        self,
        B: int,
        vlm_past_kv,
    ) -> torch.Tensor:
        """Generate action via Euler denoising. Returns (B, horizon, action_dim)."""
        noise = torch.randn(B, self.horizon, self.action_dim, device=self.device)

        def predict_velocity_fn(x_t, t):
            return self._denoise(x_t, vlm_past_kv, t)

        return euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_velocity_fn)

    def _generate_text_and_extend_kv(self, prompt: str, vlm_past_kv, max_new_tokens: int):
        """Generate text continuing from vlm_past_kv and return (generated_text, extended_kv_cache)."""
        tokenizer = self.processor.tokenizer
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        _, kv_len = self._extract_kv(vlm_past_kv)
        attn_mask = torch.ones(1, kv_len + prompt_ids.shape[1], device=self.device)
        cache_position = torch.arange(
            kv_len, kv_len + prompt_ids.shape[1], device=self.device
        )

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id

        self.vlm_model.eval()
        outputs = self.vlm_model.generate(
            input_ids=prompt_ids,
            attention_mask=attn_mask,
            past_key_values=vlm_past_kv,
            cache_position=cache_position,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
        )
        self.vlm_model.train()

        generated_ids = outputs.sequences[0, prompt_ids.shape[1] :].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self.vlm_model.train()

        return generated_text, outputs.past_key_values

    def _compute_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute scalar Q-value for a (state, action) pair."""
        q_dict = self.value_head(state, action)
        q = q_dict["output"]
        return self.hl_gauss_loss(q).view(-1) if self.num_bins > 1 else q.view(-1)

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, vlm_past_kv = self._vlm_forward(obs, rewards)
        B = obs.shape[0]
        mode = self.text_action_mode

        if mode == "none":
            action_kv = vlm_past_kv
        elif mode == "high_level":
            generated_text, action_kv = self._generate_text_and_extend_kv(
                self.high_level_prompt, vlm_past_kv, max_new_tokens=35
            )
            print(f"[HighLevel] {generated_text}")
        elif mode == "text_action":
            generated_text, action_kv = self._generate_text_and_extend_kv(
                self.text_action_prompt, vlm_past_kv, max_new_tokens=self.max_new_tokens
            )
            print(f"[TextAction] {generated_text}")
        elif mode == "pi_fast":
            raise NotImplementedError("pi_fast mode is not yet implemented")
        else:
            raise ValueError(f"Unknown text_action_mode: {mode}")

        # Diffusion action using (possibly extended) kv_cache
        diff_action = self._generate_action(B, action_kv)
        diff_q = self._compute_q(state, diff_action)

        # For text_action mode, parse generated text and compare Q values
        if mode == "text_action":
            action_array, parse_success = self.parse_action_text(generated_text, self.horizon)
            text_action = torch.from_numpy(action_array).unsqueeze(0).to(obs.device)
            text_q = self._compute_q(state, text_action)
            use_text = text_q > diff_q + self.text_q_margin
            action = torch.where(use_text.unsqueeze(-1).unsqueeze(-1), text_action, diff_action)
            q = torch.where(use_text, text_q, diff_q)
            print(
                f"[ActionSelect] diff_q={diff_q.item():.3f}, text_q={text_q.item():.3f}, "
                f"use_text={use_text.item()}, parse_success={parse_success}, "
                f"action_text={generated_text}"
            )
        else:
            action = diff_action
            q = diff_q

        return state, action, q

    @torch.no_grad()
    def _compute_target_value(
        self,
        next_q: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_dones: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = chunk_rewards.size(0)
        discounted_reward = torch.zeros(batch_size, device=self.device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=self.device)
        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()
        return discounted_reward + continuing * gamma_power * next_q

    def _compute_critic_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        curr_critic_output_dict = self.value_head(state, action_chunk)

        if self.num_bins > 1:
            maybe_update_hl_gauss_range(self, target_value)
            curr_critic_value = self.hl_gauss_loss(curr_critic_output_dict["output"]).view(-1)
            critic_loss = self.hl_gauss_loss(curr_critic_output_dict["output"], target_value)
        else:
            curr_critic_value = curr_critic_output_dict["output"].view(-1)
            critic_loss = F.mse_loss(curr_critic_value, target_value)

        delta = target_value - curr_critic_value

        activations_dict = {}

        info_dict = {
            "delta": delta.mean().item(),
            "critic_loss": critic_loss.item(),
            "curr_critic_value": curr_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
            "value_range": self.value_range,
        }

        return critic_loss, activations_dict, info_dict

    def _compute_actor_loss(
        self,
        state: torch.Tensor,
        vlm_past_kv,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Advantage-based loss + DACER loss, matching actor_critic_with_action_value."""
        B = state.shape[0]

        action = self._generate_action(B, vlm_past_kv)

        def predict_velocity_fn(a_t, t):
            return self._denoise(a_t, vlm_past_kv, t)

        total_actor_loss, advantage_dict, info_dict = compute_actor_loss_with_dacer(
            state,
            action,
            self.value_head,
            getattr(self, "hl_gauss_loss", None),
            self.num_bins,
            self.dacer_loss_weight,
            predict_velocity_fn,
        )

        activations_dict = {"critic": advantage_dict["activation"]}

        return total_actor_loss, activations_dict, info_dict

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape and project state for StatePredictionHead context.

        state: (B, target_seq_len * state_out_dim) -> (B, target_seq_len, hidden_image_dim)
        """
        B = state.shape[0]
        x = state.view(B, self._target_seq_len, -1)  # (B, target_seq_len, state_out_dim)
        return self.state_to_predictor_proj(x)  # (B, target_seq_len, hidden_image_dim)

    def _compute_sequence_loss(self, data, curr_state):
        if self.disable_state_predictor:
            dummy_loss = torch.tensor(0.0, device=curr_state.device, requires_grad=True)
            activations_dict = {"state_predictor": curr_state}
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, activations_dict, info_dict

        predictor_state = self._state_for_predictor(curr_state)
        if self.detach_predictor:
            predictor_state = predictor_state.detach()

        curr_action = data.actions[:, -1]  # (B, action_dim)

        with torch.no_grad():
            last_obs = data.observations[:, -1]  # (B, C, H, W)
            target_state_next = self.image_processor.encode(last_obs)  # (B, C', H', W')
            B, C, H, W = target_state_next.shape
            target_state_next = target_state_next.flatten(2).permute(0, 2, 1)  # (B, H'*W', C')

        reward_next = data.rewards[:, -1]  # (B, 1)
        target_reward_next = self.reward_processor.encode(reward_next)  # (B, 1, C')
        target_reward_next = target_reward_next.squeeze(1)  # (B, C')
        x1 = torch.cat(
            [target_state_next, target_reward_next.unsqueeze(1)], dim=1
        )  # (B, H'*W'+1, C')

        x0 = torch.randn_like(x1)
        shape_t = (x0.shape[0],) + (1,) * (len(x0.shape) - 1)
        t = torch.rand(shape_t, device=x1.device)

        xt = (1.0 - t) * x0 + t * x1

        pred_dict = self.prediction_head.state_predictor.forward(
            xt, t, predictor_state, curr_action
        )
        pred_vt = pred_dict["output"]

        vt = x1 - x0
        pred_loss = F.mse_loss(pred_vt, vt)

        activations_dict = {"state_predictor": pred_dict["activation"]}
        info_dict = {"seq_loss": pred_loss.item()}

        return pred_loss, activations_dict, info_dict
