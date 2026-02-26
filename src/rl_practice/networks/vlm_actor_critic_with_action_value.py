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
from .value_head import ActionValueHead
from .vlm_backbone import load_model, prepare_vlm_inputs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. q/k: (B, heads, seq, head_dim), cos/sin: (B, seq, head_dim)."""
    cos = cos.unsqueeze(1)  # (B, 1, seq, head_dim)
    sin = sin.unsqueeze(1)
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
        vlm_past_kv,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        B, action_len, _ = hidden_states.shape
        vlm_seq_len = vlm_past_kv[0][0].shape[2]

        # RoPE positions for action tokens: right after VLM sequence
        action_pos = torch.arange(
            vlm_seq_len, vlm_seq_len + action_len, device=hidden_states.device
        )
        action_pos_ids = action_pos.unsqueeze(0).expand(B, -1)  # (B, action_len)
        cos, sin = self.rotary_emb(hidden_states, position_ids=action_pos_ids)

        for j in range(self.num_layers):
            vlm_k, vlm_v = vlm_past_kv[j]
            hidden_states = self.layers[j](hidden_states, vlm_k, vlm_v, cos, sin, adarms_cond)
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

        # Image processor (for replay buffer obs_z encoding)
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )

        # Load VLM (frozen)
        device = "cuda"
        self.vlm_model, self.processor = load_model(
            args.vlm_model_id,
            use_quantization=args.use_quantization,
            use_lora=False,
            device=device,
        )
        self.vlm_model.requires_grad_(False)
        self.vlm_model.gradient_checkpointing_enable()
        self.device = device

        # VLM config
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size  # 2048
        num_layers = vlm_cfg.num_hidden_layers  # 28
        self.num_layers = num_layers
        self.vlm_num_kv_heads = vlm_cfg.num_key_value_heads
        self.vlm_head_dim = vlm_cfg.head_dim
        self.target_layer_idx = args.target_layer_idx
        self.task_prompt = ""

        # State projection (matching QwenVLEncoder)
        state_out_dim = 4
        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        self._target_seq_len, state_dim = self._compute_state_dim()
        torch.cuda.empty_cache()

        # Action Expert
        expert_hidden = args.expert_hidden_size
        rotary_emb = self.vlm_model.model.language_model.rotary_emb
        self.action_expert = ActionExpert(
            num_layers=num_layers,
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

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-args.value_range,
                max_value=+args.value_range,
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

        c, h, w = self.observation_space_shape
        return {
            "action": action,
            "a_logp": torch.zeros(s_seq.shape[0], 1, device=self.device),
            "value": q_value.item(),
            "x": state,
            "rnn_state": rnn_state,
            "next_image": np.zeros((h, w, c), dtype=np.float32),
            "next_reward": 0.0,
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

        total_loss = self.critic_loss_weight * critic_loss + actor_loss

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
        }
        info_dict = {**critic_info, **actor_info}

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

        total_loss = self.critic_loss_weight * critic_loss + actor_loss

        c, h, w = self.observation_space_shape
        infer_dict = {
            "action": next_action,
            "value": next_q.item(),
            "rnn_state": self._dummy_state.clone(),
            "next_image": np.zeros((h, w, c), dtype=np.float32),
            "next_reward": 0.0,
        }

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
        }
        info_dict = {**critic_info, **actor_info}

        return infer_dict, total_loss, activations_dict, info_dict

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
        model_inputs = prepare_vlm_inputs(
            self.processor, dummy_images, dummy_rewards, self.task_prompt
        )
        output = self.vlm_model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][self.target_layer_idx]
        target_seq_len = hidden.shape[1]
        state_dim = target_seq_len * self.state_out_proj.out_features
        return target_seq_len, state_dim

    @torch.no_grad()
    def _vlm_forward(self, images: torch.Tensor, rewards: torch.Tensor):
        """Run VLM forward and extract state + past_key_values (with RoPE)."""
        inputs = prepare_vlm_inputs(self.processor, images, rewards, self.task_prompt)

        # eval mode: bypass gradient checkpointing forcing use_cache=False
        self.vlm_model.eval()
        outputs = self.vlm_model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        self.vlm_model.train()

        all_hidden_states = outputs.hidden_states

        # State: matching QwenVLEncoder (target_layer_idx → out_proj → pad/truncate → flatten)
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
        expert_out = self.action_expert(action_embs, vlm_past_kv, adarms_cond)
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

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, vlm_past_kv = self._vlm_forward(obs, rewards)
        B = obs.shape[0]
        action = self._generate_action(B, vlm_past_kv)
        q_dict = self.value_head(state, action)
        q = q_dict["output"]
        q = self.hl_gauss_loss(q).view(-1) if self.num_bins > 1 else q.view(-1)
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
