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
from .vlm_backbone import get_action_prompt, load_model, parse_action_text, prepare_vlm_inputs


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
        vlm_k: torch.Tensor,  # (B, vlm_len, num_kv_heads, head_dim)
        vlm_v: torch.Tensor,  # (B, vlm_len, num_kv_heads, head_dim)
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

        # Concat VLM K/V with self K/V: [vlm_tokens, action_tokens]
        k = torch.cat([vlm_k, k_self], dim=1)  # (B, vlm_len+action_len, num_kv_heads, head_dim)
        v = torch.cat([vlm_v, v_self], dim=1)

        q = q.transpose(1, 2)  # (B, num_heads, action_len, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, vlm_len+action_len, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, vlm_len+action_len, head_dim)

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
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
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
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        for j in range(self.num_layers):
            vlm_k, vlm_v = vlm_kv_list[j]
            hidden_states = self.layers[j](hidden_states, vlm_k, vlm_v, adarms_cond)
        return self.norm(hidden_states)


class VLMActorCriticWithActionValue(nn.Module):
    """VLM backbone + dual-action generation (text + diffusion) + Action Value critic.

    Architecture (pi0.5-inspired):
    - VLM (Qwen3-VL, optional LoRA): processes images + text
    - Text Actor: VLM generates action text autoregressively (when LoRA enabled)
    - Diffusion Actor: ActionExpert cross-attends to VLM KV, denoises actions via flow matching
    - Critic: Q(state, action) with dueling architecture
    - Action Selection: at inference, the action with higher Q-value is chosen
    - Knowledge Insulation: VLM hidden states are detached before ActionExpert uses them;
      only text actor loss flows gradients through LoRA
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
        self.max_new_tokens = args.max_new_tokens

        # Image processor (for replay buffer obs_z encoding)
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )

        # Load VLM (with optional LoRA for text actor)
        device = "cuda"
        self.use_lora = bool(args.use_lora)
        self.vlm_model, self.processor = load_model(
            args.vlm_model_id,
            use_quantization=args.use_quantization,
            use_lora=self.use_lora,
            device=device,
        )
        if not self.use_lora:
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
        self.task_prompt = get_action_prompt(self.horizon) if self.use_lora else ""

        # State projection (matching QwenVLEncoder)
        state_out_dim = 4
        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        self._target_seq_len, state_dim = self._compute_state_dim()
        torch.cuda.empty_cache()

        # Action Expert
        expert_hidden = args.expert_hidden_size
        self.action_expert = ActionExpert(
            num_layers=num_layers,
            expert_hidden_size=expert_hidden,
            num_attention_heads=vlm_cfg.num_attention_heads,
            num_kv_heads=vlm_cfg.num_key_value_heads,
            head_dim=vlm_cfg.head_dim,
            rms_norm_eps=vlm_cfg.rms_norm_eps,
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
        self._last_vlm_inputs: dict | None = None

    @property
    def _vlm_language_layers(self):
        """Access VLM transformer layers, handling PEFT wrapper."""
        base = self.vlm_model.model
        if hasattr(base, "model"):  # PEFT wrapper adds extra .model
            base = base.model
        return base.language_model.layers

    def get_pad_token_id(self) -> int:
        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        return pad_token_id if pad_token_id is not None else eos_token_id

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
        state, vlm_kv_list = self._vlm_forward(s_seq, r_seq)

        B = s_seq.shape[0]
        diff_action = self._generate_action(B, vlm_kv_list)

        # Text action (if LoRA enabled)
        generated_ids: list[int] = []
        parse_success = True
        action = diff_action

        if self.use_lora:
            text_action, generated_ids, parse_success = self._generate_text_action(
                self._last_vlm_inputs
            )
            if parse_success:
                # Select action with higher Q value
                q_diff = self.value_head(state, diff_action)["output"]
                q_text = self.value_head(state, text_action)["output"]
                if self.num_bins > 1:
                    q_diff_scalar = self.hl_gauss_loss(q_diff)
                    q_text_scalar = self.hl_gauss_loss(q_text)
                else:
                    q_diff_scalar = q_diff.view(-1)
                    q_text_scalar = q_text.view(-1)

                if q_text_scalar.item() > q_diff_scalar.item():
                    action = text_action

        # Q value for selected action
        q_dict = self.value_head(state, action)
        q_value = q_dict["output"]
        q_value = self.hl_gauss_loss(q_value).item() if self.num_bins > 1 else q_value.item()

        c, h, w = self.observation_space_shape
        return {
            "action": action,
            "a_logp": torch.zeros(B, 1, device=self.device),
            "value": q_value,
            "x": state,
            "rnn_state": rnn_state,
            "next_image": np.zeros((h, w, c), dtype=np.float32),
            "next_reward": 0.0,
            "action_token_ids": generated_ids,
            "parse_success": parse_success,
        }

    def compute_loss(self, data) -> tuple[torch.Tensor, dict, dict]:
        target_value = self._compute_target_value(data)["target_value"]

        curr_obs = data.observations[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        state, vlm_kv_list = self._vlm_forward(curr_obs, curr_rewards)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Diffusion actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state, vlm_kv_list)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss

        # Text actor loss (if LoRA enabled)
        if self.use_lora:
            curr_action_token_ids = data.action_token_ids[:, -1]  # (B, max_new_tokens)
            text_actor_loss, text_info = self._compute_text_actor_loss(
                state, curr_obs, curr_rewards, curr_action_token_ids
            )
            total_loss = total_loss + text_actor_loss
            actor_info.update(text_info)

        activations_dict = {
            "state": state,
            **critic_activations,
            **actor_activations,
        }
        info_dict = {**critic_info, **actor_info}

        return total_loss, activations_dict, info_dict

    def infer_and_compute_loss(self, data) -> tuple[dict, torch.Tensor, dict, dict]:
        target_dict = self._compute_target_value(data)
        target_value = target_dict["target_value"]

        curr_obs = data.observations[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        state, vlm_kv_list = self._vlm_forward(curr_obs, curr_rewards)
        action_chunk = data.actions[:, -self.horizon :]

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Diffusion actor loss (advantage + DACER)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state, vlm_kv_list)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss

        # Text actor loss (if LoRA enabled)
        if self.use_lora:
            curr_action_token_ids = data.action_token_ids[:, -1]
            text_actor_loss, text_info = self._compute_text_actor_loss(
                state, curr_obs, curr_rewards, curr_action_token_ids
            )
            total_loss = total_loss + text_actor_loss
            actor_info.update(text_info)

        B = curr_obs.shape[0]
        generated_ids: list[int] = []
        parse_success = True
        with torch.no_grad():
            diff_action = self._generate_action(B, vlm_kv_list)
            infer_action = diff_action

            if self.use_lora:
                text_action, generated_ids, parse_success = self._generate_text_action(
                    self._last_vlm_inputs
                )
                if parse_success:
                    q_diff = self.value_head(state, diff_action)["output"]
                    q_text = self.value_head(state, text_action)["output"]
                    if self.num_bins > 1:
                        q_diff_s = self.hl_gauss_loss(q_diff).item()
                        q_text_s = self.hl_gauss_loss(q_text).item()
                    else:
                        q_diff_s = q_diff.item()
                        q_text_s = q_text.item()
                    if q_text_s > q_diff_s:
                        infer_action = text_action

        c, h, w = self.observation_space_shape
        infer_dict = {
            "action": infer_action,
            "value": target_dict["q_value"].mean().item(),
            "rnn_state": self._dummy_state.clone(),
            "next_image": np.zeros((h, w, c), dtype=np.float32),
            "next_reward": 0.0,
            "action_token_ids": generated_ids,
            "parse_success": parse_success,
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

    def _vlm_forward(
        self, images: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run VLM forward and extract state + KV for each layer.

        Args:
            images: (B, T, C, H, W)
            rewards: (B, T, 1)

        Returns:
            state: (B, state_dim) projected and flattened hidden state
            vlm_kv_list: list of (K, V) per layer, each (B, vlm_len, num_kv_heads, head_dim)
        """
        with torch.no_grad():
            inputs = prepare_vlm_inputs(self.processor, images, rewards, self.task_prompt)
            self._last_vlm_inputs = inputs

            outputs = self.vlm_model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )

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

        # Extract K/V from each VLM layer using VLM's own projections
        vlm_layers = self._vlm_language_layers
        vlm_kv_list = []
        for j in range(self.num_layers):
            hs_j = all_hidden_states[j].detach()
            normed = vlm_layers[j].input_layernorm(hs_j)
            k = vlm_layers[j].self_attn.k_proj(normed)
            v = vlm_layers[j].self_attn.v_proj(normed)
            B, vlm_len = k.shape[:2]
            k = k.view(B, vlm_len, self.vlm_num_kv_heads, self.vlm_head_dim)
            v = v.view(B, vlm_len, self.vlm_num_kv_heads, self.vlm_head_dim)
            k = vlm_layers[j].self_attn.k_norm(k)
            vlm_kv_list.append((k.detach(), v.detach()))

        return state, vlm_kv_list

    def _denoise(
        self,
        noisy_actions: torch.Tensor,
        vlm_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run one denoising step through the Action Expert."""
        expert_hidden = self.time_mlp_in.in_features
        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_hidden, min_period=4e-3, max_period=4.0
        )
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        action_embs = self.action_in_proj(noisy_actions)  # (B, horizon, expert_hidden)
        expert_out = self.action_expert(action_embs, vlm_kv_list, adarms_cond)
        return self.action_out_proj(expert_out.to(torch.float32))

    def _generate_action(
        self,
        B: int,
        vlm_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Generate action via Euler denoising. Returns (B, horizon, action_dim)."""
        noise = torch.randn(B, self.horizon, self.action_dim, device=self.device)

        def predict_velocity_fn(x_t, t):
            return self._denoise(x_t, vlm_kv_list, t)

        return euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_velocity_fn)

    @torch.no_grad()
    def _generate_text_action(
        self, inputs: dict
    ) -> tuple[torch.Tensor, list[int], bool]:
        """Generate action via VLM text generation.

        Args:
            inputs: VLM model inputs from prepare_vlm_inputs

        Returns:
            action: (1, horizon, action_dim) tensor
            generated_ids: list of token IDs
            parse_success: whether parsing succeeded
        """
        pad_token_id = self.get_pad_token_id()
        eos_token_id = self.processor.tokenizer.eos_token_id

        generated = self.vlm_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        generated_ids = new_tokens[0].tolist()

        action_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        action_array, parse_success = parse_action_text(action_text, self.horizon)
        action_tensor = torch.from_numpy(action_array).unsqueeze(0).to(self.device)

        return action_tensor, generated_ids, parse_success

    def _compute_text_log_prob(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
        action_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log prob of stored action tokens via teacher-forced VLM forward.

        Args:
            images: (B, T, C, H, W)
            rewards: (B, T, 1)
            action_token_ids: (B, max_new_tokens) - padded token IDs

        Returns:
            log_probs: (B,)
        """
        inputs = prepare_vlm_inputs(self.processor, images, rewards, self.task_prompt)

        pad_token_id = self.get_pad_token_id()
        target_mask = (action_token_ids != pad_token_id).float()
        action_len = action_token_ids.size(1)
        prompt_len = inputs["input_ids"].size(1)

        # Concatenate prompt + action tokens
        combined_input_ids = torch.cat([inputs["input_ids"], action_token_ids], dim=1)
        combined_attention_mask = torch.cat(
            [inputs["attention_mask"], target_mask.long()], dim=1
        )

        # Forward pass with gradient (LoRA)
        outputs = self.vlm_model.forward(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            return_dict=True,
        )

        # Log prob from logits predicting action tokens
        relevant_logits = outputs.logits[:, prompt_len - 1 : prompt_len + action_len - 1, :]
        log_prob_dist = F.log_softmax(relevant_logits, dim=-1)
        token_log_probs = log_prob_dist.gather(
            2, action_token_ids.unsqueeze(2)
        ).squeeze(2)
        batch_log_probs = (token_log_probs * target_mask).sum(dim=1)

        return batch_log_probs

    def _compute_text_actor_loss(
        self,
        state: torch.Tensor,
        curr_obs: torch.Tensor,
        curr_rewards: torch.Tensor,
        action_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Advantage-weighted text actor loss.

        Args:
            state: (B, state_dim) - detached state for advantage computation
            curr_obs: (B, T, C, H, W)
            curr_rewards: (B, T, 1)
            action_token_ids: (B, max_new_tokens) - stored action token IDs

        Returns:
            loss: scalar
            info: dict with metrics
        """
        pad_token_id = self.get_pad_token_id()

        # Check which samples have valid (non-empty) action tokens
        has_tokens = (action_token_ids != pad_token_id).any(dim=1)  # (B,)

        if not has_tokens.any():
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero, {"text_actor_loss": 0.0}

        # Filter to valid samples only
        valid_obs = curr_obs[has_tokens]
        valid_rewards = curr_rewards[has_tokens]
        valid_tokens = action_token_ids[has_tokens]
        valid_state = state[has_tokens].detach()

        # Compute log prob with gradient through LoRA
        log_prob = self._compute_text_log_prob(valid_obs, valid_rewards, valid_tokens)

        # Parse tokens back to action tensors for advantage computation
        actions_list = []
        for i in range(valid_tokens.shape[0]):
            token_ids = valid_tokens[i].tolist()
            token_ids = [t for t in token_ids if t != pad_token_id]
            action_text = self.processor.tokenizer.decode(
                token_ids, skip_special_tokens=True
            ).strip()
            action_array, _ = parse_action_text(action_text, self.horizon)
            actions_list.append(torch.from_numpy(action_array).to(self.device))
        parsed_actions = torch.stack(actions_list)  # (valid_B, horizon, action_dim)

        # Compute advantage (no grad through value head)
        with torch.no_grad():
            for param in self.value_head.parameters():
                param.requires_grad_(False)
            adv_dict = self.value_head.get_advantage(valid_state, parsed_actions)
            advantage = adv_dict["output"]
            if self.num_bins > 1:
                advantage = self.hl_gauss_loss(advantage)
            advantage = advantage.view(-1)
            for param in self.value_head.parameters():
                param.requires_grad_(True)

        # Advantage-weighted policy gradient
        text_actor_loss = -(log_prob * advantage.detach()).mean()

        info = {"text_actor_loss": text_actor_loss.item()}
        return text_actor_loss, info

    @torch.no_grad()
    def _compute_target_value(self, data) -> dict:
        next_obs = data.observations[:, self.horizon :]
        next_rewards = data.rewards[:, self.horizon :]
        next_state, next_vlm_kv = self._vlm_forward(next_obs, next_rewards)

        B = next_obs.shape[0]

        # Diffusion action
        next_diff_action = self._generate_action(B, next_vlm_kv)
        next_q_diff = self.value_head(next_state, next_diff_action)["output"]
        next_q_diff = (
            self.hl_gauss_loss(next_q_diff).view(-1)
            if self.num_bins > 1
            else next_q_diff.view(-1)
        )

        next_action = next_diff_action
        next_q = next_q_diff

        # Text action (if LoRA enabled) — select max Q across both
        if self.use_lora:
            text_action, _, text_parse_ok = self._generate_text_action(self._last_vlm_inputs)
            if text_parse_ok:
                next_q_text = self.value_head(next_state, text_action)["output"]
                next_q_text = (
                    self.hl_gauss_loss(next_q_text).view(-1)
                    if self.num_bins > 1
                    else next_q_text.view(-1)
                )
                use_text = next_q_text > next_q_diff
                next_q = torch.where(use_text, next_q_text, next_q_diff)
                next_action = torch.where(
                    use_text.unsqueeze(-1).unsqueeze(-1).expand_as(next_diff_action),
                    text_action.expand_as(next_diff_action),
                    next_diff_action,
                )

        # Discounted reward over horizon
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        batch_size = chunk_rewards.size(0)
        discounted_reward = torch.zeros(batch_size, device=self.device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=self.device)
        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()

        target_value = discounted_reward + continuing * gamma_power * next_q

        return {
            "target_value": target_value,
            "action": next_action,
            "q_value": next_q,
            "next_state": next_state,
        }

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
        vlm_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict, dict]:
        """Advantage-based loss + DACER loss, matching actor_critic_with_action_value."""
        B = state.shape[0]

        action = self._generate_action(B, vlm_kv_list)

        def predict_velocity_fn(a_t, t):
            return self._denoise(a_t, vlm_kv_list, t)

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
