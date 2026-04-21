# SPDX-License-Identifier: MIT
from collections.abc import Callable

import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .cross_attention_expert import (
    CrossAttentionActionExpert,
    CrossAttentionActionValueHead,
    create_sinusoidal_pos_embedding,
)
from .diffusion_utils import compute_actor_loss_with_dacer, euler_denoise
from .image_processor import ImageProcessor
from .prediction_head import StatePredictionHead
from .reward_processor import RewardProcessor
from .value_head import maybe_update_hl_gauss_range
from .video_encoder import VideoEncoder
from .vlm_backbone import is_qwen35, load_model, prepare_vlm_inputs


class VLMActorCriticWithActionValue(nn.Module):
    """VLM + cross-attention Action Expert (flow matching) + cross-attention Action Value critic.

    GR00T / ABot_M0 style:
    - VLM last_hidden is used as the cross-attention context for both actor and critic.
    - Actor: action tokens cross-attend to last_hidden; denoised with flow matching.
    - Critic: dueling Q(s, a) with value/advantage query tokens cross-attending to last_hidden.
    """

    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        parse_action_text: Callable[[str], tuple[np.ndarray, bool]] | None,
        gamma: float,
        num_bins: int,
        seq_len: int,
        horizon: int,
        critic_loss_weight: float,
        denoising_steps: int,
        denoising_time: float,
        dacer_loss_weight: float,
        text_q_margin: float,
        text_action_mode: str,
        predictor_step_num: int,
        disable_state_predictor: bool,
        detach_predictor: bool,
        image_processor_type: str,
        use_lora: bool,
        vlm_model_id: str,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
        expert_hidden_size: int,
        actor_block_num: int,
        critic_block_num: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_bins = num_bins
        self.seq_len = seq_len
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.observation_space_shape = observation_space_shape
        self.critic_loss_weight = critic_loss_weight
        self.denoising_steps = denoising_steps
        self.denoising_time = denoising_time
        self.dacer_loss_weight = dacer_loss_weight
        self.text_q_margin = text_q_margin
        self.text_action_mode = text_action_mode

        self.predictor_step_num = predictor_step_num
        self.disable_state_predictor = disable_state_predictor
        self.detach_predictor = detach_predictor

        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=image_processor_type
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        device = "cuda"
        self.use_lora = bool(use_lora)
        self.vlm_model, self.processor = load_model(
            vlm_model_id,
            use_lora=self.use_lora,
            device=device,
        )
        self.device = device

        self.is_qwen35 = is_qwen35(vlm_model_id)
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size
        self.num_layers = vlm_cfg.num_hidden_layers
        self.parse_action_text = parse_action_text
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = pad_token_id

        # Full-attention layer indices (used only for reading KV seq_len during text gen)
        if self.is_qwen35:
            layer_types = vlm_cfg.layer_types
            self.attn_layer_indices = [
                i for i, lt in enumerate(layer_types) if lt == "full_attention"
            ]
        else:
            self.attn_layer_indices = list(range(self.num_layers))

        expert_hidden = expert_hidden_size
        self.video_encoder = VideoEncoder()

        # Action Expert: cross-attends to VLM last_hidden
        self.action_expert = CrossAttentionActionExpert(
            num_layers=actor_block_num,
            hidden_size=expert_hidden,
            context_dim=vlm_hidden_size,
            num_heads=vlm_cfg.num_key_value_heads,
            head_dim=vlm_cfg.head_dim,
            rms_norm_eps=vlm_cfg.rms_norm_eps,
        )

        self.action_in_proj = nn.Linear(self.action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, self.action_dim)

        # Time MLP for adaRMS conditioning
        self.time_mlp_in = nn.Linear(expert_hidden, expert_hidden)
        self.time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        # Critic: cross-attends to VLM last_hidden
        self.value_head = CrossAttentionActionValueHead(
            num_layers=critic_block_num,
            hidden_size=expert_hidden,
            context_dim=vlm_hidden_size,
            num_heads=vlm_cfg.num_key_value_heads,
            head_dim=vlm_cfg.head_dim,
            rms_norm_eps=vlm_cfg.rms_norm_eps,
            action_dim=self.action_dim,
            horizon=horizon,
            num_bins=num_bins,
        )

        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_block_num=predictor_block_num,
        )
        self.state_to_predictor_proj = nn.Linear(vlm_hidden_size, hidden_image_dim)

        self.value_range = 1.0
        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-self.value_range,
                max_value=+self.value_range,
                num_bins=num_bins,
                clamp_to_range=True,
            )

        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        return self.processor.tokenizer.encode(task_prompt, add_special_tokens=False)

    def decode_task_prompt_ids(self, token_ids: torch.Tensor) -> list[str]:
        results = []
        for i in range(token_ids.shape[0]):
            ids = token_ids[i]
            mask = ids != self.pad_token_id
            valid_ids = ids[mask].tolist()
            text = self.processor.tokenizer.decode(valid_ids, skip_special_tokens=True)
            results.append(text)
        return results

    @torch.inference_mode()
    def infer(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
        task_prompts: list[str],
    ) -> dict:
        state, action, q_value = self._infer(s_seq, task_prompts)

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
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, _, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, _ = self._vlm_forward(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state)

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
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, next_action, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, _ = self._vlm_forward(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state)

        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

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
        """Embed input_ids, then replace image_pad positions with ViT-encoded last-frame embeddings."""
        vlm_inner = self._get_vlm_model_inner()
        inputs_embeds = vlm_inner.get_input_embeddings()(inputs["input_ids"])

        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["seq_len"]

        last_frame_embeds = self.video_encoder(
            self._get_visual(),
            inputs["all_pixel_values"],
            inputs["all_image_grid_thw"],
            batch_size,
            seq_len,
        )
        last_frame_embeds = last_frame_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        image_token_id = vlm_inner.config.image_token_id
        image_mask = (inputs["input_ids"] == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, last_frame_embeds)

        return inputs_embeds

    def _vlm_language_forward(self, inputs: dict, inputs_embeds: torch.Tensor):
        """Run the VLM language model with pre-built inputs_embeds (no pixel_values)."""
        vlm_inner = self._get_vlm_model_inner()

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

        return self.vlm_model.forward(**forward_kwargs)

    def _vlm_forward(self, images: torch.Tensor, task_prompts: list[str]):
        """Run VLM forward. Returns (last_hidden, past_key_values)."""
        inputs = prepare_vlm_inputs(
            self.processor,
            images,
            task_prompts,
            self.is_qwen35,
        )

        inputs_embeds = self._build_inputs_embeds(inputs)

        if self.use_lora:
            outputs = self._vlm_language_forward(inputs, inputs_embeds)
        else:
            with torch.no_grad():
                outputs = self._vlm_language_forward(inputs, inputs_embeds)

        self._last_input_ids = inputs["input_ids"]
        last_hidden = outputs.hidden_states[-1].to(torch.float32)
        return last_hidden, outputs.past_key_values

    def _kv_seq_len(self, vlm_past_kv) -> int:
        """Read the KV-cache sequence length (for text-generation position tracking)."""
        if self.is_qwen35:
            return vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[2]
        return vlm_past_kv.layers[0].keys.shape[2]

    def _denoise(
        self,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One flow-matching denoising step. state: (B, L, H) VLM last_hidden."""
        expert_hidden = self.time_mlp_in.in_features
        time_emb = create_sinusoidal_pos_embedding(
            timestep, expert_hidden, min_period=4e-3, max_period=4.0
        )
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        action_embs = self.action_in_proj(noisy_actions)
        expert_out = self.action_expert(action_embs, state, adarms_cond)
        out = self.action_out_proj(expert_out.to(torch.float32))
        return torch.tanh(out)

    def _generate_action(self, B: int, state: torch.Tensor) -> torch.Tensor:
        """Generate action via Euler denoising. Returns (B, horizon, action_dim)."""
        noise = torch.randn(B, self.horizon, self.action_dim, device=self.device)

        def predict_fn(x_t, t):
            return self._denoise(x_t, state, t)

        return euler_denoise(noise, self.denoising_time, self.denoising_steps, predict_fn)

    def _generate_text_and_extend_kv(self, prompt: str, vlm_past_kv, max_new_tokens: int):
        """Generate text via manual forward loop (greedy). Returns (first_item_text, extended_kv)."""
        tokenizer = self.processor.tokenizer
        kv_len = self._kv_seq_len(vlm_past_kv)
        eos_token_id = tokenizer.eos_token_id

        if prompt:
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            if self.is_qwen35:
                B = vlm_past_kv.key_cache[self.attn_layer_indices[0]].shape[0]
            else:
                B = vlm_past_kv.layers[0].keys.shape[0]
            next_ids = prompt_ids.expand(B, -1)
            cur_pos = kv_len
        else:
            next_ids = self._last_input_ids[:, -1:].to(self.device)
            B = next_ids.shape[0]
            cur_pos = kv_len - 1

        rope_deltas = self.vlm_model.model.rope_deltas

        self.vlm_model.eval()

        generated_tokens = [[] for _ in range(B)]
        finished = [False] * B

        for step in range(max_new_tokens + 1):
            seq_len = next_ids.shape[1]
            cache_position = torch.arange(cur_pos, cur_pos + seq_len, device=self.device)

            text_pos = cache_position.view(1, 1, -1).expand(1, B, -1)
            if rope_deltas is not None:
                text_pos = text_pos + rope_deltas.unsqueeze(0)
            position_ids = text_pos.expand(3, -1, -1)

            outputs = self.vlm_model(
                input_ids=next_ids,
                attention_mask=torch.ones(B, cur_pos + seq_len, device=self.device),
                past_key_values=vlm_past_kv,
                cache_position=cache_position,
                position_ids=position_ids,
            )

            vlm_past_kv = outputs.past_key_values
            cur_pos = cur_pos + seq_len

            if step == 0 and not prompt:
                next_ids = outputs.logits[:, -1:, :].argmax(dim=-1)
                continue

            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            for b in range(B):
                if not finished[b]:
                    tid = next_token[b, 0].item()
                    if tid == eos_token_id:
                        finished[b] = True
                    else:
                        generated_tokens[b].append(tid)

            if all(finished):
                break

            next_ids = next_token

        self.vlm_model.train()

        first_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return first_text, vlm_past_kv

    def _compute_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_dict = self.value_head(state, action)
        q = q_dict["output"]
        return self.hl_gauss_loss(q).view(-1) if self.num_bins > 1 else q.view(-1)

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, vlm_past_kv = self._vlm_forward(obs, task_prompts)
        B = obs.shape[0]
        mode = self.text_action_mode

        if mode == "none":
            pass
        elif mode == "high_level":
            generated_text, _ = self._generate_text_and_extend_kv(
                "", vlm_past_kv, max_new_tokens=30
            )
            print(f"[HighLevel] {generated_text}")
        elif mode == "text_action":
            generated_text, _ = self._generate_text_and_extend_kv(
                "", vlm_past_kv, max_new_tokens=self.max_new_tokens
            )
            print(f"[TextAction] {generated_text}")
        elif mode == "pi_fast":
            raise NotImplementedError("pi_fast mode is not yet implemented")
        else:
            raise ValueError(f"Unknown text_action_mode: {mode}")

        diff_action = self._generate_action(B, state)
        diff_q = self._compute_q(state, diff_action)

        if mode == "text_action":
            action_array, parse_success = self.parse_action_text(generated_text)
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
    ) -> tuple[torch.Tensor, dict, dict]:
        """Advantage-based loss + DACER loss."""
        B = state.shape[0]

        action = self._generate_action(B, state)

        def predict_fn(a_t, t):
            return self._denoise(a_t, state, t)

        total_actor_loss, advantage_dict, info_dict = compute_actor_loss_with_dacer(
            state,
            action,
            self.value_head,
            self.hl_gauss_loss if self.num_bins > 1 else None,
            self.num_bins,
            self.dacer_loss_weight,
            predict_fn,
        )

        activations_dict = {"critic": advantage_dict["activation"]}
        return total_actor_loss, activations_dict, info_dict

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Project VLM last_hidden (B, L, H) into the predictor's context dim."""
        return self.state_to_predictor_proj(state)

    def _compute_sequence_loss(self, data, curr_state):
        if self.disable_state_predictor:
            dummy_loss = torch.tensor(0.0, device=curr_state.device, requires_grad=True)
            activations_dict = {"state_predictor": curr_state}
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, activations_dict, info_dict

        predictor_state = self._state_for_predictor(curr_state)
        if self.detach_predictor:
            predictor_state = predictor_state.detach()

        curr_action = data.actions[:, -1]

        with torch.no_grad():
            last_obs = data.observations[:, -1]
            target_state_next = self.image_processor.encode(last_obs)
            B, C, H, W = target_state_next.shape
            target_state_next = target_state_next.flatten(2).permute(0, 2, 1)

        reward_next = data.rewards[:, -1]
        target_reward_next = self.reward_processor.encode(reward_next)
        target_reward_next = target_reward_next.squeeze(1)
        x1 = torch.cat([target_state_next, target_reward_next.unsqueeze(1)], dim=1)

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
