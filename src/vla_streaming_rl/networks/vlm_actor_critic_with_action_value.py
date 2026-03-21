# SPDX-License-Identifier: MIT
import argparse

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .action_tokenizer import ActionTokenizer
from .image_processor import ImageProcessor
from .prediction_head import StatePredictionHead
from .reward_processor import RewardProcessor
from .value_head import ActionValueHead, maybe_update_hl_gauss_range
from .video_encoder import VideoEncoder
from .vlm_backbone import load_model, prepare_vlm_inputs


class VLMActorCriticWithActionValue(nn.Module):
    """VLM backbone + Action Token policy + Action Value critic.

    Architecture:
    - VLM (Qwen3.5, frozen or LoRA): processes images + text
    - Actor: VLM lm_head outputs logits over action token bins (categorical policy)
    - Critic: Q(state, action) with dueling architecture
    - Actor loss: policy gradient weighted by Q-value advantage
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

        self.predictor_step_num = args.predictor_step_num
        self.disable_state_predictor = args.disable_state_predictor
        self.detach_predictor = args.detach_predictor

        # Image processor (for replay buffer obs_z encoding)
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        # Load VLM (LoRA is always enabled for action token policy)
        device = "cuda"
        self.vlm_model, self.processor = load_model(
            args.vlm_model_id,
            use_lora=True,
            device=device,
        )
        self.device = device

        # VLM config
        vlm_cfg = self.vlm_model.config.text_config
        vlm_hidden_size = vlm_cfg.hidden_size
        self.target_layer_idx = args.target_layer_idx
        self.default_task_prompt = args.prompt
        self.max_prompt_tokens = args.max_prompt_tokens
        self.pad_token_id = args.pad_token_id

        # Action tokenizer: maps continuous actions to discrete token IDs
        vocab_size = vlm_cfg.vocab_size
        self.action_tokenizer = ActionTokenizer(vocab_size)
        self.action_token_begin_idx = self.action_tokenizer.action_token_begin_idx
        self.n_action_bins = self.action_tokenizer.n_bins

        self.video_encoder = VideoEncoder()

        # State via projection from VLM hidden states
        state_out_dim = 4
        self.state_out_proj = nn.Linear(vlm_hidden_size, state_out_dim).to(device)
        self._target_seq_len, state_dim = self._compute_state_dim()
        torch.cuda.empty_cache()

        # Critic (state, action -> Q-value)
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

    def tokenize_task_prompt(self, task_prompt: str) -> list[int]:
        """Tokenize a task prompt string into token IDs."""
        return self.processor.tokenizer.encode(task_prompt, add_special_tokens=False)

    def decode_task_prompt_ids(self, token_ids: torch.Tensor) -> list[str]:
        """Decode task prompt token IDs back to strings.

        Args:
            token_ids: (B, max_prompt_tokens) tensor of token IDs
        Returns:
            List of decoded strings, one per batch element
        """
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
        task_prompts: list[str] | None = None,
    ) -> dict:
        if task_prompts is None:
            task_prompts = [self.default_task_prompt] * s_seq.shape[0]
        state, action, log_probs, q_value = self._infer(s_seq, task_prompts)

        next_image, next_reward = self.prediction_head.predict_next_state(
            self._state_for_predictor(state),
            action[:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        return {
            "action": action,
            "a_logp": log_probs.sum(dim=-1, keepdim=True),  # (B, 1)
            "value": q_value.item(),
            "x": state,
            "rnn_state": rnn_state,
            "next_image": next_image,
            "next_reward": next_reward,
        }

    def compute_loss(self, data) -> tuple[torch.Tensor, dict, dict]:
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, _, _, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, logits = self._vlm_forward(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (policy gradient with Q-value advantage)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(
            state, logits
        )

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
        next_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -1])
        curr_prompts = self.decode_task_prompt_ids(data.task_prompt_token_ids[:, -self.horizon - 1])

        _, next_action, _, next_q = self._infer(data.observations[:, self.horizon :], next_prompts)
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        target_value = self._compute_target_value(next_q, chunk_rewards, chunk_dones)

        curr_obs = data.observations[:, : -self.horizon]
        state, logits = self._vlm_forward(curr_obs, curr_prompts)
        action_chunk = data.actions[:, -self.horizon :]

        # Critic loss
        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state, action_chunk, target_value
        )

        # Actor loss (policy gradient with Q-value advantage)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(
            state, logits
        )

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
        inputs = prepare_vlm_inputs(
            self.processor,
            dummy_images,
            [self.default_task_prompt],
        )
        inputs_embeds = self._build_inputs_embeds(inputs)
        output = self._vlm_language_forward(inputs, inputs_embeds)
        hidden = output["hidden_states"][self.target_layer_idx]
        target_seq_len = hidden.shape[1]
        state_dim = target_seq_len * self.state_out_proj.out_features
        return target_seq_len, state_dim

    def _get_visual(self) -> nn.Module:
        """Get the visual encoder from the VLM model (PEFT wrapped)."""
        return self.vlm_model.model.model.visual

    def _get_vlm_model_inner(self) -> nn.Module:
        """Get the inner Qwen3_5Model (PEFT wrapped)."""
        return self.vlm_model.model.model

    def _build_inputs_embeds(self, inputs: dict) -> torch.Tensor:
        """Build inputs_embeds by running video encoder on all frames and injecting last-frame embeddings."""
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
            use_cache=False,
            return_dict=True,
        )

        return self.vlm_model.forward(**forward_kwargs)

    def _vlm_forward(
        self, images: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run VLM forward. Returns (state, action_logits).

        state: (B, state_dim) flattened projection of hidden states
        action_logits: (B, n_action_bins) logits over action token bins from last token
        """
        inputs = prepare_vlm_inputs(
            self.processor,
            images,
            task_prompts,
        )

        inputs_embeds = self._build_inputs_embeds(inputs)

        outputs = self._vlm_language_forward(inputs, inputs_embeds)

        # State from hidden states
        all_hidden_states = outputs.hidden_states
        hidden = all_hidden_states[self.target_layer_idx].to(torch.float32).detach()
        state = self.state_out_proj(hidden)
        seq_len = state.shape[1]
        if seq_len > self._target_seq_len:
            state = state[:, seq_len - self._target_seq_len :, :]
        elif seq_len < self._target_seq_len:
            pad = torch.zeros(
                state.shape[0],
                self._target_seq_len - seq_len,
                state.shape[2],
                device=state.device,
            )
            state = torch.cat([pad, state], dim=1)
        state = state.flatten(start_dim=1)

        # Action logits: slice out the action token range from last-position logits
        last_logits = outputs.logits[:, -1, :]  # (B, vocab_size)
        action_logits = last_logits[:, self.action_token_begin_idx : self.action_token_begin_idx + self.n_action_bins]
        # (B, n_action_bins)

        return state, action_logits

    def _sample_action(
        self, action_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from categorical distribution over action bins.

        Args:
            action_logits: (B, n_action_bins) logits for one action dimension

        Returns:
            actions: (B, horizon, action_dim) continuous actions
            log_probs: (B, action_dim) log probabilities of sampled actions
        """
        B = action_logits.shape[0]
        # For each action dimension, sample from the same logits independently
        # (the VLM produces a single set of logits; we sample action_dim times)
        all_actions = []
        all_log_probs = []
        for _ in range(self.action_dim):
            dist = torch.distributions.Categorical(logits=action_logits)
            bin_idx = dist.sample()  # (B,)
            log_prob = dist.log_prob(bin_idx)  # (B,)

            # Convert bin index to continuous action via bin centers
            bin_centers = torch.from_numpy(
                self.action_tokenizer.bin_centers
            ).to(action_logits.device, dtype=torch.float32)
            continuous = bin_centers[bin_idx]  # (B,)

            all_actions.append(continuous)
            all_log_probs.append(log_prob)

        # Stack: (B, action_dim)
        action = torch.stack(all_actions, dim=-1)
        log_probs = torch.stack(all_log_probs, dim=-1)

        # Expand to (B, horizon, action_dim) by repeating the same action
        action = action.unsqueeze(1).expand(-1, self.horizon, -1)

        return action, log_probs

    @torch.inference_mode()
    def _infer(
        self, obs: torch.Tensor, task_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference: VLM forward -> sample action -> compute Q.

        Returns:
            state: (B, state_dim)
            action: (B, horizon, action_dim)
            log_probs: (B, action_dim)
            q: (B,) scalar Q-values
        """
        state, action_logits = self._vlm_forward(obs, task_prompts)
        action, log_probs = self._sample_action(action_logits)
        q = self._compute_q(state, action)
        return state, action, log_probs, q

    def _compute_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute scalar Q-value for a (state, action) pair."""
        q_dict = self.value_head(state, action)
        q = q_dict["output"]
        return self.hl_gauss_loss(q).view(-1) if self.num_bins > 1 else q.view(-1)

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
        action_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Policy gradient loss weighted by Q-value advantage.

        Sample action from categorical policy, compute Q(s, a),
        and use -log_prob * advantage as the loss.
        """
        action, log_probs = self._sample_action(action_logits)

        # Advantage from dueling critic: A(s, a) = Q(s, a) - V(s)
        advantage_dict = self.value_head.get_advantage(state.detach(), action.detach())
        advantage = advantage_dict["output"]
        if self.num_bins > 1:
            advantage = self.hl_gauss_loss(advantage)
        advantage = advantage.view(-1).detach()

        # Policy gradient: -log_prob * advantage
        # log_probs: (B, action_dim), sum over action dims
        total_log_prob = log_probs.sum(dim=-1)  # (B,)
        actor_loss = -(total_log_prob * advantage).mean()

        # Entropy bonus for exploration
        dist = torch.distributions.Categorical(logits=action_logits)
        entropy = dist.entropy().mean()

        total_loss = actor_loss - 0.01 * entropy

        activations_dict = {"critic": advantage_dict["activation"]}

        info_dict = {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.item(),
            "advantage": advantage.mean().item(),
        }

        return total_loss, activations_dict, info_dict

    def _state_for_predictor(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape and project state for StatePredictionHead context."""
        B = state.shape[0]
        x = state.view(B, self._target_seq_len, -1)
        return self.state_to_predictor_proj(x)

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
