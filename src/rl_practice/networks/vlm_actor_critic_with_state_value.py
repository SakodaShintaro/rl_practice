# SPDX-License-Identifier: MIT
import argparse

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from .image_processor import ImageProcessor
from .value_head import SeparateCritic
from .vlm_backbone import (
    get_action_prompt,
    load_model,
    parse_action_text,
    prepare_vlm_inputs,
)


class VLMActorCriticWithStateValue(nn.Module):
    """VLM-based Actor-Critic with text action output and state value estimation.

    Actor: Generates action as text via autoregressive decoding.
    Critic: Uses the final hidden state after action token generation to compute V(s).
    """

    def __init__(
        self,
        observation_space_shape: tuple[int, ...],
        action_space_shape: tuple[int, ...],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.clip_param_policy = args.clip_param_policy
        self.clip_param_value = args.clip_param_value
        self.action_dim = action_space_shape[0]
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        self.task_prompt = get_action_prompt(self.horizon)
        self.episode_prompt = ""
        self.target_layer_idx = args.target_layer_idx
        self.max_new_tokens = args.max_new_tokens
        self.num_bins = args.num_bins
        self.value_min = -args.value_range
        self.value_max = +args.value_range
        self.gamma = args.gamma
        self.separate_critic = args.separate_critic
        self.observation_space_shape = observation_space_shape
        self.image_processor_type = args.image_processor_type
        self.critic_block_num = args.critic_block_num
        self.critic_hidden_dim = args.critic_hidden_dim

        # Load model and processor
        assert torch.cuda.is_available(), "CUDA is required for VLM training"
        device = "cuda"

        self.model, self.processor = load_model(
            args.vlm_model_id,
            use_quantization=args.use_quantization,
            use_lora=args.use_lora,
            device=device,
        )
        self.device = device

        # Enable gradient checkpointing to reduce memory usage
        self.model.gradient_checkpointing_enable()

        hidden_size = int(self.model.config.text_config.hidden_size)
        self.value_head = (
            SeparateCritic(
                observation_space_shape,
                args.image_processor_type,
                args.critic_hidden_dim,
                args.critic_block_num,
                args.num_bins,
            ).to(device)
            if self.separate_critic
            else nn.Sequential(
                nn.Linear(hidden_size, args.critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.critic_hidden_dim, self.num_bins),
            ).to(device)
        )

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=self.value_min,
                max_value=self.value_max,
                num_bins=self.num_bins,
                clamp_to_range=True,
            ).to(device)

        self.image_processor = ImageProcessor(
            observation_space_shape, args.image_processor_type
        ).to(device)
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def get_pad_token_id(self) -> int:
        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        return pad_token_id if pad_token_id is not None else eos_token_id

    def forward(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """Forward pass: generate action text and compute state value."""
        inputs = prepare_vlm_inputs(
            self.processor, s_seq, r_seq, self.task_prompt + self.episode_prompt
        )

        outputs = self.model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        hidden = hidden_states[self.target_layer_idx][:, -1, :].to(torch.float32)

        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generated = self.model.generate(
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

        log_prob = torch.tensor(0.0, device=self.device)

        action_array, parse_success = parse_action_text(action_text, self.horizon)
        # action_array: (horizon, action_dim) -> (1, horizon, action_dim)
        action_tensor = torch.from_numpy(action_array).unsqueeze(0).to(s_seq.device)
        print(
            f"{input_len=}, {len(generated_ids)=}, {action_text=}, {action_tensor=}, {parse_success=}"
        )

        value_dict = (
            self.value_head(s_seq[:, -1])
            if self.separate_critic
            else {"output": self.value_head(hidden)}
        )
        value_logits = value_dict["output"]
        value = (
            self.hl_gauss_loss(value_logits).unsqueeze(-1) if self.num_bins > 1 else value_logits
        )

        # log_prob is already a scalar (sum of token log probs)
        a_logp = log_prob.unsqueeze(0).unsqueeze(-1)
        entropy = torch.zeros((s_seq.size(0), 1), device=s_seq.device)

        return {
            "action": action_tensor,
            "a_logp": a_logp,
            "entropy": entropy,
            "value": value,
            "x": hidden,
            "rnn_state": rnn_state,
            "action_text": action_text,
            "action_token_ids": generated_ids,
            "parse_success": parse_success,
        }

    def _action_to_text(self, action: torch.Tensor) -> str:
        """Convert action tensor to canonical text representation.

        Args:
            action: (horizon, action_dim) tensor
        """
        parts = []
        for t in range(action.shape[0]):
            steer = action[t, 0].item()
            accel = action[t, 1].item()
            parts.append(f"t{t}: steer={steer:.2f}, accel={accel:.2f}")
        return "Actions: " + "; ".join(parts)

    def _compute_value_and_log_prob(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
        action_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute value and log prob in a single forward pass.

        Args:
            images: (B, T, C, H, W)
            rewards: (B, T, 1)
            action_token_ids: (B, max_token_len) - padded token IDs

        Returns:
            value: (B, 1) or (B, num_bins)
            hidden: (B, hidden_size)
            log_probs: (B,)
        """
        inputs = prepare_vlm_inputs(
            self.processor, images, rewards, self.task_prompt + self.episode_prompt
        )

        # Create attention mask for action tokens (non-padding tokens)
        pad_token_id = self.get_pad_token_id()
        target_mask = (action_token_ids != pad_token_id).float()
        action_len = action_token_ids.size(1)
        prompt_len = inputs["input_ids"].size(1)

        # Concatenate prompt + action tokens
        combined_input_ids = torch.cat([inputs["input_ids"], action_token_ids], dim=1)
        combined_attention_mask = torch.cat([inputs["attention_mask"], target_mask.long()], dim=1)

        # Single forward pass with hidden states
        outputs = self.model.forward(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )

        # Value: from prompt's last hidden state (before action tokens)
        hidden_states = outputs.hidden_states
        state_hidden = hidden_states[self.target_layer_idx][:, prompt_len - 1, :].to(torch.float32)
        value_dict = (
            self.value_head(images[:, -1])
            if self.separate_critic
            else {"output": self.value_head(state_hidden)}
        )
        value_logits = value_dict["output"]
        value = (
            self.hl_gauss_loss(value_logits).unsqueeze(-1) if self.num_bins > 1 else value_logits
        )

        # Log prob: from logits predicting action tokens
        relevant_logits = outputs.logits[:, prompt_len - 1 : prompt_len + action_len - 1, :]
        log_prob_dist = F.log_softmax(relevant_logits, dim=-1)
        token_log_probs = log_prob_dist.gather(2, action_token_ids.unsqueeze(2)).squeeze(2)
        batch_log_probs = (token_log_probs * target_mask).sum(dim=1)

        return value, state_hidden, batch_log_probs

    def compute_loss(
        self,
        data,
        curr_target_v: torch.Tensor,
        curr_adv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Compute PPO actor loss and critic loss in a single forward pass."""
        obs_curr = data.observations[:, :-1]
        rewards_curr = data.rewards[:, :-1]
        action_token_ids_curr = data.action_token_ids[:, -1]  # (B, max_token_len)
        old_log_prob = data.log_probs[:, -1].view(-1)

        # Single forward pass for value and log prob
        value, hidden, new_log_prob = self._compute_value_and_log_prob(
            obs_curr, rewards_curr, action_token_ids_curr
        )

        # PPO actor loss
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * curr_adv.view(-1)
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy
        ) * curr_adv.view(-1)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        value_logits_for_loss = (
            self.value_head(data.observations[:, -1])["output"]
            if self.separate_critic
            else self.value_head(hidden)
        )
        value_loss = (
            self.hl_gauss_loss(value_logits_for_loss, curr_target_v.view(-1))
            if self.num_bins > 1
            else F.mse_loss(value.view(-1), curr_target_v.view(-1))
        )

        total_loss = actor_loss + value_loss

        activations_dict = {
            "critic": hidden,
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "critic_loss": value_loss.item(),
        }

        return total_loss, activations_dict, info_dict

    def train_with_feedback(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
        feedback_text: str,
    ) -> dict:
        """Train VLM with feedback text using next token prediction.

        Args:
            images: (B, T, C, H, W) observation images
            rewards: (B, T, 1) rewards
            feedback_text: Target text to predict

        Returns:
            Dictionary with training metrics
        """
        inputs = prepare_vlm_inputs(self.processor, images, rewards, "")

        # Tokenize feedback text
        feedback_tokens = self.processor.tokenizer(
            feedback_text, add_special_tokens=False, return_tensors="pt"
        )
        target_ids = feedback_tokens["input_ids"].to(self.device)
        prompt_len = inputs["input_ids"].size(1)

        # Concatenate prompt + feedback tokens
        combined_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
        combined_attention_mask = torch.cat(
            [inputs["attention_mask"], feedback_tokens["attention_mask"].to(self.device)], dim=1
        )

        # Forward pass
        outputs = self.model.forward(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            return_dict=True,
        )

        # Compute next token prediction loss for feedback tokens only
        # logits[:, prompt_len-1:-1] predicts tokens at positions prompt_len:end
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        self.episode_prompt = feedback_text

        return {"feedback_loss": loss}

    @torch.no_grad()
    def compute_target_value(self, data) -> torch.Tensor:
        """Compute target value for TD learning with n-step returns."""
        batch_size = data.observations.shape[0]
        device = data.observations.device

        # For action chunking, next state is after the full horizon
        obs_next = data.observations[:, self.horizon :]
        rewards_next = data.rewards[:, self.horizon :]
        actions_next = data.actions[:, -1]  # Use current action as dummy

        # Use _compute_value_and_log_prob, ignore log_prob
        _, hidden_next, _ = self._compute_value_and_log_prob(obs_next, rewards_next, actions_next)
        value_logits_next = (
            self.value_head(obs_next[:, -1])["output"]
            if self.separate_critic
            else self.value_head(hidden_next)
        )
        next_value = (
            self.hl_gauss_loss(value_logits_next)
            if self.num_bins > 1
            else value_logits_next.view(-1)
        )

        # Accumulate discounted rewards over the horizon
        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]

        discounted_reward = torch.zeros(batch_size, device=device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=device)

        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()

        return discounted_reward + continuing * gamma_power * next_value
