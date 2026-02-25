# SPDX-License-Identifier: MIT
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss

from rl_practice.networks.backbone import SpatialTemporalEncoder, TemporalOnlyEncoder
from rl_practice.networks.diffusion_utils import compute_actor_loss_with_dacer
from rl_practice.networks.image_processor import ImageProcessor
from rl_practice.networks.policy_head import BetaPolicy, CFGDiffusionPolicy, DiffusionPolicy
from rl_practice.networks.prediction_head import StatePredictionHead
from rl_practice.networks.reward_processor import RewardProcessor
from rl_practice.networks.value_head import ActionValueHead
from rl_practice.networks.vlm_backbone import QwenVLEncoder


class ActorCriticWithActionValue(nn.Module):
    def __init__(
        self,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.gamma = args.gamma
        self.num_bins = args.num_bins
        self.sparsity = args.sparsity
        self.seq_len = args.seq_len
        self.dacer_loss_weight = args.dacer_loss_weight
        self.critic_loss_weight = args.critic_loss_weight

        self.action_dim = action_space_shape[0]
        self.predictor_step_num = args.predictor_step_num
        self.observation_space_shape = observation_space_shape

        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        if args.encoder == "spatial_temporal":
            self.encoder = SpatialTemporalEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=self.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=args.temporal_model_type,
                use_image_only=True,
            )
        elif args.encoder == "temporal_only":
            self.encoder = TemporalOnlyEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=self.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=args.temporal_model_type,
                use_image_only=False,
            )
        elif args.encoder == "qwenvl":
            self.encoder = QwenVLEncoder(
                use_quantization=args.use_quantization,
                use_lora=args.use_lora,
                target_layer_idx=args.target_layer_idx,
                seq_len=args.seq_len,
                task_prompt=args.encoder_task_prompt,
            )
        else:
            raise ValueError(f"Unknown encoder: {args.encoder=}")

        self.horizon = args.horizon
        self.policy_type = args.policy_type
        if self.policy_type == "diffusion":
            self.policy_head = DiffusionPolicy(
                state_dim=self.encoder.output_dim,
                action_dim=self.action_dim,
                hidden_dim=args.actor_hidden_dim,
                block_num=args.actor_block_num,
                denoising_time=args.denoising_time,
                sparsity=args.sparsity,
                horizon=args.horizon,
                denoising_steps=args.denoising_steps,
            )
        elif self.policy_type == "beta":
            self.policy_head = BetaPolicy(
                hidden_dim=self.encoder.output_dim,
                action_dim=self.action_dim,
                horizon=args.horizon,
            )
        elif self.policy_type == "cfgrl":
            self.policy_head = CFGDiffusionPolicy(
                state_dim=self.encoder.output_dim,
                action_dim=self.action_dim,
                hidden_dim=args.actor_hidden_dim,
                block_num=args.actor_block_num,
                denoising_time=args.denoising_time,
                sparsity=args.sparsity,
                cfgrl_beta=1.5,
                horizon=args.horizon,
                denoising_steps=args.denoising_steps,
            )
        self.value_head = ActionValueHead(
            in_channels=self.encoder.output_dim,
            action_dim=self.action_dim,
            horizon=args.horizon,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=self.num_bins,
            sparsity=args.sparsity,
        )
        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
        )

        self.detach_actor = args.detach_actor
        self.detach_critic = args.detach_critic
        self.detach_predictor = args.detach_predictor
        # CFGRL parameters
        self.condition_drop_prob = 0.1
        # Disable state prediction when using VLM encoder
        is_vlm_encoder = args.encoder == "qwenvl"
        self.disable_state_predictor = args.disable_state_predictor or is_vlm_encoder

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-args.value_range,
                max_value=+args.value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    @torch.inference_mode()
    def infer(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W')
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,
    ) -> dict:
        assert s_seq.shape[0] == 1, "Batch size must be 1 for inference"

        x, rnn_state = self.encoder(s_seq, obs_z_seq, a_seq, r_seq, rnn_state)  # (B, hidden_dim)

        # Get action chunk from policy_head
        action, a_logp = self.policy_head.get_action(x)  # (B, horizon, action_dim)

        # Get action-value from value_head
        q_dict = self.value_head(x, action)
        q_value = q_dict["output"]  # (B, 1) or (B, num_bins)
        q_value = q_value.item() if self.num_bins == 1 else self.hl_gauss_loss(q_value).item()

        # Get predicted next state
        next_image, next_reward = self.prediction_head.predict_next_state(
            x,
            action[:, 0],  # use first action in chunk for prediction
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        return {
            "action": action,  # (B, horizon, action_dim)
            "a_logp": a_logp,  # (B, 1)
            "value": q_value,  # float
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (B, ...)
            "next_image": next_image,  # predicted next image
            "next_reward": next_reward,  # predicted next reward
            "action_token_ids": [],  # empty for non-VLM networks
            "parse_success": True,  # always True for non-VLM networks
        }

    def compute_loss(self, data) -> tuple[torch.Tensor, dict, dict]:
        # compute target value
        target_value = self._compute_target_value(data)["target_value"]  # (B,)

        # Use seq_len frames (excluding last horizon frames)
        curr_obs = data.observations[:, : -self.horizon]
        curr_obs_z = data.obs_z[:, : -self.horizon]
        curr_actions = data.actions[:, : -self.horizon]
        curr_rewards = data.rewards[:, : -self.horizon]
        curr_rnn_state = data.rnn_state[:, 0]  # (B, ...)

        curr_state, _ = self.encoder.forward(
            curr_obs, curr_obs_z, curr_actions, curr_rewards, curr_rnn_state
        )  # (B, state_dim)

        # Action chunk: (B, horizon, action_dim)
        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            curr_state, action_chunk, target_value
        )
        if self.policy_type == "diffusion":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss(curr_state)
        elif self.policy_type == "beta":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss_pg(curr_state)
        elif self.policy_type == "cfgrl":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss_cfgrl(
                curr_state, action_chunk
            )
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, curr_state)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        activations_dict = {
            "state": curr_state,
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }

        info_dict = {
            **critic_info,
            **actor_info,
            **seq_info,
        }

        return total_loss, activations_dict, info_dict

    def infer_and_compute_loss(self, data) -> tuple[dict, torch.Tensor, dict, dict]:
        """Combined inference and loss computation."""
        target_dict = self._compute_target_value(data)
        target_value = target_dict["target_value"]

        prev_obs = data.observations[:, : -self.horizon]
        prev_obs_z = data.obs_z[:, : -self.horizon]
        prev_actions = data.actions[:, : -self.horizon]
        prev_rewards = data.rewards[:, : -self.horizon]
        prev_rnn_state = data.rnn_state[:, 0]

        prev_state, _ = self.encoder.forward(
            prev_obs, prev_obs_z, prev_actions, prev_rewards, prev_rnn_state
        )

        action_chunk = data.actions[:, -self.horizon :]

        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            prev_state, action_chunk, target_value
        )
        if self.policy_type == "diffusion":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss(prev_state)
        elif self.policy_type == "beta":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss_pg(prev_state)
        elif self.policy_type == "cfgrl":
            actor_loss, actor_activations, actor_info = self._compute_actor_loss_cfgrl(
                prev_state, action_chunk
            )
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(
            data, target_dict["next_state"]
        )

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        next_image, next_reward = self.prediction_head.predict_next_state(
            target_dict["next_state"],
            target_dict["action"][:, 0],
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )

        infer_dict = {
            "action": target_dict["action"],
            "value": target_dict["q_value"].item(),
            "rnn_state": target_dict["rnn_state"],
            "next_image": next_image,
            "next_reward": next_reward,
        }

        activations_dict = {
            "state": target_dict["next_state"],
            **critic_activations,
            **actor_activations,
            **seq_activations,
        }

        info_dict = {
            **critic_info,
            **actor_info,
            **seq_info,
        }

        return infer_dict, total_loss, activations_dict, info_dict

    ####################
    # Internal methods #
    ####################

    @torch.no_grad()
    def _compute_target_value(self, data) -> dict:
        next_obs = data.observations[:, self.horizon :]
        next_obs_z = data.obs_z[:, self.horizon :]
        next_actions = data.actions[:, self.horizon :]
        next_rewards = data.rewards[:, self.horizon :]
        next_rnn_state = data.rnn_state[:, self.horizon]

        next_state, rnn_state_out = self.encoder.forward(
            next_obs, next_obs_z, next_actions, next_rewards, next_rnn_state
        )
        action, _ = self.policy_head.get_action(next_state)
        next_critic_output_dict = self.value_head(next_state, action)
        next_critic_value = next_critic_output_dict["output"]
        if self.num_bins > 1:
            next_critic_value = self.hl_gauss_loss(next_critic_value).view(-1)
        else:
            next_critic_value = next_critic_value.view(-1)

        chunk_rewards = data.rewards[:, -self.horizon :]
        chunk_dones = data.dones[:, -self.horizon :]
        batch_size = chunk_rewards.size(0)
        device = chunk_rewards.device
        discounted_reward = torch.zeros(batch_size, device=device)
        gamma_power = 1.0
        continuing = torch.ones(batch_size, device=device)

        for i in range(self.horizon):
            discounted_reward += continuing * gamma_power * chunk_rewards[:, i].flatten()
            gamma_power *= self.gamma
            continuing *= 1 - chunk_dones[:, i].flatten()

        target_value = discounted_reward + continuing * gamma_power * next_critic_value

        return {
            "target_value": target_value,
            "action": action,
            "rnn_state": rnn_state_out,
            "q_value": next_critic_value,
            "next_state": next_state,
        }

    def _compute_critic_loss(self, curr_state, action_chunk, target_value):
        """
        Args:
            curr_state: (B, state_dim)
            action_chunk: (B, horizon, action_dim)
            target_value: (B,)
        """
        if self.detach_critic:
            curr_state = curr_state.detach()

        curr_critic_output_dict = self.value_head(curr_state, action_chunk)

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

    def _compute_actor_loss(self, curr_state):
        """
        Args:
            curr_state: (B, state_dim)
        """
        if self.detach_actor:
            curr_state = curr_state.detach()
        action, log_pi = self.policy_head.get_action(curr_state)  # (B, horizon, action_dim)

        bs = action.shape[0]
        actor_activation = {}

        def predict_velocity_fn(a_t, t):
            a_flat = a_t.view(bs, -1)
            result = self.policy_head.forward(a_flat, t, curr_state)
            actor_activation["value"] = result["activation"]
            return result["output"].view(bs, self.horizon, self.action_dim)

        total_actor_loss, advantage_dict, info_dict = compute_actor_loss_with_dacer(
            curr_state,
            action,
            self.value_head,
            getattr(self, "hl_gauss_loss", None),
            self.num_bins,
            self.dacer_loss_weight,
            predict_velocity_fn,
        )

        activations_dict = {
            "actor": actor_activation["value"],
            "critic": advantage_dict["activation"],
        }
        info_dict["log_pi"] = log_pi.mean().item()

        return total_actor_loss, activations_dict, info_dict

    def _compute_actor_loss_pg(self, curr_state):
        if self.detach_actor:
            curr_state = curr_state.detach()

        policy_output = self.policy_head.forward(curr_state, None)
        action = policy_output["action"]
        log_pi = policy_output["a_logp"]
        entropy = policy_output["entropy"]

        advantage_dict = self.value_head.get_advantage(curr_state, action)
        advantage = advantage_dict["output"]
        if self.num_bins > 1:
            advantage = self.hl_gauss_loss(advantage)
        advantage = advantage.view(-1, 1)

        actor_loss = -(log_pi * advantage.detach()).mean() - 0.02 * entropy.mean()

        activations_dict = {
            "actor": policy_output["activation"],
            "critic": advantage_dict["activation"],
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "log_pi": log_pi.mean().item(),
            "advantage": advantage.mean().item(),
            "entropy": entropy.mean().item(),
        }

        return actor_loss, activations_dict, info_dict

    def _compute_actor_loss_cfgrl(self, curr_state, action_chunk):
        """
        CFGRL/pistar06 style conditional supervised learning

        I=1 (positive) if advantage >= threshold, otherwise I=0 (negative)
        Drop condition with condition_drop_prob probability (I=2, unconditional)

        Args:
            curr_state: (B, state_dim)
            action_chunk: (B, horizon, action_dim)
        """
        if self.detach_actor:
            curr_state = curr_state.detach()

        batch_size = curr_state.shape[0]
        device = curr_state.device

        # Flatten action chunk: (B, horizon, action_dim) -> (B, horizon * action_dim)
        action_flat = action_chunk.view(batch_size, -1)

        # Calculate advantage and determine condition I
        with torch.no_grad():
            advantage_dict = self.value_head.get_advantage(curr_state, action_chunk)
            advantage = advantage_dict["output"]
            if self.num_bins > 1:
                advantage = self.hl_gauss_loss(advantage)
            advantage = advantage.view(-1)

            # I = 1 if A >= median else 0 (use median as threshold)
            threshold = advantage.median()
            condition = (advantage >= threshold).long()

            # Set to unconditional (I=2) with condition_drop_prob probability
            drop_mask = torch.rand(batch_size, device=device) < self.condition_drop_prob
            condition = torch.where(drop_mask, torch.full_like(condition, 2), condition)

        # Flow Matching for conditional policy
        eps = 1e-4
        t = torch.rand((batch_size, 1), device=device) * (1 - eps) + eps

        # Interpolation from noise to action_flat
        noise = torch.randn_like(action_flat)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * action_flat

        # Predict velocity conditionally
        actor_output_dict = self.policy_head.forward(a_t, t.squeeze(1), curr_state, condition)
        v_pred = actor_output_dict["output"]

        # Target velocity: action_flat - noise
        v_target = action_flat - noise

        # Flow Matching loss
        actor_loss = F.mse_loss(v_pred, v_target)

        activations_dict = {
            "actor": actor_output_dict["activation"],
        }

        # Calculate positive and negative ratios
        positive_ratio = (condition == 1).float().mean().item()
        negative_ratio = (condition == 0).float().mean().item()
        uncond_ratio = (condition == 2).float().mean().item()

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": 0.0,
            "log_pi": 0.0,
            "advantage": advantage.mean().item(),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "uncond_ratio": uncond_ratio,
        }

        return actor_loss, activations_dict, info_dict

    def _compute_sequence_loss(self, data, curr_state):
        if self.disable_state_predictor:
            # Return dummy loss when state_predictor is disabled
            dummy_loss = torch.tensor(0.0, device=curr_state.device, requires_grad=True)
            # Return dummy activation with same shape as state_curr
            activations_dict = {"state_predictor": curr_state}
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, activations_dict, info_dict

        if self.detach_predictor:
            curr_state = curr_state.detach()

        # Get last action (actions[:, -1] corresponds to current_state)
        curr_action = data.actions[:, -1]  # (B, action_dim)

        # Encode next state
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

        # Flow Matching for state prediction
        x0 = torch.randn_like(x1)
        shape_t = (x0.shape[0],) + (1,) * (len(x0.shape) - 1)
        t = torch.rand(shape_t, device=x1.device)

        # Sample from interpolation path for state
        xt = (1.0 - t) * x0 + t * x1

        # Convert tensors
        curr_state = curr_state.view(B, -1, C)

        # Predict velocity for state
        pred_dict = self.prediction_head.state_predictor.forward(xt, t, curr_state, curr_action)
        pred_vt = pred_dict["output"]  # (B, H*W, C)

        # Flow Matching loss
        vt = x1 - x0
        pred_loss = F.mse_loss(pred_vt, vt)

        activations_dict = {"state_predictor": pred_dict["activation"]}

        info_dict = {"seq_loss": pred_loss.item()}

        return pred_loss, activations_dict, info_dict
