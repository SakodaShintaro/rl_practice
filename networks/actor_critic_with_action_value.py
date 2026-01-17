import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss

from networks.backbone import SpatialTemporalEncoder, TemporalOnlyEncoder
from networks.image_processor import ImageProcessor
from networks.policy_head import DiffusionPolicy
from networks.prediction_head import StatePredictionHead
from networks.reward_processor import RewardProcessor
from networks.value_head import ActionValueHead
from networks.vlm_backbone import MMMambaEncoder, QwenVLEncoder


class Network(nn.Module):
    def __init__(
        self, observation_space_shape: tuple[int], action_dim: int, args: argparse.Namespace
    ) -> None:
        super(Network, self).__init__()
        self.gamma = args.gamma
        self.num_bins = args.num_bins
        self.sparsity = args.sparsity
        self.seq_len = args.seq_len
        self.dacer_loss_weight = args.dacer_loss_weight
        self.critic_loss_weight = args.critic_loss_weight

        self.action_dim = action_dim
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
                action_dim=action_dim,
                temporal_model_type="transformer",
                use_image_only=True,
            )
        elif args.encoder == "temporal_only":
            self.encoder = TemporalOnlyEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=self.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=action_dim,
                temporal_model_type="transformer",
                use_image_only=False,
            )
        elif args.encoder == "qwenvl":
            self.encoder = QwenVLEncoder(
                output_text=False,
                use_quantization=args.use_quantization,
                use_lora=args.use_lora,
                use_pixel_values=args.use_pixel_values,
                target_layer_idx=args.target_layer_idx,
                seq_len=args.seq_len,
            )
        elif args.encoder == "mmmamba":
            self.encoder = MMMambaEncoder()
        else:
            raise ValueError(f"Unknown encoder: {args.encoder=}")

        self.policy_head = DiffusionPolicy(
            state_dim=self.encoder.output_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            denoising_time=args.denoising_time,
            sparsity=args.sparsity,
        )
        self.value_head = ActionValueHead(
            in_channels=self.encoder.output_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=self.num_bins,
            sparsity=args.sparsity,
        )
        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=action_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
        )

        self.detach_actor = args.detach_actor
        self.detach_critic = args.detach_critic
        self.detach_predictor = args.detach_predictor
        self.disable_state_predictor = args.disable_state_predictor

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-args.value_range,
                max_value=+args.value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def forward(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W') - pre-encoded observations
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,  # SpatialTemporal: (B, space_len, state_size, n_layer); TemporalOnly: (B, state_size, n_layer)
        action: torch.Tensor | None,  # (B, action_dim) or None
    ) -> dict:
        """Forward pass compatible with actor_critic_with_state_value interface.

        This method encodes the sequence and returns action and action-value.
        """
        x, rnn_state, action_text = self.encoder(
            s_seq, obs_z_seq, a_seq, r_seq, rnn_state
        )  # (B, hidden_dim)

        # Get action from policy_head
        if action is None:
            action, a_logp = self.policy_head.get_action(x)
        else:
            _, a_logp = self.policy_head.get_action(x)

        # Get action-value from value_head
        q_dict = self.value_head(x, action)
        q_value = q_dict["output"]  # (B, 1) or (B, num_bins)

        return {
            "action": action,  # (B, action_dim)
            "a_logp": a_logp,  # (B, 1)
            "value": q_value,  # (B, 1) or (B, num_bins)
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (B, ...)
        }

    def infer(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W')
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,
    ) -> dict:
        return self.forward(s_seq, obs_z_seq, a_seq, r_seq, rnn_state, None)

    def compute_loss(self, data, target_value) -> tuple[torch.Tensor, dict, dict]:
        obs_curr = data.observations[:, :-1]
        obs_z_curr = data.obs_z[:, :-1]
        actions_curr = data.actions[:, :-1]
        rewards_curr = data.rewards[:, :-1]
        rnn_state_curr = data.rnn_state[:, :-1]  # (B, T-1, ...)
        rnn_state_curr = rnn_state_curr[:, 0]  # (B, ...)

        state_curr, _, _ = self.encoder.forward(
            obs_curr, obs_z_curr, actions_curr, rewards_curr, rnn_state_curr
        )  # (B, state_dim)

        action_curr = data.actions[:, -1]  # (B, action_dim)

        critic_loss, critic_activations, critic_info = self._compute_critic_loss(
            state_curr, action_curr, target_value
        )
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(state_curr)
        seq_loss, seq_activations, seq_info = self._compute_sequence_loss(data, state_curr)

        total_loss = self.critic_loss_weight * critic_loss + actor_loss + seq_loss

        activations_dict = {
            "state": state_curr,
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

    @torch.no_grad()
    def compute_target_value(self, data) -> torch.Tensor:
        obs_next = data.observations[:, 1:]
        obs_z_next = data.obs_z[:, 1:]
        actions_next = data.actions[:, 1:]
        rewards_next = data.rewards[:, 1:]
        rnn_state_next = data.rnn_state[:, 1:]  # (B, T-1, ...)
        rnn_state_next = rnn_state_next[:, 0]  # (B, ...)
        state_next, _, _ = self.encoder.forward(
            obs_next, obs_z_next, actions_next, rewards_next, rnn_state_next
        )
        next_state_actions, _ = self.policy_head.get_action(state_next)
        next_critic_output_dict = self.value_head(state_next, next_state_actions)
        next_critic_value = next_critic_output_dict["output"]
        if self.num_bins > 1:
            next_critic_value = self.hl_gauss_loss(next_critic_value).view(-1)
        else:
            next_critic_value = next_critic_value.view(-1)
        curr_reward = data.rewards[:, -1].flatten()
        curr_continue = 1 - data.dones[:, -1].flatten()
        target_value = curr_reward + curr_continue * self.gamma * next_critic_value
        return target_value

    ####################
    # Internal methods #
    ####################

    def _compute_critic_loss(self, curr_state, curr_action, target_value):
        if self.detach_critic:
            curr_state = curr_state.detach()

        curr_critic_output_dict = self.value_head(curr_state, curr_action)

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

    def _compute_actor_loss(self, state_curr):
        if self.detach_actor:
            state_curr = state_curr.detach()
        pi, log_pi = self.policy_head.get_action(state_curr)

        for param in self.value_head.parameters():
            param.requires_grad_(False)

        critic_pi_output_dict = self.value_head(state_curr, pi)
        critic_pi = critic_pi_output_dict["output"]
        if self.num_bins > 1:
            critic_pi = self.hl_gauss_loss(critic_pi).unsqueeze(-1)
        else:
            critic_pi = critic_pi.unsqueeze(-1)
        actor_loss = -critic_pi.mean()

        for param in self.value_head.parameters():
            param.requires_grad_(True)

        # DACER2 loss (https://arxiv.org/abs/2505.23426)
        actions = pi.clone().detach()
        actions.requires_grad = True
        eps = 1e-4
        device = pi.device
        batch_size = pi.shape[0]
        t = (torch.rand((batch_size, 1), device=device)) * (1 - eps) + eps
        c = 0.4
        d = -1.8
        w_t = torch.exp(c * t + d)

        def calc_target(q_network, actions):
            q_output_dict = q_network(state_curr, actions)
            q_values = q_output_dict["output"]
            if self.num_bins > 1:
                q_values = self.hl_gauss_loss(q_values).unsqueeze(-1)
            else:
                q_values = q_values.unsqueeze(-1)
            q_grad = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=actions,
                create_graph=True,
            )[0]
            with torch.no_grad():
                target = (1 - t) / t * q_grad + 1 / t * actions
                target /= target.norm(dim=1, keepdim=True) + 1e-8
                return w_t * target

        target = calc_target(self.value_head, actions)
        noise = torch.randn_like(actions)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        actor_output_dict = self.policy_head.forward(a_t, t.squeeze(1), state_curr)
        v = actor_output_dict["output"]
        dacer_loss = F.mse_loss(v, target)

        # Combine actor losses
        total_actor_loss = actor_loss + dacer_loss * self.dacer_loss_weight

        activations_dict = {
            "actor": actor_output_dict["activation"],
            "critic": critic_pi_output_dict["activation"],
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "log_pi": log_pi.mean().item(),
        }

        return total_actor_loss, activations_dict, info_dict

    def _compute_sequence_loss(self, data, state_curr):
        if self.disable_state_predictor:
            # state_predictorを無効化する場合はダミー損失を返す
            dummy_loss = torch.tensor(0.0, device=state_curr.device, requires_grad=True)
            # state_currと同じ形状のダミーアクティベーションを返す
            activations_dict = {"state_predictor": state_curr}
            info_dict = {"seq_loss": 0.0}
            return dummy_loss, activations_dict, info_dict

        if self.detach_predictor:
            state_curr = state_curr.detach()

        # 最後のactionを取得 (actions[:, -1]がcurrent_stateに対応するaction)
        action_curr = data.actions[:, -1]  # (B, action_dim)

        # 次のstateをencodeする
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
        state_curr = state_curr.view(B, -1, C)

        # Predict velocity for state
        pred_dict = self.prediction_head.state_predictor.forward(xt, t, state_curr, action_curr)
        pred_vt = pred_dict["output"]  # (B, H*W, C)

        # Flow Matching loss
        vt = x1 - x0
        pred_loss = F.mse_loss(pred_vt, vt)

        activations_dict = {"state_predictor": pred_dict["activation"]}

        info_dict = {"seq_loss": pred_loss.item()}

        return pred_loss, activations_dict, info_dict

    @torch.inference_mode()
    def predict_next_state(self, state_curr, action_curr) -> tuple[np.ndarray, float]:
        return self.prediction_head.predict_next_state(
            state_curr,
            action_curr,
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )
