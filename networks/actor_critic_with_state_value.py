import argparse

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn import functional as F

from networks.backbone import SpatialTemporalEncoder, TemporalOnlyEncoder
from networks.image_processor import ImageProcessor
from networks.policy_head import BetaPolicy, CategoricalPolicy
from networks.prediction_head import StatePredictionHead
from networks.reward_processor import RewardProcessor
from networks.value_head import StateValueHead
from networks.vlm_backbone import MMMambaEncoder, QwenVLEncoder


class Network(nn.Module):
    clip_param_policy = 0.2
    clip_param_value = 0.2

    def __init__(
        self,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.action_dim = action_space_shape[0]
        self.num_bins = args.num_bins
        self.observation_space_shape = observation_space_shape
        self.predictor_step_num = args.predictor_step_num
        self.critic_loss_weight = args.critic_loss_weight

        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=args.image_processor_type
        )
        hidden_image_dim = self.image_processor.output_shape[0]
        self.reward_processor = RewardProcessor(embed_dim=hidden_image_dim)

        if args.encoder == "spatial_temporal":
            self.encoder = SpatialTemporalEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=args.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type="transformer",
                use_image_only=True,
            )
        elif args.encoder == "temporal_only":
            self.encoder = TemporalOnlyEncoder(
                image_processor=self.image_processor,
                reward_processor=self.reward_processor,
                seq_len=args.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type="transformer",
                use_image_only=True,
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

        hidden_dim = self.encoder.output_dim

        self.value_head = StateValueHead(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            block_num=1,
            num_bins=args.num_bins,
            sparsity=0.0,
        )

        if args.policy_type == "Beta":
            self.policy_head = BetaPolicy(hidden_dim, self.action_dim)
        elif args.policy_type == "Categorical":
            self.policy_head = CategoricalPolicy(hidden_dim, self.action_dim)
        else:
            raise ValueError("Invalid policy type")

        self.prediction_head = StatePredictionHead(
            image_processor=self.image_processor,
            reward_processor=self.reward_processor,
            action_dim=self.action_dim,
            predictor_hidden_dim=args.predictor_hidden_dim,
            predictor_block_num=args.predictor_block_num,
        )

        self.disable_state_predictor = args.disable_state_predictor

        self.apply(self._init_weights)

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-args.value_range,
                max_value=+args.value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with orthogonal initialization.

        Arguments:
            module {nn.Module} -- Module to initialize
        """
        for name, param in module.named_parameters():
            if "ae." in name:
                continue
            if param.dim() != 2:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

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
    ) -> tuple:
        x, rnn_state, action_text = self.encoder(
            s_seq, obs_z_seq, a_seq, r_seq, rnn_state
        )  # (B, hidden_dim)

        value_dict = self.value_head(x)

        policy_dict = self.policy_head(x, action)

        return {
            "action": policy_dict["action"],  # (B, action_dim)
            "a_logp": policy_dict["a_logp"],  # (B, 1)
            "entropy": policy_dict["entropy"],  # (B, 1)
            "value": value_dict["output"],  # (B, 1)
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (B, ...)
        }

    def compute_loss(self, data, curr_target_v, curr_adv) -> tuple[torch.Tensor, dict, dict]:
        # Encode state
        obs_curr = data.observations[:, :-1]
        obs_z_curr = data.obs_z[:, :-1]
        actions_curr = data.actions[:, :-1]
        rewards_curr = data.rewards[:, :-1]
        rnn_state_curr = data.rnn_state[:, :-1]
        rnn_state_curr = rnn_state_curr[:, 0]

        state_curr, _, _ = self.encoder.forward(
            obs_curr, obs_z_curr, actions_curr, rewards_curr, rnn_state_curr
        )

        # Get policy output
        policy_dict = self.policy_head(state_curr, action=data.actions[:, -1])
        a_logp = policy_dict["a_logp"]
        entropy = policy_dict["entropy"]
        policy_activation = policy_dict["activation"]

        # Get value output
        value_dict = self.value_head(state_curr)
        value = value_dict["output"]
        value_activation = value_dict["activation"]

        # Compute policy loss
        ratio = torch.exp(a_logp - data.log_probs[:, -1])
        surr1 = ratio * curr_adv
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
            * curr_adv
        )
        action_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        if self.num_bins > 1:
            value_loss = self.hl_gauss_loss(value, curr_target_v.squeeze(1))
        else:
            value_clipped = torch.clamp(
                value,
                data.values[:, -1] - self.clip_param_value,
                data.values[:, -1] + self.clip_param_value,
            )
            value_loss_unclipped = F.mse_loss(value, curr_target_v)
            value_loss_clipped = F.mse_loss(value_clipped, curr_target_v)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        loss = action_loss + self.critic_loss_weight * value_loss - 0.02 * entropy.mean()

        activations_dict = {
            "state": state_curr,
            "actor": policy_activation,
            "critic": value_activation,
            "state_predictor": state_curr,
        }

        info_dict = {
            "actor_loss": action_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

        return loss, activations_dict, info_dict

    @torch.inference_mode()
    def predict_next_state(self, state_curr, action_curr):
        return self.prediction_head.predict_next_state(
            state_curr,
            action_curr,
            self.observation_space_shape,
            self.predictor_step_num,
            self.disable_state_predictor,
        )
