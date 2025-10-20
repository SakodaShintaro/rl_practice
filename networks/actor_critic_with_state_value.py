import argparse

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from networks.backbone import (
    SpatialTemporalEncoder,
    TemporalOnlyEncoder,
)
from networks.value_head import StateValueHead


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

        if args.encoder == "spatial_temporal":
            self.encoder = SpatialTemporalEncoder(
                observation_space_shape,
                seq_len=args.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=args.temporal_model_type,
                image_processor_type="ae",
                freeze_image_processor=True,
                use_image_only=True,
            )
        elif args.encoder == "temporal_only":
            self.encoder = TemporalOnlyEncoder(
                observation_space_shape,
                seq_len=args.seq_len,
                n_layer=args.encoder_block_num,
                action_dim=self.action_dim,
                temporal_model_type=args.temporal_model_type,
                image_processor_type="simple_cnn",
                freeze_image_processor=False,
                use_image_only=True,
            )
        else:
            raise ValueError(f"Unknown encoder: {args.encoder=}")

        hidden_dim = self.encoder.output_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = StateValueHead(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            block_num=1,
            num_bins=args.num_bins,
            sparsity=0.0,
        )
        self.policy_enc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.policy_type = args.policy_type
        if self.policy_type == "Beta":
            self.alpha_head = nn.Linear(hidden_dim, self.action_dim)
            self.beta_head = nn.Linear(hidden_dim, self.action_dim)
        elif self.policy_type == "Categorical":
            self.logits_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            raise ValueError("Invalid policy type")
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
        r_seq: torch.Tensor,  # (B, T, 1)
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        a_seq: torch.Tensor,  # (B, T, action_dim)
        rnn_state: torch.Tensor,  # (1, B, hidden_size)
        action: torch.Tensor | None = None,  # (B, action_dim) or None
    ) -> tuple:
        x, rnn_state = self.encoder(s_seq, a_seq, r_seq, rnn_state)  # (B, hidden_dim)
        x = self.linear(x)  # (B, hidden_dim)

        value_dict = self.value_head(x)

        policy_x = self.policy_enc(x)

        if self.policy_type == "Beta":
            alpha = self.alpha_head(policy_x).exp() + 1
            beta = self.beta_head(policy_x).exp() + 1

            dist = Beta(alpha, beta)
            if action is None:
                action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1, keepdim=True)
        elif self.policy_type == "Categorical":
            logits = self.logits_head(policy_x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
                a_logp = dist.log_prob(action).unsqueeze(1)
                action = F.one_hot(action, num_classes=self.action_dim).float()
            else:
                a_logp = dist.log_prob(action.argmax(dim=1)).unsqueeze(1)

        return {
            "action": action,  # (B, action_dim)
            "a_logp": a_logp,  # (B, 1)
            "entropy": dist.entropy().unsqueeze(1),  # (B, 1)
            "value": value_dict["output"],  # (B, 1)
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (1, B, hidden_size)
        }

    def compute_loss(self, data, curr_target_v, curr_adv) -> tuple[torch.Tensor, dict, dict]:
        net_out_dict = self.forward(
            data.rewards[:, :-1],
            data.observations[:, :-1],
            data.actions[:, :-1],
            data.rnn_state[:, 0].permute(1, 0, 2).contiguous(),
            action=data.actions[:, -1],
        )
        a_logp = net_out_dict["a_logp"]
        entropy = net_out_dict["entropy"]
        value = net_out_dict["value"]
        ratio = torch.exp(a_logp - data.log_probs[:, -1])

        surr1 = ratio * curr_adv
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
            * curr_adv
        )
        action_loss = -torch.min(surr1, surr2).mean()

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

        loss = action_loss + 0.25 * value_loss - 0.02 * entropy.mean()

        activations_dict = {}

        info_dict = {
            "actor_loss": action_loss.item(),
            "critic_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

        return loss, activations_dict, info_dict
