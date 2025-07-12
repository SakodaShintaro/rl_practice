import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss
from torch import optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.backbone import AE, SmolVLAEncoder
from networks.diffusion_policy import DiffusionPolicy
from networks.sac_tanh_policy_and_q import SacQ
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer
from sequence_modeling import SequenceModelingHelper, SequenceModelingModule


class Network(nn.Module):
    def __init__(
        self,
        sparsity: float,
        action_dim: int,
        seq_len: int,
        args,
        enable_sequence_modeling: bool,
    ):
        super(Network, self).__init__()
        num_bins = 51
        self.gamma = 0.99
        self.sparsity = sparsity

        self.action_dim = action_dim
        self.cnn_dim = 4 * 12 * 12  # 576
        self.reward_dim = 32
        self.token_dim = self.cnn_dim + self.reward_dim  # 608

        if args.image_encoder == "ae":
            self.encoder_image = AE()
        elif args.image_encoder == "smolvla":
            self.encoder_image = SmolVLAEncoder()
        else:
            raise ValueError(f"Unknown image encoder: {args.image_encoder}")
        self.actor = DiffusionPolicy(
            state_dim=self.cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.qf1 = SacQ(
            in_channels=self.cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
        )

        # Sequence modeling components (optional)
        if enable_sequence_modeling:
            self.sequence_model = SequenceModelingModule(self.cnn_dim, action_dim, seq_len, args)
        else:
            self.sequence_model = None

        self.hl_gauss_loss = HLGaussLoss(
            min_value=-30,
            max_value=+30,
            num_bins=num_bins,
            clamp_to_range=True,
        )

    def compute_critic_loss(self, data, state_curr):
        with torch.no_grad():
            state_next = self.encoder_image.encode(data.observations[:, -1])
            next_state_actions, _, _ = self.actor.get_action(state_next)
            qf1_next_output_dict = self.qf1(state_next, next_state_actions)
            qf1_next_target = qf1_next_output_dict["output"]
            qf1_next_target = self.hl_gauss_loss(qf1_next_target).unsqueeze(-1)
            min_q = qf1_next_target
            min_qf_next_target = min_q.view(-1)
            curr_reward = data.rewards[:, -2].flatten()
            curr_continue = 1 - data.dones[:, -2].flatten()
            next_q_value = curr_reward + curr_continue * self.gamma * min_qf_next_target

        qf1_output_dict = self.qf1(state_curr, data.actions[:, -2])

        qf1_a_values = qf1_output_dict["output"]

        qf1_loss = self.hl_gauss_loss(qf1_a_values, next_q_value)
        qf_loss = qf1_loss

        activations_dict = {}

        info_dict = {
            "qf1_loss": qf1_loss.item(),
            "qf1_values": qf1_a_values.mean().item(),
            "min_qf_next_target": min_qf_next_target.mean().item(),
            "next_q_value": next_q_value.mean().item(),
        }

        return qf_loss, activations_dict, info_dict

    def compute_actor_loss(self, state_curr):
        pi, log_pi, _ = self.actor.get_action(state_curr)

        for param in self.qf1.parameters():
            param.requires_grad_(False)

        qf1_pi_output_dict = self.qf1(state_curr, pi)
        qf1_pi = qf1_pi_output_dict["output"]
        qf1_pi = self.hl_gauss_loss(qf1_pi).unsqueeze(-1)
        min_qf_pi = qf1_pi
        actor_loss = -min_qf_pi.mean()

        for param in self.qf1.parameters():
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
            q_values = self.hl_gauss_loss(q_values).unsqueeze(-1)
            q_grad = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=actions,
                create_graph=True,
            )[0]
            with torch.no_grad():
                target = (1 - t) / t * q_grad + 1 / t * actions
                target /= target.norm(dim=1, keepdim=True) + 1e-8
                return w_t * target

        target1 = calc_target(self.qf1, actions)
        target = target1
        noise = torch.randn_like(actions)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        actor_output_dict = self.actor.forward(a_t, t.squeeze(1), state_curr)
        v = actor_output_dict["output"]
        dacer_loss = F.mse_loss(v, target)

        # Combine actor losses
        total_actor_loss = actor_loss + dacer_loss * 0.05

        activations_dict = {
            "actor": actor_output_dict["activation"],
            "qf1": qf1_pi_output_dict["activation"],
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "log_pi": log_pi.mean().item(),
        }

        return total_actor_loss, activations_dict, info_dict

    def compute_sequence_loss(self, data):
        if self.sequence_model is None:
            # Return zero loss when sequence modeling is disabled
            return (
                torch.tensor(0.0, device=data.observations.device),
                {},
                {},
            )

        return self.sequence_model.compute_sequence_loss(data, self.encoder_image)


class SacAgent:
    def __init__(self, args, observation_space, action_space) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seq_len = 2
        if args.enable_sequence_modeling:
            assert seq_len >= 2, "seq_len must be >= 2 for sequence modeling"

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.enable_sequence_modeling = args.enable_sequence_modeling
        self.learning_starts = args.learning_starts
        self.action_noise = args.action_noise
        self.batch_size = args.batch_size
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training

        self.network = Network(
            sparsity=args.sparsity,
            action_dim=self.action_dim,
            seq_len=seq_len,
            args=args,
            enable_sequence_modeling=args.enable_sequence_modeling,
        ).to(self.device)
        lr = 1e-4
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-5)

        self.rb = ReplayBuffer(
            args.buffer_size,
            seq_len,
            observation_space.shape,
            action_space.shape,
            self.device,
        )

        # Initialize gradient norm targets
        self.monitoring_targets = {
            "total": self.network,
            "actor": self.network.actor,
            "qf1": self.network.qf1,
        }

        # Initialize weight projection if enabled
        self.weight_projection_norms = {}
        if args.use_weight_projection:
            self.weight_projection_norms["actor"] = get_initial_norms(self.network.actor)
            self.weight_projection_norms["qf1"] = get_initial_norms(self.network.qf1)

        if args.enable_sequence_modeling:
            self.monitoring_targets["sequence_model"] = self.network.sequence_model
            self.weight_projection_norms["sequence_model"] = get_initial_norms(
                self.network.sequence_model
            )
            self.sequence_helper = SequenceModelingHelper(seq_len, self.device)

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "qf1": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
        }

    def initialize_for_episode(self):
        # Initialize sequence modeling lists if enabled
        if self.enable_sequence_modeling:
            self.sequence_helper.initialize_lists(self.action_dim)

    @torch.inference_mode()
    def select_action(self, global_step, obs):
        info_dict = {}

        obs_tensor = torch.Tensor(obs).to(self.device).unsqueeze(0)
        if global_step < self.learning_starts:
            action = self.action_space.sample()
        else:
            output_enc = self.network.encoder_image.encode(obs_tensor)
            action, selected_log_pi, _ = self.network.actor.get_action(output_enc)
            action = action[0].detach().cpu().numpy()
            action = action * self.action_scale + self.action_bias

            action_noise = self.action_space.sample()
            c = self.action_noise
            action = (1 - c) * action + c * action_noise
            action = np.clip(action, self.action_low, self.action_high)
            info_dict["selected_log_pi"] = selected_log_pi[0].item()

        # Update sequence modeling lists
        # if self.enable_sequence_modeling:
        #     self.sequence_helper.update_lists(obs_tensor, reward, action)
        #     curr_obs_float, pred_obs_float = self.sequence_helper.predict_next_state(
        #         action, next_obs, self.network
        #     )

        return action, info_dict

    def process_env_feedback(
        self, global_step, obs, action, reward, termination, truncation
    ) -> dict:
        info_dict = {}

        reward /= 10.0

        self.rb.add(obs, action, reward, termination or truncation)

        if global_step < self.learning_starts:
            return info_dict
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        # training
        data = self.rb.sample(self.batch_size)

        # Compute all losses using refactored methods
        state_curr = self.network.encoder_image.encode(data.observations[:, -2])

        qf_loss, qf_activations, qf_info = self.network.compute_critic_loss(data, state_curr)
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(state_curr)
        seq_loss, seq_activations, seq_info = self.network.compute_sequence_loss(data)

        # Combine all activations for feature_dict
        feature_dict = {
            "state": state_curr,
            **actor_activations,
            **qf_activations,
            **seq_activations,
        }

        # optimize the model
        loss = actor_loss + qf_loss + seq_loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        # Compute gradient and parameter norms
        grad_metrics = {
            key: compute_gradient_norm(model) for key, model in self.monitoring_targets.items()
        }
        param_metrics = {
            key: compute_parameter_norm(model) for key, model in self.monitoring_targets.items()
        }
        activation_norms = {
            key: value.norm(dim=1).mean().item() for key, value in feature_dict.items()
        }

        self.optimizer.step()

        # Apply weight projection after optimizer step
        if self.use_weight_projection:
            weight_project(self.network.actor, self.weight_projection_norms["actor"])
            weight_project(self.network.qf1, self.weight_projection_norms["qf1"])

            if self.enable_sequence_modeling:
                weight_project(
                    self.network.sequence_model, self.weight_projection_norms["sequence_model"]
                )

        # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
        if self.apply_masks_during_training:
            apply_masks_during_training(self.network.actor)
            apply_masks_during_training(self.network.qf1)

            if self.enable_sequence_modeling:
                apply_masks_during_training(self.network.sequence_model)

        # Add loss information from compute methods
        for key, value in qf_info.items():
            info_dict[f"losses/{key}"] = value

        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        for key, value in seq_info.items():
            info_dict[f"losses/{key}"] = value

        info_dict["losses/qf_loss"] = qf_loss

        # Add gradient norm metrics
        for key, value in grad_metrics.items():
            info_dict[f"gradients/{key}"] = value

        # Add parameter norm metrics
        for key, value in param_metrics.items():
            info_dict[f"parameters/{key}"] = value

        # Add activation norms
        for key, value in activation_norms.items():
            info_dict[f"activation_norms/{key}"] = value

        # Trigger statistical metrics computation
        for feature_name, feature in feature_dict.items():
            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        return info_dict
