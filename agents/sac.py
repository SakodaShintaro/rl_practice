import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss
from torch import optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.backbone import AE, MMMambaEncoder, SmolVLMEncoder
from networks.diffusion_policy import DiffusionPolicy
from networks.sac_tanh_policy_and_q import SacQ
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer
from sequence_modeling import SequenceModelingModule


class Network(nn.Module):
    def __init__(
        self,
        action_dim: int,
        seq_len: int,
        args,
        enable_sequence_modeling: bool,
    ):
        super(Network, self).__init__()
        self.gamma = 0.99
        self.num_bins = args.num_bins
        self.sparsity = args.sparsity

        self.action_dim = action_dim

        if args.image_encoder == "ae":
            self.encoder_image = AE()
        elif args.image_encoder == "smolvlm":
            self.encoder_image = SmolVLMEncoder()
        elif args.image_encoder == "mmmamba":
            self.encoder_image = MMMambaEncoder()
        else:
            raise ValueError(f"Unknown image encoder: {args.image_encoder}")

        self.state_dim = self.encoder_image.output_dim
        self.reward_dim = 32
        self.token_dim = self.state_dim + self.reward_dim  # 608

        self.actor = DiffusionPolicy(
            state_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.critic = SacQ(
            in_channels=self.state_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=self.num_bins,
            sparsity=args.sparsity,
        )

        # Sequence modeling components (optional)
        if enable_sequence_modeling:
            self.sequence_model = SequenceModelingModule(self.state_dim, action_dim, seq_len, args)
        else:
            self.sequence_model = None

        if self.num_bins > 1:
            value_range = 60
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-value_range,
                max_value=+value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def compute_critic_loss(self, data, state_curr):
        with torch.no_grad():
            obs_next = data.observations[:, -1]
            state_next, _ = self.encoder_image.forward(obs_next)
            next_state_actions, _ = self.actor.get_action(state_next)
            next_critic_output_dict = self.critic(state_next, next_state_actions)
            next_critic_value = next_critic_output_dict["output"]
            if self.num_bins > 1:
                next_critic_value = self.hl_gauss_loss(next_critic_value).view(-1)
            else:
                next_critic_value = next_critic_value.view(-1)
            curr_reward = data.rewards[:, -2].flatten()
            curr_continue = 1 - data.dones[:, -2].flatten()
            target_value = curr_reward + curr_continue * self.gamma * next_critic_value

        curr_critic_output_dict = self.critic(state_curr, data.actions[:, -2])

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
            "next_critic_value": next_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
        }

        return critic_loss, activations_dict, info_dict

    def compute_actor_loss(self, state_curr):
        pi, log_pi = self.actor.get_action(state_curr)

        for param in self.critic.parameters():
            param.requires_grad_(False)

        critic_pi_output_dict = self.critic(state_curr, pi)
        critic_pi = critic_pi_output_dict["output"]
        if self.num_bins > 1:
            critic_pi = self.hl_gauss_loss(critic_pi).unsqueeze(-1)
        else:
            critic_pi = critic_pi.unsqueeze(-1)
        actor_loss = -critic_pi.mean()

        for param in self.critic.parameters():
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

        target = calc_target(self.critic, actions)
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
            "critic": critic_pi_output_dict["activation"],
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

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.action_norm_penalty = args.action_norm_penalty

        self.learning_starts = args.learning_starts
        self.action_noise = args.action_noise
        self.batch_size = args.batch_size
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training

        self.network = Network(
            action_dim=self.action_dim,
            seq_len=seq_len,
            args=args,
            enable_sequence_modeling=False,
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
            "critic": self.network.critic,
        }

        # Initialize weight projection if enabled
        self.weight_projection_norms = {}
        if args.use_weight_projection:
            self.weight_projection_norms["actor"] = get_initial_norms(self.network.actor)
            self.weight_projection_norms["critic"] = get_initial_norms(self.network.critic)

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
            "critic": StatisticalMetricsComputer(),
        }
        self.prev_obs = None
        self.prev_action = None

    def initialize_for_episode(self) -> None:
        self.prev_obs = None
        self.prev_action = None

    @torch.inference_mode()
    def select_action(self, global_step, obs) -> tuple[np.ndarray, dict]:
        info_dict = {}

        obs_tensor = torch.Tensor(obs).to(self.device).unsqueeze(0)
        output_enc, _ = self.network.encoder_image.forward(obs_tensor)

        if global_step < self.learning_starts:
            action = self.action_space.sample()
        else:
            action, selected_log_pi = self.network.actor.get_action(output_enc)
            action = action[0].detach().cpu().numpy()
            action = action * self.action_scale + self.action_bias

            action_noise = self.action_space.sample()
            c = self.action_noise
            action = (1 - c) * action + c * action_noise
            action = np.clip(action, self.action_low, self.action_high)
            info_dict["selected_log_pi"] = selected_log_pi[0].item()

        self.prev_obs = obs
        self.prev_action = action
        return action, info_dict

    def train(self, global_step, obs, reward, termination, truncation) -> dict:
        info_dict = {}

        action_norm = np.linalg.norm(self.prev_action)
        train_reward = 0.1 * reward - self.action_norm_penalty * action_norm
        info_dict["action_norm"] = action_norm
        info_dict["train_reward"] = train_reward

        self.rb.add(
            self.prev_obs,
            self.prev_action,
            train_reward,
            False,
        )

        if global_step < self.learning_starts:
            return info_dict
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        # training
        data = self.rb.sample(self.batch_size)

        # Compute all losses using refactored methods
        obs_curr = data.observations[:, -2]
        state_curr, _ = self.network.encoder_image.forward(obs_curr)

        # Actor
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(state_curr)
        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        # Critic
        critic_loss, critic_activations, critic_info = self.network.compute_critic_loss(
            data, state_curr
        )
        for key, value in critic_info.items():
            info_dict[f"losses/{key}"] = value

        # optimize the model
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        # Gradient and parameter norms
        for key, value in self.monitoring_targets.items():
            info_dict[f"gradients/{key}"] = compute_gradient_norm(value)
            info_dict[f"parameters/{key}"] = compute_parameter_norm(value)

        self.optimizer.step()

        # Apply weight projection after optimizer step
        if self.use_weight_projection:
            weight_project(self.network.actor, self.weight_projection_norms["actor"])
            weight_project(self.network.critic, self.weight_projection_norms["critic"])

        # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
        if self.apply_masks_during_training:
            apply_masks_during_training(self.network.actor)
            apply_masks_during_training(self.network.critic)

        # Feature metrics
        feature_dict = {
            "state": state_curr,
            **actor_activations,
            **critic_activations,
        }
        for feature_name, feature in feature_dict.items():
            info_dict[f"activation_norms/{feature_name}"] = feature.norm(dim=1).mean().item()

            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        return info_dict

    def step(self, global_step, obs, reward, termination, truncation) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # train
        train_info = self.train(global_step, obs, reward, termination, truncation)
        info_dict.update(train_info)

        # make decision
        action, action_info = self.select_action(global_step, obs)
        info_dict.update(action_info)

        return action, info_dict
