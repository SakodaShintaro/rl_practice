import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss
from torch import optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.backbone import AE, SimpleTransformerEncoder, STTEncoder
from networks.diffusion_policy import DiffusionPolicy, DiffusionStatePredictor
from networks.sac_tanh_policy_and_q import SacQ
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer


class Network(nn.Module):
    def __init__(self, action_dim: int, args):
        super(Network, self).__init__()
        self.gamma = 0.99
        self.num_bins = args.num_bins
        self.sparsity = args.sparsity
        self.seq_len = args.seq_len

        self.action_dim = action_dim

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.encoder == "ae":
            self.encoder_sequence = AE(seq_len=self.seq_len, device=device)
        elif args.encoder == "stt":
            self.encoder_sequence = STTEncoder(seq_len=self.seq_len, device=device)
        elif args.encoder == "simple":
            self.encoder_sequence = SimpleTransformerEncoder(seq_len=self.seq_len, device=device)
        else:
            raise ValueError(
                f"Unknown encoder: {args.encoder}. Only 'ae', 'stt', and 'simple' are supported."
            )

        self.actor = DiffusionPolicy(
            state_dim=self.encoder_sequence.output_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.critic = SacQ(
            in_channels=self.encoder_sequence.output_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=self.num_bins,
            sparsity=args.sparsity,
        )

        self.detach_actor = args.detach_actor
        self.detach_critic = args.detach_critic

        # Sequence modeling components (optional)
        self.enable_sequence_modeling = args.enable_sequence_modeling
        if args.enable_sequence_modeling:
            self.state_predictor = DiffusionStatePredictor(
                input_dim=self.encoder_sequence.output_dim + action_dim,
                state_dim=self.encoder_sequence.output_dim,
                hidden_dim=args.actor_hidden_dim,
                block_num=args.actor_block_num,
                sparsity=args.sparsity,
            )

        if self.num_bins > 1:
            value_range = 60
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-value_range,
                max_value=+value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def compute_critic_loss(self, data, state_curr):
        if self.detach_critic:
            state_curr = state_curr.detach()
        with torch.no_grad():
            obs_next = data.observations[:, 1:]
            actions_next = data.actions[:, 1:]
            rewards_next = data.rewards[:, 1:]
            state_next = self.encoder_sequence.forward(obs_next, actions_next, rewards_next)
            next_state_actions, _ = self.actor.get_action(state_next)
            next_critic_output_dict = self.critic(state_next, next_state_actions)
            next_critic_value = next_critic_output_dict["output"]
            if self.num_bins > 1:
                next_critic_value = self.hl_gauss_loss(next_critic_value).view(-1)
            else:
                next_critic_value = next_critic_value.view(-1)
            curr_reward = data.rewards[:, -1].flatten()
            curr_continue = 1 - data.dones[:, -1].flatten()
            target_value = curr_reward + curr_continue * self.gamma * next_critic_value

        curr_critic_output_dict = self.critic(state_curr, data.actions[:, -1])

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
        if self.detach_actor:
            state_curr = state_curr.detach()
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

    def compute_sequence_loss(self, data, state_curr):
        if not self.enable_sequence_modeling:
            # Return zero loss when sequence modeling is disabled
            return (
                torch.tensor(0.0, device=state_curr.device),
                {},
                {},
            )

        state_curr = state_curr.detach()

        # 最後のactionを取得 (actions[:, -1]がcurrent_stateに対応するaction)
        action_curr = data.actions[:, -1]  # (B, action_dim)

        # 次のstateをencodeする
        with torch.no_grad():
            obs_next_sequence = data.observations[:, 1:]  # (B, T, C, H, W)
            actions_next_sequence = data.actions[:, 1:]  # (B, T, action_dim)
            rewards_next_sequence = data.rewards[:, 1:]  # (B, T)
            target_state_next = self.encoder_sequence.forward(
                obs_next_sequence, actions_next_sequence, rewards_next_sequence
            )

        # current_stateとactionを結合
        state_action_input = torch.cat([state_curr, action_curr], dim=-1)

        # Flow Matching for state prediction
        x_0_state = torch.randn_like(target_state_next)
        t_state = torch.rand(size=(target_state_next.shape[0], 1), device=target_state_next.device)

        # Sample from interpolation path for state
        x_t_state = (1.0 - t_state) * x_0_state + t_state * target_state_next

        # Predict velocity for state using DiffusionStatePredictor
        pred_state_dict = self.state_predictor.forward(
            x_t_state, t_state.squeeze(1), state_action_input
        )
        pred_states_flat = pred_state_dict["output"]

        # Conditional vector field for state
        u_t_state = target_state_next - x_0_state

        # Flow Matching loss for state
        state_loss = F.mse_loss(pred_states_flat, u_t_state)

        activations_dict = {"state_predictor": pred_state_dict["activation"]}

        info_dict = {"seq_loss": state_loss.item()}

        return state_loss, activations_dict, info_dict


class SacAgent:
    def __init__(self, args, observation_space, action_space) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = observation_space

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.action_norm_penalty = args.action_norm_penalty

        self.learning_starts = args.learning_starts
        self.batch_size = args.batch_size
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training

        # Sequence observation management
        self.seq_len = args.seq_len
        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

        self.network = Network(action_dim=self.action_dim, args=args).to(self.device)
        lr = args.learning_rate
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.0)

        self.rb = ReplayBuffer(
            args.buffer_size,
            self.seq_len + 1,
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
        if args.enable_sequence_modeling:
            self.monitoring_targets["state_predictor"] = self.network.state_predictor

        # Initialize weight projection if enabled
        self.weight_projection_norms = {}
        if args.use_weight_projection:
            self.weight_projection_norms["actor"] = get_initial_norms(self.network.actor)
            self.weight_projection_norms["critic"] = get_initial_norms(self.network.critic)
            if args.enable_sequence_modeling:
                self.weight_projection_norms["state_predictor"] = get_initial_norms(
                    self.network.state_predictor
                )

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
            "critic": StatisticalMetricsComputer(),
        }
        if args.enable_sequence_modeling:
            self.metrics_computers["state_predictor"] = StatisticalMetricsComputer()
        self.prev_action = None

    def initialize_for_episode(self) -> None:
        self.prev_action = None
        self.observation_buffer = [
            torch.zeros(self.observation_space.shape, device=self.device)
            for _ in range(self.seq_len)
        ]
        self.action_buffer = [
            torch.zeros(self.action_dim, device=self.device) for _ in range(self.seq_len)
        ]
        self.reward_buffer = [torch.tensor(0.0, device=self.device) for _ in range(self.seq_len)]

    @torch.inference_mode()
    def select_action(self, global_step, obs, reward: float) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # Update observation buffer for sequence tracking
        obs_tensor = torch.Tensor(obs).to(self.device)
        self.observation_buffer.append(obs_tensor)
        self.observation_buffer.pop(0)

        # Update reward buffer
        reward_tensor = torch.tensor(reward, device=self.device)
        self.reward_buffer.append(reward_tensor)
        self.reward_buffer.pop(0)

        # (1, T, C, H, W)
        obs_sequence = torch.stack(self.observation_buffer, dim=0).unsqueeze(0)

        # (1, T, action_dim)
        action_sequence = torch.stack(self.action_buffer, dim=0).unsqueeze(0)

        # (1, T)
        reward_sequence = torch.stack(self.reward_buffer, dim=0).unsqueeze(0)

        output_enc = self.network.encoder_sequence.forward(
            obs_sequence, action_sequence, reward_sequence
        )

        if global_step < self.learning_starts:
            action = self.action_space.sample()
        else:
            action, selected_log_pi = self.network.actor.get_action(output_enc)
            action = action[0].detach().cpu().numpy()
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
            info_dict["selected_log_pi"] = selected_log_pi[0].item()

        # Update action buffer with action tensor
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.action_buffer.append(action_tensor)
        self.action_buffer.pop(0)

        self.prev_action = action
        return action, info_dict

    def train(self, global_step, obs, reward, termination, truncation) -> dict:
        info_dict = {}

        action_norm = np.linalg.norm(self.prev_action)
        train_reward = 0.1 * reward - self.action_norm_penalty * action_norm
        info_dict["action_norm"] = action_norm
        info_dict["train_reward"] = train_reward

        self.rb.add(obs, self.prev_action, train_reward, False)

        if global_step < self.learning_starts:
            return info_dict
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        # training
        data = self.rb.sample(self.batch_size)

        # Compute all losses using refactored methods
        obs_curr = data.observations[:, :-1]  # (B, T, C, H, W)
        actions_curr = data.actions[:, :-1]  # (B, T, action_dim)
        rewards_curr = data.rewards[:, :-1]  # (B, T)
        state_curr = self.network.encoder_sequence.forward(obs_curr, actions_curr, rewards_curr)

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

        # Sequence modeling
        sequence_loss, sequence_activations, sequence_info = self.network.compute_sequence_loss(
            data, state_curr
        )
        for key, value in sequence_info.items():
            info_dict[f"losses/{key}"] = value

        # optimize the model
        loss = actor_loss + critic_loss + sequence_loss
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
            if self.network.enable_sequence_modeling:
                weight_project(
                    self.network.state_predictor, self.weight_projection_norms["state_predictor"]
                )

        # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
        if self.apply_masks_during_training:
            apply_masks_during_training(self.network.actor)
            apply_masks_during_training(self.network.critic)
            if self.network.enable_sequence_modeling:
                apply_masks_during_training(self.network.state_predictor)

        # Feature metrics
        feature_dict = {
            "state": state_curr,
            **actor_activations,
            **critic_activations,
            **sequence_activations,
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
        action, action_info = self.select_action(global_step, obs, reward)
        info_dict.update(action_info)

        return action, info_dict
