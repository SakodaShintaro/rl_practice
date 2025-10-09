import numpy as np
import torch
from torch import optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.actor_critic_with_action_value import Network
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer


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
            "state_predictor": self.network.state_predictor,
        }

        # Initialize weight projection if enabled
        self.weight_projection_norms = {}
        if args.use_weight_projection:
            self.weight_projection_norms["actor"] = get_initial_norms(self.network.actor)
            self.weight_projection_norms["critic"] = get_initial_norms(self.network.critic)
            self.weight_projection_norms["state_predictor"] = get_initial_norms(
                self.network.state_predictor
            )

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
            "critic": StatisticalMetricsComputer(),
            "state_predictor": StatisticalMetricsComputer(),
        }
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

        output_enc = self.network.encoder.forward(obs_sequence, action_sequence, reward_sequence)

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

        # predict next state
        next_image, next_reward = self.network.predict_next_state(
            output_enc, action_tensor.unsqueeze(0)
        )
        info_dict["next_image"] = next_image
        info_dict["next_reward"] = next_reward

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
        state_curr = self.network.encoder.forward(obs_curr, actions_curr, rewards_curr)

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
            weight_project(
                self.network.state_predictor, self.weight_projection_norms["state_predictor"]
            )

        # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
        if self.apply_masks_during_training:
            apply_masks_during_training(self.network.actor)
            apply_masks_during_training(self.network.critic)
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
