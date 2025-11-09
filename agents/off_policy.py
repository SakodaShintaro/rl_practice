import numpy as np
import torch
from torch import optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.actor_critic_with_action_value import Network
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer
from reward_processor import RewardProcessor


class OffPolicyAgent:
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
        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = args.normalizing_by_return

        self.learning_starts = args.learning_starts
        self.batch_size = args.batch_size
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training
        self.max_grad_norm = args.max_grad_norm
        self.use_done = args.use_done

        # Sequence observation management
        self.seq_len = args.seq_len

        self.network = Network(observation_space.shape, action_dim=self.action_dim, args=args).to(
            self.device
        )
        self.rnn_state = self.network.init_state().to(self.device)
        lr = args.learning_rate
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.0)

        obs_z_shape = tuple(self.network.encoder.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=args.buffer_size,
            seq_len=self.seq_len + 1,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
        )

        # Initialize gradient norm targets
        self.monitoring_targets = {
            "total": self.network,
            "actor": self.network.policy_head,
            "critic": self.network.value_head,
            "state_predictor": self.network.prediction_head.state_predictor,
        }

        # Initialize weight projection if enabled
        self.weight_projection_norms = {}
        if args.use_weight_projection:
            self.weight_projection_norms["actor"] = get_initial_norms(self.network.policy_head)
            self.weight_projection_norms["critic"] = get_initial_norms(self.network.value_head)
            self.weight_projection_norms["state_predictor"] = get_initial_norms(
                self.network.prediction_head.state_predictor
            )

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
            "critic": StatisticalMetricsComputer(),
            "state_predictor": StatisticalMetricsComputer(),
        }
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # calculate train reward
        action_norm = np.linalg.norm(self.prev_action)
        reward_with_penalty = reward - self.action_norm_penalty * action_norm
        if not self.normalizing_by_return:
            self.reward_processor.update(reward_with_penalty)
        info_dict["action_norm"] = action_norm
        info_dict["reward_with_penalty"] = reward_with_penalty
        info_dict["processed_reward"] = self.reward_processor.normalize(
            torch.tensor(reward_with_penalty)
        ).item()

        # add to replay buffer
        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            obs_z = self.network.encoder.image_processor.encode(obs_tensor.unsqueeze(0))
            obs_z = obs_z.squeeze(0)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward_with_penalty,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            0.0,
            0.0,
        )

        # inference
        latest_data = self.rb.get_latest(self.seq_len)
        output_enc, self.rnn_state = self.network.encoder.forward(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
        )

        # action
        if global_step < self.learning_starts:
            action = self.action_space.sample()
        else:
            action, _ = self.network.policy_head.get_action(output_enc)
            action = action[0].cpu().numpy()
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action

        # predict next state
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        next_image, next_reward = self.network.predict_next_state(
            output_enc, action_tensor.unsqueeze(0)
        )
        info_dict["next_image"] = next_image
        info_dict["next_reward"] = next_reward

        return action, info_dict

    def step(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # train
        train_info = self._train(global_step)
        info_dict.update(train_info)

        # make decision
        action, action_info = self.select_action(global_step, obs, reward, terminated, truncated)
        info_dict.update(action_info)

        return action, info_dict

    ####################
    # Internal methods #
    ####################

    def _train(self, global_step) -> dict:
        info_dict = {}

        if global_step < self.learning_starts:
            return info_dict
        elif global_step == self.learning_starts:
            print(f"Start training at global step {global_step}.")

        # Sample data for training using ReplayBuffer
        data = self.rb.sample(self.batch_size)

        # apply reward processing
        data.rewards = self.reward_processor.normalize(data.rewards)

        # compute target value
        target_value = self.network.compute_target_value(data)

        # compute loss
        loss, activation_dict, info_dict = self.network.compute_loss(data, target_value)

        # add prefixes to info_dict keys
        info_dict = {f"losses/{key}": value for key, value in info_dict.items()}

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)

        # Gradient and parameter norms
        for key, value in self.monitoring_targets.items():
            info_dict[f"gradients/{key}"] = compute_gradient_norm(value)
            info_dict[f"parameters/{key}"] = compute_parameter_norm(value)

        self.optimizer.step()

        # Apply weight projection after optimizer step
        if self.use_weight_projection:
            weight_project(self.network.policy_head, self.weight_projection_norms["actor"])
            weight_project(self.network.value_head, self.weight_projection_norms["critic"])
            weight_project(
                self.network.prediction_head.state_predictor,
                self.weight_projection_norms["state_predictor"],
            )

        # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
        if self.apply_masks_during_training:
            apply_masks_during_training(self.network.policy_head)
            apply_masks_during_training(self.network.value_head)
            apply_masks_during_training(self.network.prediction_head.state_predictor)

        # Feature metrics
        for feature_name, feature in activation_dict.items():
            info_dict[f"activation_norms/{feature_name}"] = feature.norm(dim=1).mean().item()

            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        return info_dict
