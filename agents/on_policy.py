import numpy as np
import torch
from torch import nn, optim

from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.actor_critic_with_action_value import Network as ActionValueNetwork
from networks.actor_critic_with_state_value import Network as StateValueNetwork
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer, ReplayBufferData


class SequentialBatchSampler:
    def __init__(self, buffer_capacity, batch_size, k_frames, drop_last):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.k_frames = k_frames
        self.drop_last = drop_last

    def __iter__(self):
        valid_starts = range(self.buffer_capacity - self.k_frames + 1)
        randomized_starts = torch.randperm(len(valid_starts)).tolist()

        batch = []
        for start_idx in randomized_starts:
            seq_indices = list(range(start_idx, start_idx + self.k_frames))
            batch.append(seq_indices)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        num_samples = self.buffer_capacity - self.k_frames + 1
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size


class OnPolicyAgent:
    on_policy_epoch = 4
    gamma = 0.99

    def __init__(self, args, observation_space, action_space) -> None:
        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.buffer_capacity = args.buffer_capacity
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.device = torch.device("cuda")
        self.num_bins = args.num_bins
        self.use_action_value = args.use_action_value
        self.action_norm_penalty = args.action_norm_penalty
        self.reward_scale = args.reward_scale
        self.max_grad_norm = args.max_grad_norm
        self.use_done = args.use_done
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training

        if self.use_action_value:
            self.network = ActionValueNetwork(
                observation_space.shape, action_dim=self.action_dim, args=args
            ).to(self.device)
        else:
            self.network = StateValueNetwork(observation_space.shape, action_space.shape, args).to(
                self.device
            )
        self.rnn_state = self.network.init_state().to(self.device)
        self.rb = ReplayBuffer(
            size=self.buffer_capacity,
            seq_len=self.seq_len + 1,
            obs_shape=observation_space.shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
        )

        lr = args.learning_rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

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
        self.prev_logp = 0.0
        self.prev_value = 0.0

    def initialize_for_episode(self) -> None:
        pass

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # calculate train reward
        action_norm = np.linalg.norm(self.prev_action)
        train_reward = self.reward_scale * (reward - self.action_norm_penalty * action_norm)
        info_dict["action_norm"] = action_norm
        info_dict["train_reward"] = train_reward

        # add to replay buffer
        self.rb.add(
            torch.from_numpy(obs).to(self.device),
            train_reward,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            self.prev_logp,
            self.prev_value,
        )

        # inference
        latest_data = self.rb.get_latest(1)
        result_dict = self.network(
            latest_data.rewards, latest_data.observations, latest_data.actions, self.rnn_state
        )
        self.rnn_state = result_dict["rnn_state"]

        # action
        action = result_dict["action"]
        action = action[0].cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action

        # log prob
        a_logp = result_dict["a_logp"]
        a_logp = a_logp.item()
        self.prev_logp = a_logp
        info_dict["a_logp"] = a_logp

        # value
        value = result_dict["value"]
        if self.num_bins > 1:
            value = self.network.hl_gauss_loss(value)
        value = value.item()
        self.prev_value = value
        info_dict["value"] = value

        # predict next state
        action_tensor = result_dict["action"]
        next_image, next_reward = self.network.predict_next_state(result_dict["x"], action_tensor)
        info_dict["next_image"] = next_image
        info_dict["next_reward"] = next_reward

        return action, info_dict

    def step(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # train
        train_info = self._train(self.prev_value)
        info_dict.update(train_info)

        # make decision
        action, action_info = self.select_action(global_step, obs, reward, terminated, truncated)
        info_dict.update(action_info)

        return action, info_dict

    ####################
    # Internal methods #
    ####################

    def _train(self, last_value: float) -> dict:
        info_dict = {}

        if not self.rb.is_full():
            return info_dict

        buffer_data = self.rb.get_all_data()
        s = buffer_data.observations
        a = buffer_data.actions
        r = buffer_data.rewards.view(-1, 1)
        v = buffer_data.values.view(-1, 1)
        old_a_logp = buffer_data.log_probs.view(-1, 1)
        done = buffer_data.dones.view(-1, 1)
        rnn_states = buffer_data.rnn_state.view(-1, 1, self.network.encoder.output_dim)

        bias = torch.tensor(self.action_bias, device=self.device).view(1, -1)
        scale = torch.tensor(self.action_scale, device=self.device).view(1, -1)
        a = (a - bias) / scale

        # preprocessing
        target_v = torch.zeros_like(v[:-1])
        adv = torch.zeros_like(v[:-1])
        last_advantage = 0
        for i in range(len(v) - 2, -1, -1):
            if done[i + 1]:
                last_value = 0
                last_advantage = 0
            target_v[i] = r[i + 1] + self.gamma * last_value
            delta = target_v[i] - v[i + 1]
            last_advantage = delta + self.gamma * 0.95 * last_advantage
            adv[i] = last_advantage
            last_value = v[i + 1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ave_action_loss_list = []
        ave_value_loss_list = []
        metrics_dict = {}
        is_first_batch = True

        for _ in range(self.on_policy_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - 1,
                self.batch_size,
                k_frames=self.seq_len + 1,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)  # [B, T]
                data = ReplayBufferData(
                    observations=s[indices],
                    rewards=r[indices],
                    dones=done[indices],
                    rnn_state=rnn_states[indices],
                    actions=a[indices],
                    log_probs=old_a_logp[indices],
                    values=v[indices],
                )
                curr_adv = adv[indices[:, -2]]
                curr_target_v = target_v[indices[:, -2]]

                if self.use_action_value:
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v.squeeze(1)
                    )
                else:
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v, curr_adv
                    )

                sum_action_loss += info_dict["actor_loss"] * len(data.observations)
                sum_value_loss += info_dict["critic_loss"] * len(data.observations)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

                # Collect metrics on the first batch only
                if is_first_batch:
                    # Gradient and parameter norms
                    for key, value in self.monitoring_targets.items():
                        metrics_dict[f"gradients/{key}"] = compute_gradient_norm(value)
                        metrics_dict[f"parameters/{key}"] = compute_parameter_norm(value)

                    # Feature metrics
                    for feature_name, feature in activations_dict.items():
                        metrics_dict[f"activation_norms/{feature_name}"] = (
                            feature.norm(dim=1).mean().item()
                        )

                        result_dict = self.metrics_computers[feature_name](feature)
                        for key, value in result_dict.items():
                            metrics_dict[f"{key}/{feature_name}"] = value

                    is_first_batch = False

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

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)

        result_dict = {
            "losses/actor_loss": np.mean(ave_action_loss_list),
            "losses/critic_loss": np.mean(ave_value_loss_list),
            **metrics_dict,
        }

        self.rb.reset()

        return result_dict
