import numpy as np
import torch
from torch import nn, optim

from networks.actor_critic_with_action_value import Network as ActionValueNetwork
from networks.actor_critic_with_state_value import Network as StateValueNetwork
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
    max_grad_norm = 5.0
    on_policy_epoch = 4
    gamma = 0.99

    def __init__(self, args, observation_space, action_space) -> None:
        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = action_space.high - action_space.low
        self.action_bias = (action_space.high + action_space.low) / 2.0 - 0.5 * self.action_scale
        print(f"Action space: {action_space}, dim: {self.action_dim}")
        print(f"  scale: {self.action_scale}, bias: {self.action_bias}")

        self.buffer_capacity = args.buffer_capacity
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.device = torch.device("cuda")
        self.num_bins = args.num_bins
        self.use_action_value = args.use_action_value

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
        self.latest_buffer = ReplayBuffer(
            size=self.seq_len,
            seq_len=self.seq_len,
            obs_shape=observation_space.shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
        )

        lr = args.learning_rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def initialize_for_episode(self) -> None:
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_logp = 0.0
        self.prev_value = 0.0

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        self.latest_buffer.add(
            torch.from_numpy(obs).to(self.device),
            reward,
            terminated or truncated,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            self.prev_logp,
            self.prev_value,
        )
        latest_data = self.latest_buffer.get_latest(1)

        result_dict = self.network(
            latest_data.rewards, latest_data.observations, latest_data.actions, self.rnn_state
        )
        action = result_dict["action"]
        a_logp = result_dict["a_logp"]
        value = result_dict["value"]
        self.rnn_state = result_dict["rnn_state"]

        if self.num_bins > 1:
            value = self.network.hl_gauss_loss(value)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        value = value.item()

        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)

        action_info = {
            "a_logp": a_logp,
            "value": value,
        }

        for key in ["x", "value_x", "policy_x"]:
            if key in result_dict:
                value_tensor = result_dict[key]
                action_info[f"activation/{key}_norm"] = value_tensor.norm(dim=1).mean().item()
                action_info[f"activation/{key}_mean"] = value_tensor.mean(dim=1).mean().item()
                action_info[f"activation/{key}_std"] = value_tensor.std(dim=1).mean().item()

        self.prev_action = action
        self.prev_value = value
        self.prev_logp = a_logp

        return action, action_info

    def step(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # store data to buffer
        self.rb.add(
            torch.from_numpy(obs).to(self.device),
            reward,
            terminated or truncated,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            self.prev_logp,
            self.prev_value,
        )

        # make decision
        action, action_info = self.select_action(global_step, obs, reward, terminated, truncated)
        info_dict.update(action_info)

        # train
        if self.rb.is_full():
            train_result = self._train(action_info["value"])
            info_dict.update(train_result)
            self.rb.reset()

        return action, info_dict

    ####################
    # Internal methods #
    ####################

    def _train(self, last_value: float) -> None:
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
                self.optimizer.step()

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)

        result_dict = {
            "on_policy/average_action_loss": np.mean(ave_action_loss_list),
            "on_policy/average_value_loss": np.mean(ave_value_loss_list),
        }
        return result_dict
