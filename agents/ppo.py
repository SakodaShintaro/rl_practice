import numpy as np
import torch
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss
from torch import nn, optim

from networks.actor_critic_with_state_value import Network
from replay_buffer import ReplayBuffer


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


class PpoAgent:
    max_grad_norm = 5.0
    clip_param_policy = 0.2
    clip_param_value = 0.2
    ppo_epoch = 4
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
        self.training_step = 0
        self.device = torch.device("cuda")
        self.num_bins = args.num_bins
        self.network = Network(observation_space.shape, action_space.shape, args).to(self.device)
        self.rnn_state = self.network.init_state().to(self.device)
        self.rb = ReplayBuffer(
            size=self.buffer_capacity,
            seq_len=self.seq_len + 1,
            obs_shape=observation_space.shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=action_space.shape,
            device=self.device,
        )
        self.latest_buffer = ReplayBuffer(
            size=self.seq_len,
            seq_len=self.seq_len,
            obs_shape=observation_space.shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=action_space.shape,
            device=self.device,
        )

        lr = args.learning_rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        if self.num_bins > 1:
            value_range = 1
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-value_range,
                max_value=+value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            ).to(self.device)

    def initialize_for_episode(self) -> None:
        self.prev_action = None
        self.prev_value = None
        self.prev_logp = None

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        self.latest_buffer.add(
            obs,
            reward,
            terminated or truncated,
            self.rnn_state.cpu().numpy(),
            self.prev_action,
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
            value = self.hl_gauss_loss(value)

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
            obs,
            reward,
            terminated or truncated,
            self.rnn_state.cpu().numpy(),
            self.prev_action,
            self.prev_logp,
            self.prev_value,
        )

        # make decision
        action, action_info = self.select_action(global_step, obs, reward, terminated, truncated)
        info_dict.update(action_info)

        # train
        if self.rb.is_full():
            train_result = self._update(action_info["value"])
            info_dict.update(train_result)
            self.rb.reset()

        return action, info_dict

    ####################
    # Internal methods #
    ####################

    def _update(self, last_value: float) -> None:
        self.training_step += 1

        s = torch.tensor(self.rb.observations).to(self.device)
        a = torch.tensor(self.rb.actions).to(self.device)
        r = torch.tensor(self.rb.rewards).to(self.device).view(-1, 1)
        v = torch.tensor(self.rb.values).to(self.device).view(-1, 1)
        old_a_logp = torch.tensor(self.rb.log_probs).to(self.device).view(-1, 1)
        done = torch.tensor(self.rb.dones).to(self.device).view(-1, 1)
        rnn_states = (
            torch.tensor(self.rb.rnn_states)
            .to(self.device)
            .view(-1, 1, self.network.encoder.output_dim)
        )

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
        for _ in range(self.ppo_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - 1,
                self.batch_size,
                k_frames=self.seq_len + 1,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)  # [B, T]
                indices_input = indices[:, :-1]  # [B, self.seq_len]
                index_curr = indices[:, -2]  # [B]
                index_next = indices[:, -1]  # [B]

                net_out_dict = self.network(
                    r[indices_input],
                    s[indices_input],
                    a[indices_input],
                    rnn_states[indices[:, 0]].permute(1, 0, 2).contiguous(),
                    a[index_next],
                )
                a_logp = net_out_dict["a_logp"]
                entropy = net_out_dict["entropy"]
                value = net_out_dict["value"]
                ratio = torch.exp(a_logp - old_a_logp[index_next])

                surr1 = ratio * adv[index_curr]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
                    * adv[index_curr]
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.num_bins > 1:
                    value_loss = self.hl_gauss_loss(value, target_v[index_curr].squeeze(1))
                else:
                    value_clipped = torch.clamp(
                        value,
                        v[index_next] - self.clip_param_value,
                        v[index_next] + self.clip_param_value,
                    )
                    value_loss_unclipped = F.mse_loss(value, target_v[index_curr])
                    value_loss_clipped = F.mse_loss(value_clipped, target_v[index_curr])
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                loss = action_loss + 0.25 * value_loss - 0.02 * entropy.mean()
                sum_action_loss += action_loss.item() * len(index_curr)
                sum_value_loss += value_loss.item() * len(index_curr)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)
        result_dict = {}
        result_dict["ppo/average_action_loss"] = np.mean(ave_action_loss_list)
        result_dict["ppo/average_value_loss"] = np.mean(ave_value_loss_list)
        result_dict["ppo/average_target_v"] = target_v.mean().item()
        result_dict["ppo/average_adv"] = adv.mean().item()
        return result_dict
