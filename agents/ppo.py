import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from networks.ppo_beta_policy_and_value import PpoBetaPolicyAndValue
from networks.ppo_paligemma_policy_value import PpoPaligemmaPolicyAndValue


class SequentialBatchSampler:
    def __init__(self, buffer_capacity, batch_size, k_frames, drop_last=False):
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
    max_grad_norm = 0.5
    clip_param_policy = 0.1
    clip_param_value = 1.0
    ppo_epoch = 10
    gamma = 0.99

    def __init__(self, args, observation_space, action_space) -> None:
        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = action_space.high - action_space.low
        self.action_bias = (action_space.high + action_space.low) / 2.0 - 0.5 * self.action_scale

        self.buffer_capacity = args.buffer_capacity
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.training_step = 0
        self.device = torch.device("cuda")
        self.net = {
            "default": PpoBetaPolicyAndValue(3, self.seq_len).to(self.device),
            "paligemma": PpoPaligemmaPolicyAndValue(3).to(self.device),
        }[args.model_name]
        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")
        self.buffer = np.empty(
            self.buffer_capacity,
            dtype=np.dtype(
                [
                    ("s", np.float32, (3, 96, 96)),
                    ("a", np.float32, (3,)),
                    ("a_logp", np.float32),
                    ("r", np.float32),
                    ("v", np.float32),
                    ("done", np.int32),
                ]
            ),
        )
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.r_list = []
        self.s_list = []
        self.a_list = []

    def initialize_for_episode(self) -> None:
        self.episode_reward = 0.0
        self.episode_states = []
        self.episode_actions = []
        self.episode_values = []
        self.episode_logps = []

    def select_action(self, global_step, obs) -> tuple[np.ndarray, dict]:
        reward = getattr(self, "episode_reward", 0.0)
        action, a_logp, value, result_dict = self._select_action_original(reward, obs)

        action_info = {
            "a_logp": a_logp,
            "value": value,
            "reward": reward,
            "normed_reward": reward / 10.0,
        }

        for key in ["x", "value_x", "policy_x"]:
            if key in result_dict:
                value_tensor = result_dict[key]
                action_info[f"activation/{key}_norm"] = value_tensor.norm(dim=1).mean().item()
                action_info[f"activation/{key}_mean"] = value_tensor.mean(dim=1).mean().item()
                action_info[f"activation/{key}_std"] = value_tensor.std(dim=1).mean().item()

        # パラメータのnorm値を追加
        for name, p in self.net.named_parameters():
            action_info[f"params/{name}"] = p.norm().item()

        self.episode_states.append(obs)
        self.episode_actions.append(action)
        self.episode_values.append(value)
        self.episode_logps.append(a_logp)

        return action, action_info

    def step(self, global_step, obs, reward, termination, truncation) -> tuple[np.ndarray, dict]:
        self.episode_reward += reward
        normed_reward = reward / 10.0

        info_dict = {}

        if len(self.episode_states) > 0:
            prev_obs = self.episode_states[-1]
            prev_action = self.episode_actions[-1]
            prev_value = self.episode_values[-1]
            prev_logp = self.episode_logps[-1]

            self.buffer[self.counter] = (
                prev_obs,
                prev_action,
                prev_logp,
                normed_reward,
                prev_value,
                termination or truncation,
            )
            self.counter += 1
            if self.counter == self.buffer_capacity:
                train_result = self._update()
                info_dict.update(train_result)
                self.counter = 0

        if termination or truncation:
            # エピソード終了時の統計情報を追加
            if len(self.episode_values) > 0:
                info_dict["first_value"] = self.episode_values[0]
                info_dict["weighted_reward"] = getattr(self, "episode_reward", 0.0)

        # make decision
        action, action_info = self.select_action(global_step, obs)
        info_dict.update(action_info)

        return action, info_dict

    ####################
    # Internal methods #
    ####################

    @torch.inference_mode()
    def _select_action_original(self, reward: float, state: np.ndarray) -> tuple:
        reward = torch.from_numpy(np.array(reward)).to(self.device).unsqueeze(0)
        state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        self.r_list.append(reward)
        self.r_list = self.r_list[-self.seq_len :]
        self.s_list.append(state)
        self.s_list = self.s_list[-self.seq_len :]

        curr_r = torch.cat(self.r_list, dim=0).unsqueeze(0).unsqueeze(-1)
        curr_s = torch.cat(self.s_list, dim=0).unsqueeze(0)
        a_with_dummy = self.a_list + [torch.tensor([[0.0, 0.0, 0.0]], device=self.device)]
        curr_a = torch.cat(a_with_dummy, dim=0).unsqueeze(0)

        if curr_r.shape[1] < self.seq_len:
            padding_size = self.seq_len - curr_r.shape[1]
            pad_r = torch.zeros(1, padding_size, *curr_r.shape[2:], device=self.device)
            pad_s = torch.zeros(1, padding_size, *curr_s.shape[2:], device=self.device)
            pad_a = torch.zeros(1, padding_size, *curr_a.shape[2:], device=self.device)
            curr_r = torch.cat((pad_r, curr_r), dim=1)
            curr_s = torch.cat((pad_s, curr_s), dim=1)
            curr_a = torch.cat((pad_a, curr_a), dim=1)
        assert curr_r.shape[1] == self.seq_len
        assert curr_s.shape[1] == self.seq_len
        assert curr_a.shape[1] == self.seq_len

        result_dict = self.net(curr_r, curr_s, curr_a)
        action = result_dict["action"]
        a_logp = result_dict["a_logp"]
        value = result_dict["value"]
        self.a_list.append(action)
        self.a_list = self.a_list[-self.seq_len :]
        if len(self.a_list) == self.seq_len:
            self.a_list = self.a_list[1:]

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        value = value.item()

        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)

        return action, a_logp, value, result_dict

    def _update(self) -> None:
        self.training_step += 1

        s = torch.tensor(self.buffer["s"]).to(self.device)
        a = torch.tensor(self.buffer["a"]).to(self.device)
        r = torch.tensor(self.buffer["r"]).to(self.device).view(-1, 1)
        v = torch.tensor(self.buffer["v"]).to(self.device).view(-1, 1)
        old_a_logp = torch.tensor(self.buffer["a_logp"]).to(self.device).view(-1, 1)
        done = torch.tensor(self.buffer["done"]).to(self.device).view(-1, 1)

        bias = torch.tensor(self.action_bias, device=self.device).view(1, -1)
        scale = torch.tensor(self.action_scale, device=self.device).view(1, -1)
        a = (a - bias) / scale

        target_v = r[:-1] + (1 - done[:-1]) * self.gamma * v[1:]
        adv = target_v - v[:-1]
        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # noqa: ERA001

        ave_action_loss_list = []
        ave_value_loss_list = []
        ave_pred_s_loss_list = []
        for _ in range(self.ppo_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            sum_pred_s_loss = 0.0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - 1,
                self.batch_size,
                k_frames=self.seq_len,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)
                index = indices[:, -1]
                curr_action = a[indices][:, :-1]
                dummy_action = torch.zeros((curr_action.shape[0], 1, 3), device=self.device)
                curr_action = torch.cat((curr_action, dummy_action), dim=1)

                net_out_dict = self.net(r[indices], s[indices], curr_action, a[index])
                a_logp = net_out_dict["a_logp"]
                value = net_out_dict["value"]
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
                    * adv[index]
                )
                action_loss = -torch.min(surr1, surr2).mean()

                value_clipped = torch.clamp(
                    value, v[index] - self.clip_param_value, v[index] + self.clip_param_value
                )
                value_loss_unclipped = F.smooth_l1_loss(value, target_v[index])
                value_loss_clipped = F.smooth_l1_loss(value_clipped, target_v[index])
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                if "error" in net_out_dict:
                    pred_error = net_out_dict["error"]
                    pred_error_s = pred_error[:, 3::3]  # 先頭は明らかに予測不可能なので3から
                    pred_error_a = pred_error[:, 1::3]
                    pred_error_r = pred_error[:, 2::3]
                    pred_loss_s = pred_error_s.mean()
                else:
                    pred_loss_s = torch.tensor(0.0, device=self.device)

                loss = action_loss + 2.0 * value_loss + pred_loss_s
                sum_action_loss += action_loss.item() * len(index)
                sum_value_loss += value_loss.item() * len(index)
                sum_pred_s_loss += pred_loss_s.item() * len(index)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_pred_s_loss = sum_pred_s_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)
            ave_pred_s_loss_list.append(ave_pred_s_loss)
        result_dict = {}
        ratio_list = []
        result_dict["ppo/average_action_loss"] = np.mean(ave_action_loss_list)
        result_dict["ppo/average_value_loss"] = np.mean(ave_value_loss_list)
        result_dict["ppo/average_pred_s_loss"] = np.mean(ave_pred_s_loss_list)
        result_dict["ppo/average_ratio"] = np.mean(ratio_list)
        result_dict["ppo/average_target_v"] = target_v.mean().item()
        result_dict["ppo/average_adv"] = adv.mean().item()
        return result_dict
