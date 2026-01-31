# SPDX-License-Identifier: MIT
import numpy as np
import torch
from torch import nn, optim

from rl_practice.networks.actor_critic_with_action_value import Network as ActionValueNetwork
from rl_practice.networks.actor_critic_with_state_value import Network as StateValueNetwork
from rl_practice.networks.vlm_actor_critic_with_state_value import VLMActorCriticWithStateValue
from rl_practice.replay_buffer import ReplayBuffer, ReplayBufferData
from rl_practice.reward_processor import RewardProcessor


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
    def __init__(self, args, observation_space, action_space) -> None:
        self.on_policy_epoch = args.on_policy_epoch
        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.gamma = args.gamma
        self.buffer_capacity = args.buffer_capacity
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.accumulation_steps = args.accumulation_steps
        self.device = torch.device("cuda")
        self.num_bins = args.num_bins
        self.network_class = args.network_class
        self.action_norm_penalty = args.action_norm_penalty
        self.max_grad_norm = args.max_grad_norm
        self.use_done = args.use_done
        self.use_weight_projection = args.use_weight_projection
        self.apply_masks_during_training = args.apply_masks_during_training

        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = args.normalizing_by_return

        if self.network_class == "actor_critic_with_state_value":
            self.network = StateValueNetwork(observation_space.shape, action_space.shape, args).to(
                self.device
            )
            self.network = torch.compile(self.network)
        elif self.network_class == "actor_critic_with_action_value":
            self.network = ActionValueNetwork(
                observation_space.shape, action_dim=self.action_dim, args=args
            ).to(self.device)
            self.network = torch.compile(self.network)
        elif self.network_class == "vlm_actor_critic_with_state_value":
            self.network = VLMActorCriticWithStateValue(
                observation_space.shape, action_space.shape, args
            )
        else:
            raise ValueError(f"Invalid network_class: {self.network_class}")
        self.rnn_state = self.network.init_state().to(self.device)
        obs_z_shape = tuple(self.network.image_processor.output_shape)

        self.max_token_len = args.max_token_len
        self.pad_token_id = args.pad_token_id

        self.rb = ReplayBuffer(
            size=self.buffer_capacity,
            seq_len=self.seq_len + 1,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
            max_token_len=self.max_token_len,
            pad_token_id=self.pad_token_id,
        )

        lr = args.learning_rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.parse_fail_penalty = args.parse_fail_penalty

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_logp = 0.0
        self.prev_value = 0.0
        self.prev_action_token_ids: list[int] = []
        self.prev_parse_success = True

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # calculate train reward
        action_norm = np.linalg.norm(self.prev_action)
        parse_fail_penalty = 0.0 if self.prev_parse_success else self.parse_fail_penalty
        reward_with_penalty = reward - self.action_norm_penalty * action_norm - parse_fail_penalty
        if not self.normalizing_by_return:
            self.reward_processor.update(reward_with_penalty)
        info_dict["action_norm"] = action_norm
        info_dict["parse_fail_penalty"] = parse_fail_penalty
        info_dict["reward_with_penalty"] = reward_with_penalty
        info_dict["processed_reward"] = self.reward_processor.normalize(
            torch.tensor(reward_with_penalty)
        ).item()

        # add to replay buffer
        obs_tensor = torch.from_numpy(obs).to(self.device)
        obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
        obs_z = obs_z.squeeze(0)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward_with_penalty,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            self.prev_logp,
            self.prev_value,
            self.prev_action_token_ids,
        )

        # inference
        latest_data = self.rb.get_latest(1)
        result_dict = self.network(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
            None,
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
        if self.num_bins > 1 and self.network_class not in ["vlm_actor_critic_with_state_value"]:
            value = self.network.hl_gauss_loss(value)
        value = value.item()
        self.prev_value = value
        info_dict["value"] = value

        # action token ids and parse success
        self.prev_action_token_ids = result_dict["action_token_ids"]
        self.prev_parse_success = result_dict["parse_success"]

        # predict next state
        if self.network_class != "vlm_actor_critic_with_state_value":
            action_tensor = result_dict["action"]
            next_image, next_reward = self.network.predict_next_state(
                result_dict["x"], action_tensor
            )
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

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        info_dict = {}

        # Train with feedback text when using VLM
        if self.network_class != "vlm_actor_critic_with_state_value":
            return info_dict
        if not feedback_text:
            return info_dict

        latest_data = self.rb.get_latest(self.seq_len)
        loss_dict = self.network.train_with_feedback(
            latest_data.observations,
            latest_data.rewards,
            feedback_text,
        )

        self.optimizer.zero_grad()
        loss_dict["feedback_loss"].backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        info_dict["losses/feedback_loss"] = loss_dict["feedback_loss"].item()
        print(f"Feedback training loss: {info_dict['losses/feedback_loss']:.4f}")
        return info_dict

    ####################
    # Internal methods #
    ####################

    def _train(self, last_value: float) -> dict:
        info_dict = {}

        if not self.rb.is_full():
            return info_dict

        buffer_data = self.rb.get_all_data()
        s = buffer_data.observations
        s_z = buffer_data.obs_z
        a = buffer_data.actions
        r = buffer_data.rewards.view(-1, 1)
        v = buffer_data.values.view(-1, 1)
        old_a_logp = buffer_data.log_probs.view(-1, 1)
        done = buffer_data.dones.view(-1, 1)
        rnn_states = buffer_data.rnn_state
        action_token_ids = buffer_data.action_token_ids

        bias = torch.tensor(self.action_bias, device=self.device).view(1, -1)
        scale = torch.tensor(self.action_scale, device=self.device).view(1, -1)
        a = (a - bias) / scale

        # apply reward processing
        r = self.reward_processor.normalize(r)

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
            self.optimizer.zero_grad()
            batch_idx = 0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - 1,
                self.batch_size,
                k_frames=self.seq_len + 1,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)  # [B, T]
                data = ReplayBufferData(
                    observations=s[indices],
                    obs_z=s_z[indices],
                    rewards=r[indices],
                    dones=done[indices],
                    rnn_state=rnn_states[indices],
                    actions=a[indices],
                    log_probs=old_a_logp[indices],
                    values=v[indices],
                    action_token_ids=action_token_ids[indices],
                )
                curr_adv = adv[indices[:, -2]]
                curr_target_v = target_v[indices[:, -2]]

                if self.network_class == "actor_critic_with_state_value":
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v, curr_adv
                    )
                elif self.network_class == "actor_critic_with_action_value":
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v.squeeze(1)
                    )
                elif self.network_class == "vlm_actor_critic_with_state_value":
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v.squeeze(1), curr_adv
                    )
                else:
                    raise ValueError(f"Invalid network_class: {self.network_class}")

                sum_action_loss += info_dict.get("actor_loss", 0.0) * len(data.observations)
                sum_value_loss += info_dict["critic_loss"] * len(data.observations)

                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

                # Collect metrics on the first batch only
                if is_first_batch:
                    # Feature metrics
                    for feature_name, feature in activations_dict.items():
                        metrics_dict[f"activation_norms/{feature_name}"] = (
                            feature.norm(dim=1).mean().item()
                        )
                    is_first_batch = False

                batch_idx += 1
                if batch_idx % self.accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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
