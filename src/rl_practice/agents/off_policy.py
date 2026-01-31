# SPDX-License-Identifier: MIT
import numpy as np
import torch
from torch import optim

from rl_practice.networks.actor_critic_with_action_value import Network
from rl_practice.replay_buffer import ReplayBuffer
from rl_practice.reward_processor import RewardProcessor


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
        self.max_grad_norm = args.max_grad_norm
        self.use_done = args.use_done

        # Sequence observation management
        self.seq_len = args.seq_len
        self.horizon = args.horizon

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        if args.network_class == "actor_critic_with_action_value":
            self.network = Network(
                observation_space.shape, action_dim=self.action_dim, args=args
            ).to(self.device)
        else:
            raise ValueError(f"Unknown network class: {args.network_class}")
        self.network = torch.compile(self.network)
        self.rnn_state = self.network.init_state().to(self.device)
        lr = args.learning_rate
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.0)

        obs_z_shape = tuple(self.network.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=args.buffer_size,
            seq_len=self.seq_len + self.horizon,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
            max_token_len=args.max_token_len,
            pad_token_id=args.pad_token_id,
        )

        # Initialize gradient norm targets

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # Reset chunk on episode boundary
        if terminated or truncated:
            self.action_chunk = None
            self.chunk_step = 0

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
        obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
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
            [],
        )

        # Use cached action from chunk if available (except during random exploration)
        if (
            global_step >= self.learning_starts
            and self.action_chunk is not None
            and self.chunk_step < self.horizon
        ):
            action = self.action_chunk[self.chunk_step]
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
            self.prev_action = action
            self.chunk_step += 1
            info_dict["chunk_step"] = self.chunk_step
            return action, info_dict

        # inference - predict new action chunk
        latest_data = self.rb.get_latest(self.seq_len)
        infer_dict = self.network.infer(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
        )
        self.rnn_state = infer_dict["rnn_state"]

        # action
        if global_step < self.learning_starts:
            action = self.action_space.sample()
            self.action_chunk = None
            self.chunk_step = 0
        else:
            # action chunk: (B, horizon, action_dim) -> (horizon, action_dim)
            action_chunk = infer_dict["action"][0].cpu().numpy()
            self.action_chunk = action_chunk
            self.chunk_step = 1

            # Use first action from chunk
            action = action_chunk[0]
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action

        # predict next state
        if hasattr(self.network, "predict_next_state"):
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            next_image, next_reward = self.network.predict_next_state(
                infer_dict["x"], action_tensor.unsqueeze(0)
            )
            info_dict["next_image"] = next_image
            info_dict["next_reward"] = next_reward

        info_dict["chunk_step"] = self.chunk_step
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

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}

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

        self.optimizer.step()

        return info_dict
