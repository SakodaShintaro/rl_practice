# SPDX-License-Identifier: MIT
import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from rl_practice.networks.actor_critic_with_action_value import ActorCriticWithActionValue
from rl_practice.networks.actor_critic_with_state_value import ActorCriticWithStateValue
from rl_practice.networks.vlm_actor_critic_with_state_value import VLMActorCriticWithStateValue
from rl_practice.replay_buffer import ReplayBuffer
from rl_practice.reward_processor import RewardProcessor


class StreamingAgent:
    def __init__(
        self,
        args: argparse.Namespace,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
    ) -> None:
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

        self.max_grad_norm = args.max_grad_norm
        self.use_done = args.use_done
        self.accumulation_steps = args.accumulation_steps
        self._accumulation_count = 0

        # Sequence observation management
        self.seq_len = args.seq_len
        self.horizon = args.horizon

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        if args.network_class == "actor_critic_with_action_value":
            self.network = ActorCriticWithActionValue(
                observation_space.shape, action_space.shape, args
            ).to(self.device)
            self.network = torch.compile(self.network)
        elif args.network_class == "actor_critic_with_state_value":
            self.network = ActorCriticWithStateValue(
                observation_space.shape, action_space.shape, args
            ).to(self.device)
            self.network = torch.compile(self.network)
        elif args.network_class == "vlm_actor_critic_with_state_value":
            self.network = VLMActorCriticWithStateValue(
                observation_space.shape, action_space.shape, args
            )
        else:
            raise ValueError(f"Unknown network class: {args.network_class}")
        self.rnn_state = self.network.init_state().to(self.device)
        lr = args.learning_rate
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.1)

        obs_z_shape = tuple(self.network.image_processor.output_shape)
        self.rb = ReplayBuffer(
            size=self.seq_len + self.horizon,
            seq_len=self.seq_len + self.horizon,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
            max_new_tokens=args.max_new_tokens,
            pad_token_id=args.pad_token_id,
        )

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    def _prepare_step(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info_dict: dict
    ) -> None:
        if terminated or truncated:
            self.action_chunk = None
            self.chunk_step = 0

        action_norm = np.linalg.norm(self.prev_action)
        reward_with_penalty = reward - self.action_norm_penalty * action_norm
        if not self.normalizing_by_return:
            self.reward_processor.update(reward_with_penalty)
        info_dict["action_norm"] = action_norm
        info_dict["reward_with_penalty"] = reward_with_penalty
        info_dict["processed_reward"] = self.reward_processor.normalize(
            torch.tensor(reward_with_penalty)
        ).item()

        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.inference_mode():
            obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
            obs_z = obs_z.squeeze(0)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        self.rb.add(
            obs_tensor,
            obs_z,
            reward_with_penalty,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            0.0,
            0.0,
            [],
        )

    def _use_action_chunk(self, info_dict: dict) -> np.ndarray:
        action = self.action_chunk[self.chunk_step]
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action
        self.chunk_step += 1
        info_dict["chunk_step"] = self.chunk_step
        return action

    def _start_new_chunk(self, infer_dict: dict, info_dict: dict) -> np.ndarray:
        self.rnn_state = infer_dict["rnn_state"]
        info_dict["value"] = infer_dict["value"]
        info_dict["next_image"] = infer_dict["next_image"]
        info_dict["next_reward"] = infer_dict["next_reward"]

        action_chunk = infer_dict["action"][0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1

        action = action_chunk[0]
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action
        info_dict["chunk_step"] = self.chunk_step
        return action

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}
        self._prepare_step(obs, reward, terminated, truncated, info_dict)

        if self.action_chunk is not None and self.chunk_step < self.horizon:
            return self._use_action_chunk(info_dict), info_dict

        latest_data = self.rb.get_latest(self.seq_len)
        infer_dict = self.network.infer(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
        )
        return self._start_new_chunk(infer_dict, info_dict), info_dict

    def step(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}
        self._prepare_step(obs, reward, terminated, truncated, info_dict)

        # cached action: no inference, no training
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            return self._use_action_chunk(info_dict), info_dict

        # combined inference + training
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)

        infer_dict, loss, activation_dict, loss_info = self.network.infer_and_compute_loss(data)
        action = self._start_new_chunk(infer_dict, info_dict)

        info_dict.update({f"losses/{key}": value for key, value in loss_info.items()})
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self._accumulation_count += 1
        if self._accumulation_count % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return action, info_dict

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}
