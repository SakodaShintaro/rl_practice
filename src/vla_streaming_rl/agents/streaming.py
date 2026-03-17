# SPDX-License-Identifier: MIT
import argparse

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from vla_streaming_rl.networks.actor_critic_with_action_value import ActorCriticWithActionValue
from vla_streaming_rl.networks.actor_critic_with_state_value import ActorCriticWithStateValue
from vla_streaming_rl.networks.vlm_actor_critic_with_action_value import (
    VLMActorCriticWithActionValue,
)
from vla_streaming_rl.optimizers.adam_et import AdamET
from vla_streaming_rl.replay_buffer import ReplayBuffer
from vla_streaming_rl.reward_processor import RewardProcessor


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
        elif args.network_class == "actor_critic_with_state_value":
            self.network = ActorCriticWithStateValue(
                observation_space.shape, action_space.shape, args
            ).to(self.device)
        elif args.network_class == "vlm_actor_critic_with_action_value":
            self.network = VLMActorCriticWithActionValue(
                observation_space.shape, action_space.shape, args
            ).to(self.device)
        else:
            raise ValueError(f"Unknown network class: {args.network_class}")
        self.rnn_state = self.network.init_state().to(self.device)

        self.use_eligibility_trace = bool(args.use_eligibility_trace)
        lr = args.learning_rate
        self.critic_optimizer = None
        if self.use_eligibility_trace:
            critic_params = list(self.network.value_head.parameters())
            critic_param_ids = {id(p) for p in critic_params}
            other_params = [p for p in self.network.parameters() if id(p) not in critic_param_ids]
            self.critic_optimizer = AdamET(
                critic_params,
                lr=lr,
                gamma=args.gamma,
                et_lambda=args.et_lambda,
            )
            self.optimizer = optim.AdamW(other_params, lr=lr, weight_decay=0.1)
        else:
            self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.1)

        if args.network_class != "vlm_actor_critic_with_action_value":
            self.network = torch.compile(self.network)

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
            max_prompt_tokens=args.max_prompt_tokens,
            pad_token_id=args.pad_token_id,
        )

        self.network_class = args.network_class
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_action_token_ids = []
        self._episode_reset = False

    def _prepare_step(
        self,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info_dict: dict,
        task_prompt: str,
    ) -> None:
        if terminated or truncated:
            self.action_chunk = None
            self.chunk_step = 0
            self.prev_action_token_ids = []
            self._episode_reset = self.use_done

        action_norm = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(reward)
        info_dict["action_norm"] = action_norm
        info_dict["processed_reward"] = self.reward_processor.normalize(torch.tensor(reward)).item()

        obs_tensor = torch.from_numpy(obs).to(self.device)
        with torch.inference_mode():
            obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
            obs_z = obs_z.squeeze(0)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        task_prompt_token_ids = self.network.tokenize_task_prompt(task_prompt)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            0.0,
            0.0,
            self.prev_action_token_ids,
            task_prompt_token_ids,
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
        self.prev_action_token_ids = infer_dict.get("action_token_ids", [])
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
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}
        self._prepare_step(obs, reward, terminated, truncated, info_dict, task_prompt)

        if self.action_chunk is not None and self.chunk_step < self.horizon:
            return self._use_action_chunk(info_dict), info_dict

        latest_data = self.rb.get_latest(self.seq_len)
        infer_dict = self.network.infer(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
            task_prompts=[task_prompt],
        )
        return self._start_new_chunk(infer_dict, info_dict), info_dict

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}
        self._prepare_step(obs, reward, terminated, truncated, info_dict, task_prompt)

        # cached action: no inference, no training
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            return self._use_action_chunk(info_dict), info_dict

        # combined inference + training
        data = self.rb.get_latest(self.seq_len + self.horizon)
        data.rewards = self.reward_processor.normalize(data.rewards)

        infer_dict, loss, activation_dict, loss_info, et_info = self.network.infer_and_compute_loss(
            data
        )
        action = self._start_new_chunk(infer_dict, info_dict)

        info_dict.update({f"losses/{key}": value for key, value in loss_info.items()})

        if self.use_eligibility_trace:
            # Actor: backward actor-only loss → encoder + actor grads
            actor_loss = et_info["actor_entropy_loss"] / self.accumulation_steps
            actor_loss.backward(retain_graph=True)

            # Critic: backward -V(s) → value_head grads only (detached from encoder)
            neg_value = et_info["neg_value"] / self.accumulation_steps
            neg_value.backward()
        else:
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

        self._accumulation_count += 1
        if self._accumulation_count % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            if self.use_eligibility_trace:
                self.critic_optimizer.step(delta=et_info["delta"], reset=self._episode_reset)
                self._episode_reset = False
                self.critic_optimizer.zero_grad()
            self.optimizer.zero_grad()

        return action, info_dict

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}
