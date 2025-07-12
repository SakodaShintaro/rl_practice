"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse

import numpy as np
import torch
from torch import optim

from agents.sac import Network
from replay_buffer import ReplayBufferData
from td_error_scaler import TDErrorScaler


class AvgAgent:
    def __init__(self, args: argparse.Namespace, observation_space, action_space) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        # Use SAC's Network class
        seq_len = 2
        enable_sequence_modeling = False  # AVG doesn't use sequence modeling
        self.network = Network(
            sparsity=args.sparsity,
            action_dim=self.action_dim,
            seq_len=seq_len,
            args=args,
            enable_sequence_modeling=enable_sequence_modeling,
        ).to(self.device)

        # Use learning rates from AVG args
        self.optimizer = optim.AdamW(self.network.parameters(), lr=1e-4, weight_decay=1e-5)

        self.gamma = args.gamma
        self.td_error_scaler = TDErrorScaler()
        self.G = 0

        self.use_eligibility_trace = args.use_eligibility_trace
        self.et_lambda = args.et_lambda

        with torch.no_grad():
            self.eligibility_traces_q = [
                torch.zeros_like(p, requires_grad=False) for p in self.network.qf1.parameters()
            ]

        # Initialize state tracking
        self._prev_obs = None
        self._prev_action = None

    def select_action(self, global_step, obs):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        obs_encoded = self.network.encoder_image.encode(obs_tensor)
        action, log_prob, _ = self.network.actor.get_action(obs_encoded)

        # Store current state and action for next update
        self._prev_obs = obs
        self._prev_action = action

        action = action[0].detach().cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        return action, {"selected_log_pi": log_prob[0].item()}

    def process_env_feedback(self, global_step, obs, action, reward, termination, truncation):
        info_dict = {}

        reward /= 10.0

        # Create ReplayBufferData for SAC's loss computation
        done = termination or truncation

        # Create batch data with seq_len=2 format that SAC expects
        observations = torch.stack(
            [
                torch.Tensor(self._prev_obs).unsqueeze(0),
                torch.Tensor(obs).unsqueeze(0),
            ],
            dim=1,
        ).to(self.device)  # [batch_size=1, seq_len=2, obs_dim]

        actions = (
            torch.stack(
                [
                    self._prev_action.squeeze(0),  # [action_dim]
                    self._prev_action.squeeze(0),  # dummy, only [:, -2] used
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(self.device)
        )  # [batch_size=1, seq_len=2, action_dim]

        rewards = torch.tensor([[0.0, reward]], device=self.device, dtype=torch.float32)
        dones = torch.tensor([[False, done]], device=self.device, dtype=torch.float32)

        data = ReplayBufferData(
            observations=observations, actions=actions, rewards=rewards, dones=dones
        )

        # Encode current state
        state_curr = self.network.encoder_image.encode(data.observations[:, -2])

        qf_loss, qf_activations, qf_info = self.network.compute_critic_loss(data, state_curr)
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(state_curr)

        # Combine losses (no sequence modeling for AVG)
        loss = actor_loss + qf_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Apply eligibility traces if enabled
        if self.use_eligibility_trace:
            with torch.no_grad():
                for p, et in zip(self.network.qf1.parameters(), self.eligibility_traces_q):
                    if p.grad is not None:
                        et.mul_(self.et_lambda * self.gamma).add_(p.grad.data)
                        p.grad.data = et

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Combine info from both losses
        for key, value in qf_info.items():
            info_dict[f"losses/{key}"] = value
        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        return info_dict

    def initialize_for_episode(self):
        """Initialize for new episode."""
        self._prev_obs = None
        self._prev_action = None

        # Reset eligibility traces
        if self.use_eligibility_trace:
            for et in self.eligibility_traces_q:
                et.zero_()
