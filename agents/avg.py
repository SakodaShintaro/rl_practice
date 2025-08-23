"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse

import numpy as np
import torch
from torch import optim

from agents.sac import Network
from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from replay_buffer import ReplayBufferData


# https://github.com/mohmdelsayed/streaming-drl/blob/main/stream_ac_continuous.py
# https://github.com/mohmdelsayed/streaming-drl/blob/main/LICENSE.md
class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, et_lambda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, et_lambda=et_lambda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["et_lambda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


class AdaptiveObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, et_lambda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, et_lambda=et_lambda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        z_sum = 0.0
        self.counter += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue
                e, v = state["eligibility_trace"], state["v"]
                e.mul_(group["gamma"] * group["et_lambda"]).add_(p.grad, alpha=1.0)

                v.mul_(group["beta2"]).addcmul_(delta * e, delta * e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                if reset:
                    e.zero_()


class AvgAgent:
    def __init__(self, args: argparse.Namespace, observation_space, action_space) -> None:
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

        # Use SAC's Network class
        self.network = Network(action_dim=self.action_dim, args=args).to(self.device)
        self.seq_len = args.seq_len
        # Store observation, action, and reward history for sequence modeling
        self.obs_history = []
        self.action_history = []
        self.reward_history = []

        self.gamma = args.gamma

        self.use_eligibility_trace = args.use_eligibility_trace
        self.et_lambda = args.et_lambda

        self.monitoring_targets = {
            "total": self.network,
            "actor": self.network.actor,
            "critic": self.network.critic,
        }

        self.metrics_computers = {
            "state": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
            "critic": StatisticalMetricsComputer(),
        }

        lr = args.learning_rate
        if self.use_eligibility_trace:
            self.optimizer = AdaptiveObGD(
                self.network.parameters(), lr=lr, gamma=self.gamma, et_lambda=self.et_lambda
            )
        else:
            self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-5)

        # Initialize state tracking
        self._prev_action = None

    def initialize_for_episode(self) -> None:
        """Initialize for new episode."""
        self._prev_action = torch.zeros(self.action_dim, device=self.device)
        self.obs_history = [
            torch.zeros(self.observation_space.shape, device=self.device)
            for _ in range(self.seq_len)
        ]
        self.action_history = [
            torch.zeros(self.action_dim, device=self.device) for _ in range(self.seq_len)
        ]
        self.reward_history = [torch.tensor(0.0, device=self.device) for _ in range(self.seq_len)]

    @torch.inference_mode()
    def select_action(self, global_step, obs, reward: float) -> tuple[np.ndarray, dict]:
        obs_tensor = torch.Tensor(obs).to(self.device)

        # Update observation history
        self.obs_history.append(obs_tensor)
        self.obs_history.pop(0)

        # Update reward history
        reward_tensor = torch.tensor(reward, device=self.device)
        self.reward_history.append(reward_tensor)
        self.reward_history.pop(0)

        # Update action history
        self.action_history.append(self._prev_action.squeeze(0))
        self.action_history.pop(0)

        # Create sequence tensor
        obs_sequence = torch.stack(self.obs_history, dim=0).unsqueeze(0)  # (1, seq_len, C, H, W)

        # Create action sequence for encoder
        action_sequence = torch.stack(self.action_history, dim=0).unsqueeze(0)

        # Create reward sequence for encoder
        reward_sequence = torch.stack(self.reward_history, dim=0).unsqueeze(0)

        obs_encoded = self.network.encoder.forward(obs_sequence, action_sequence, reward_sequence)
        action, log_prob = self.network.actor.get_action(obs_encoded)

        # Store current state and action for next update
        self._prev_action = action

        action = action[0].detach().cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self._prev_action_np = action
        return action, {"selected_log_pi": log_prob[0].item()}

    def step(self, global_step, obs, reward, termination, truncation) -> tuple[np.ndarray, dict]:
        info_dict = {}

        action_norm = np.linalg.norm(self._prev_action_np)
        train_reward = 0.1 * reward - self.action_norm_penalty * action_norm
        info_dict["action_norm"] = action_norm
        info_dict["train_reward"] = train_reward

        curr_obs = torch.Tensor(obs).unsqueeze(0).to(self.device)

        # Create observations compatible with SAC's expectations
        # SAC expects obs_curr = [:, :-1] and obs_next = [:, 1:]
        # So we need seq_len+1 total frames

        # Add batch dimension to obs_history and append current observation
        obs_with_batch = [obs.unsqueeze(0) for obs in self.obs_history] + [curr_obs]
        observations = torch.stack(obs_with_batch, dim=1).to(self.device)  # (1, seq_len+1, C, H, W)

        # Create actions using action history and current action
        action_list = self.action_history + [self._prev_action.squeeze(0)]
        actions = torch.stack(action_list, dim=0).unsqueeze(0)  # [1, seq_len+1, action_dim]

        # Create rewards using reward history and current reward
        train_reward_tensor = torch.tensor(train_reward, device=self.device)
        reward_list = self.reward_history + [train_reward_tensor]
        rewards = torch.stack(reward_list, dim=0).unsqueeze(0)

        dones = torch.tensor(
            [[False] * (self.seq_len + 1)], device=self.device, dtype=torch.float32
        )

        data = ReplayBufferData(
            observations=observations, actions=actions, rewards=rewards, dones=dones
        )

        # Encode current state
        curr_obs = data.observations[:, :-1]
        curr_action = data.actions[:, :-1]
        curr_reward = data.rewards[:, :-1]
        curr_state = self.network.encoder.forward(curr_obs, curr_action, curr_reward)

        # Actor
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(curr_state)
        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        # Critic
        critic_loss, critic_activations, critic_info = self.network.compute_critic_loss(
            data, curr_state
        )
        for key, value in critic_info.items():
            info_dict[f"losses/{key}"] = value

        # optimize the model
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        # Gradient and parameter norms
        for key, value in self.monitoring_targets.items():
            info_dict[f"gradients/{key}"] = compute_gradient_norm(value)
            info_dict[f"parameters/{key}"] = compute_parameter_norm(value)

        if self.use_eligibility_trace:
            delta = critic_info["delta"]
            self.optimizer.step(delta, reset=(termination or truncation))
        else:
            self.optimizer.step()

        # Feature metrics
        feature_dict = {
            "state": curr_state,
            **actor_activations,
            **critic_activations,
        }
        for feature_name, feature in feature_dict.items():
            info_dict[f"activation_norms/{feature_name}"] = feature.norm(dim=1).mean().item()

            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        # make decision
        action, action_info = self.select_action(global_step, obs, reward)
        info_dict.update(action_info)

        return action, info_dict
