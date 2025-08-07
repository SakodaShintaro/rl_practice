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

        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.action_norm_penalty = args.action_norm_penalty

        # Use SAC's Network class
        seq_len = 2
        self.network = Network(
            action_dim=self.action_dim,
            seq_len=seq_len,
            args=args,
            enable_sequence_modeling=False,
        ).to(self.device)

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

        if self.use_eligibility_trace:
            self.optimizer = AdaptiveObGD(
                self.network.parameters(), lr=1e-5, gamma=self.gamma, et_lambda=self.et_lambda
            )
        else:
            self.optimizer = optim.AdamW(self.network.parameters(), lr=1e-5, weight_decay=1e-5)

        # Initialize state tracking
        self._prev_obs = None
        self._prev_action = None

    def initialize_for_episode(self) -> None:
        """Initialize for new episode."""
        self._prev_obs = None
        self._prev_action = None

    def select_action(self, global_step, obs) -> tuple[np.ndarray, dict]:
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        obs_encoded, _ = self.network.encoder_image.forward(obs_tensor)
        action, log_prob = self.network.actor.get_action(obs_encoded)

        # Store current state and action for next update
        self._prev_obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
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

        observations = torch.stack([self._prev_obs, curr_obs], dim=1).to(self.device)

        actions = torch.stack(
            [
                self._prev_action.squeeze(0),
                self._prev_action.squeeze(0),  # dummy
            ],
            dim=0,
        ).unsqueeze(0)  # [batch_size=1, seq_len=2, action_dim]

        rewards = torch.tensor(
            [[train_reward, train_reward]], device=self.device, dtype=torch.float32
        )
        dones = torch.tensor([[False, False]], device=self.device, dtype=torch.float32)

        data = ReplayBufferData(
            observations=observations, actions=actions, rewards=rewards, dones=dones
        )

        # Encode current state
        raw_obs_curr = data.observations[:, -2]
        state_curr, _ = self.network.encoder_image.forward(raw_obs_curr)

        # Actor
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(state_curr)
        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        # Critic
        critic_loss, critic_activations, critic_info = self.network.compute_critic_loss(
            data, state_curr
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
            "state": state_curr,
            **actor_activations,
            **critic_activations,
        }
        for feature_name, feature in feature_dict.items():
            info_dict[f"activation_norms/{feature_name}"] = feature.norm(dim=1).mean().item()

            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        # make decision
        action, action_info = self.select_action(global_step, obs)
        info_dict.update(action_info)

        return action, info_dict
