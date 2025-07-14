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

        # Use SAC's Network class
        seq_len = 2
        enable_sequence_modeling = False  # AVG doesn't use sequence modeling
        self.network = Network(
            action_dim=self.action_dim,
            seq_len=seq_len,
            args=args,
            enable_sequence_modeling=enable_sequence_modeling,
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
            "critic": StatisticalMetricsComputer(),
            "actor": StatisticalMetricsComputer(),
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

    def select_action(self, global_step, obs):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        obs_encoded = self.network.encoder_image.encode(obs_tensor)
        action, log_prob = self.network.actor.get_action(obs_encoded)

        # Store current state and action for next update
        self._prev_obs = obs_encoded
        self._prev_action = action

        action = action[0].detach().cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        return action, {"selected_log_pi": log_prob[0].item()}

    def process_env_feedback(self, global_step, next_obs, action, reward, termination, truncation):
        info_dict = {}

        reward /= 10.0

        done = termination or truncation

        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(self.device)
        self._next_obs = self.network.encoder_image.encode(next_obs)

        observations = torch.stack([self._prev_obs, self._next_obs], dim=1).to(self.device)

        actions = torch.stack(
            [
                self._prev_action.squeeze(0),
                torch.Tensor(action).to(self.device),
            ],
            dim=0,
        ).unsqueeze(0)  # [batch_size=1, seq_len=2, action_dim]

        rewards = torch.tensor([[reward, reward]], device=self.device, dtype=torch.float32)
        dones = torch.tensor([[done, done]], device=self.device, dtype=torch.float32)

        data = ReplayBufferData(
            observations=observations, actions=actions, rewards=rewards, dones=dones
        )

        # Encode current state
        state_curr = data.observations[:, -2]
        qf_loss, _, qf_activations, qf_info = self.network.compute_critic_loss(data, state_curr)
        actor_loss, actor_activations, actor_info = self.network.compute_actor_loss(state_curr)

        feature_dict = {
            "state": state_curr,
            **actor_activations,
            **qf_activations,
        }

        # Combine losses (no sequence modeling for AVG)
        loss = actor_loss + qf_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        grad_metrics = {
            key: compute_gradient_norm(model) for key, model in self.monitoring_targets.items()
        }
        param_metrics = {
            key: compute_parameter_norm(model) for key, model in self.monitoring_targets.items()
        }
        activation_norms = {
            key: value.norm(dim=1).mean().item() for key, value in feature_dict.items()
        }

        if self.use_eligibility_trace:
            delta = qf_info["delta"]
            self.optimizer.step(delta, reset=done)
        else:
            self.optimizer.step()

        # Combine info from both losses
        for key, value in qf_info.items():
            info_dict[f"losses/{key}"] = value
        for key, value in actor_info.items():
            info_dict[f"losses/{key}"] = value

        # Add gradient norm metrics
        for key, value in grad_metrics.items():
            info_dict[f"gradients/{key}"] = value

        # Add parameter norm metrics
        for key, value in param_metrics.items():
            info_dict[f"parameters/{key}"] = value

        # Add activation norms
        for key, value in activation_norms.items():
            info_dict[f"activation_norms/{key}"] = value

        # Trigger statistical metrics computation
        for feature_name, feature in feature_dict.items():
            result_dict = self.metrics_computers[feature_name](feature)
            for key, value in result_dict.items():
                info_dict[f"{key}/{feature_name}"] = value

        return info_dict

    def initialize_for_episode(self):
        """Initialize for new episode."""
        self._prev_obs = None
        self._prev_action = None
