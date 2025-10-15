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
from replay_buffer import ReplayBuffer


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
        self.reward_scale = args.reward_scale

        # Use SAC's Network class
        self.network = Network(observation_space.shape, action_dim=self.action_dim, args=args).to(
            self.device
        )
        self.rnn_state = self.network.init_state().to(self.device)
        self.seq_len = args.seq_len

        self.rb = ReplayBuffer(
            size=args.buffer_size if hasattr(args, "buffer_size") else 10000,
            seq_len=self.seq_len + 1,
            obs_shape=observation_space.shape,
            rnn_state_shape=self.rnn_state.squeeze(1).shape,
            action_shape=(self.action_dim,),
            output_device=self.device,
            storage_device=torch.device(args.buffer_device),
        )

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

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    def initialize_for_episode(self) -> None:
        """Initialize for new episode."""
        pass

    @torch.inference_mode()
    def select_action(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        action_norm = np.linalg.norm(self.prev_action)
        train_reward = self.reward_scale * (reward - self.action_norm_penalty * action_norm)
        info_dict["action_norm"] = action_norm
        info_dict["train_reward"] = train_reward

        self.rb.add(
            torch.from_numpy(obs).to(self.device),
            train_reward,
            False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(self.prev_action).to(self.device),
            0.0,
            0.0,
        )

        latest_data = self.rb.get_latest(self.seq_len)

        output_enc, self.rnn_state = self.network.encoder.forward(
            latest_data.observations, latest_data.actions, latest_data.rewards, self.rnn_state
        )

        action, selected_log_pi = self.network.actor.get_action(output_enc)
        action = action[0].detach().cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        info_dict["selected_log_pi"] = selected_log_pi[0].item()

        self.prev_action = action
        return action, info_dict

    def step(
        self, global_step: int, obs: np.ndarray, reward: float, terminated: bool, truncated: bool
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # Sample data for training using ReplayBuffer
        data = self.rb.get_latest(self.seq_len + 1)

        # Encode current state
        obs_curr = data.observations[:, :-1]
        actions_curr = data.actions[:, :-1]
        rewards_curr = data.rewards[:, :-1]
        rnn_state_initial = (
            data.rnn_state[:, 0].permute(1, 0, 2).contiguous()
        )  # (1, B, hidden_size)

        curr_state, _ = self.network.encoder.forward(
            obs_curr, actions_curr, rewards_curr, rnn_state_initial
        )

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
            self.optimizer.step(delta, reset=terminated or truncated)
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
        action, action_info = self.select_action(global_step, obs, reward, terminated, truncated)
        info_dict.update(action_info)

        return action, info_dict
