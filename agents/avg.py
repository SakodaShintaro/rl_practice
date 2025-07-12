"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse

import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss

from networks.backbone import AE, SmolVLABackbone
from networks.sac_tanh_policy_and_q import SacQ, SacTanhPolicy
from td_error_scaler import TDErrorScaler


class AvgAgent:
    def __init__(self, args: argparse.Namespace, observation_space, action_space) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 0

        if args.image_encoder == "ae":
            self.encoder_image = AE()
        elif args.image_encoder == "smolvla":
            self.encoder_image = SmolVLABackbone()
        else:
            raise ValueError(f"Unknown image encoder: {args.image_encoder}")
        self.encoder_image.to(self.device)
        self.cnn_dim = 4 * 12 * 12  # 576

        action_dim = np.prod(action_space.shape)
        self.action_space = action_space
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0
        self.actor = SacTanhPolicy(
            in_channels=self.cnn_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
        ).to(self.device)
        num_bins = 51
        self.Q = SacQ(
            in_channels=self.cnn_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
        ).to(self.device)

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        betas = [0.0, 0.999]
        weight_decay = 1e-5
        self.popt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.qopt = torch.optim.AdamW(
            self.Q.parameters(),
            lr=args.critic_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.gamma = args.gamma
        self.td_error_scaler = TDErrorScaler()
        self.G = 0

        self.use_eligibility_trace = args.use_eligibility_trace

        self.et_lambda = args.et_lambda
        with torch.no_grad():
            self.eligibility_traces_q = [
                torch.zeros_like(p, requires_grad=False) for p in self.Q.parameters()
            ]

        if args.without_entropy_term:
            self.log_alpha = None
        else:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item()
            self.log_alpha = torch.nn.Parameter(
                torch.zeros(1, requires_grad=True, device=self.device)
            )
            self.aopt = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

        self.hl_gauss_loss = HLGaussLoss(
            min_value=-30,
            max_value=+30,
            num_bins=num_bins,
            clamp_to_range=True,
        ).to(self.device)

        # Initialize state tracking
        self._prev_obs = None
        self._prev_action = None

    def select_action(self, global_step, obs):
        obs_tensor = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        obs_encoded = self.encoder_image.encode(obs_tensor)
        action, log_prob, _ = self.actor.get_action(obs_encoded)

        # Store current state and action for next update
        self._prev_obs = obs
        self._prev_action = action

        action = action[0].detach().cpu().numpy()
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        return action, {"selected_log_pi": log_prob[0].item()}

    def process_env_feedback(self, global_step, obs, action, reward, termination, truncation):
        # Compute log_prob for the previous action
        prev_obs_tensor = (
            torch.Tensor(self._prev_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        prev_obs_encoded = self.encoder_image.encode(prev_obs_tensor)
        _, prev_log_prob, _ = self.actor.get_action(prev_obs_encoded)

        # Update the actor and critic networks based on the observed transition
        obs_tensor = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        obs_encoded = self.encoder_image.encode(obs_tensor)
        next_obs = obs_encoded

        #### Q loss
        with torch.no_grad():
            alpha = self.log_alpha.exp().item() if self.log_alpha is not None else 0.0
            next_action, next_lprob, _ = self.actor.get_action(next_obs)
            next_q_dict = self.Q(next_obs, next_action)
            next_q_logit = next_q_dict["output"]
            next_q = self.hl_gauss_loss(next_q_logit).unsqueeze(-1)
            target_V = next_q - alpha * next_lprob

        #### Return scaling
        done = termination or truncation
        r_ent = reward - alpha * prev_log_prob.detach().item()
        self.G += r_ent
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.gamma, G=None)

        curr_q_dict = self.Q(prev_obs_encoded, self._prev_action.detach())
        curr_q_logit = curr_q_dict["output"]
        curr_q = self.hl_gauss_loss(curr_q_logit).unsqueeze(-1)
        delta = reward + (1 - done) * self.gamma * target_V - curr_q
        delta /= self.td_error_scaler.sigma

        # Policy loss
        curr_q_dict = self.Q(prev_obs_encoded, self._prev_action)
        curr_q_logit = curr_q_dict["output"]
        curr_q = self.hl_gauss_loss(curr_q_logit).unsqueeze(-1)
        ploss = alpha * prev_log_prob - curr_q
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        if self.use_eligibility_trace:
            curr_q_dict.backward()
            with torch.no_grad():
                for p, et in zip(self.Q.parameters(), self.eligibility_traces_q):
                    et.mul_(self.et_lambda * self.gamma).add_(p.grad.data)
                    p.grad.data = -2.0 * delta.item() * et
        else:
            qloss = delta**2
            qloss.backward()
        self.qopt.step()

        # alpha
        if self.log_alpha is None:
            alpha_loss = torch.Tensor([0.0])
        else:
            alpha_loss = (
                -self.log_alpha.exp() * (prev_log_prob.detach() + self.target_entropy)
            ).mean()
            self.aopt.zero_grad()
            alpha_loss.backward()
            self.aopt.step()

        self.steps += 1

        return {
            "delta": delta.item(),
            "q": curr_q.item(),
            "policy_loss": ploss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha,
        }

    def initialize_for_episode(self):
        """Initialize for new episode."""
        self._prev_obs = None
        self._prev_action = None

        for et in self.eligibility_traces_q:
            et.zero_()
