import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss

from networks.backbone import (
    RecurrentEncoder,
    SimpleTransformerEncoder,
    SingleFrameEncoder,
    STTEncoder,
)
from networks.epona.flux_dit import FluxDiT
from networks.epona.layers import LinearEmbedder
from networks.policy_head import DiffusionPolicy
from networks.value_head import ActionValueHead


class Network(nn.Module):
    def __init__(self, observation_space_shape: list[int], action_dim: int, args):
        super(Network, self).__init__()
        self.gamma = 0.99
        self.num_bins = args.num_bins
        self.sparsity = args.sparsity
        self.seq_len = args.seq_len

        self.action_dim = action_dim
        self.predictor_step_num = args.predictor_step_num

        if args.encoder == "single_frame":
            self.encoder = SingleFrameEncoder(observation_space_shape)
        elif args.encoder == "stt":
            self.encoder = STTEncoder(
                observation_space_shape,
                seq_len=self.seq_len,
                n_layer=args.encoder_block_num,
                tempo_block_type=args.tempo_block_type,
                action_dim=action_dim,
            )
        elif args.encoder == "simple":
            self.encoder = SimpleTransformerEncoder(observation_space_shape, self.seq_len)
        elif args.encoder == "recurrent":
            self.encoder = RecurrentEncoder(observation_space_shape)
        else:
            raise ValueError(
                f"Unknown encoder: {args.encoder}. Only 'single_frame', 'stt', 'simple', and 'recurrent' are supported."
            )

        vae_hidden_dim = 4
        self.reward_encoder = LinearEmbedder(vae_hidden_dim)

        self.actor = DiffusionPolicy(
            state_dim=self.encoder.output_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.critic = ActionValueHead(
            in_channels=self.encoder.output_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=self.num_bins,
            sparsity=args.sparsity,
        )
        self.state_predictor = FluxDiT(
            in_channels=vae_hidden_dim,
            out_channels=vae_hidden_dim,
            vec_in_dim=action_dim,
            context_in_dim=vae_hidden_dim,
            hidden_size=args.predictor_hidden_dim,
            mlp_ratio=4.0,
            num_heads=8,
            depth_double_blocks=args.predictor_block_num,
            depth_single_blocks=args.predictor_block_num,
            axes_dim=[64],
            theta=10000,
            qkv_bias=True,
        )

        self.detach_actor = args.detach_actor
        self.detach_critic = args.detach_critic
        self.detach_predictor = args.detach_predictor

        if self.num_bins > 1:
            value_range = 60
            self.hl_gauss_loss = HLGaussLoss(
                min_value=-value_range,
                max_value=+value_range,
                num_bins=self.num_bins,
                clamp_to_range=True,
            )

    def compute_critic_loss(self, data, state_curr):
        if self.detach_critic:
            state_curr = state_curr.detach()
        with torch.no_grad():
            obs_next = data.observations[:, 1:]
            actions_next = data.actions[:, 1:]
            rewards_next = data.rewards[:, 1:]
            state_next = self.encoder.forward(obs_next, actions_next, rewards_next)
            next_state_actions, _ = self.actor.get_action(state_next)
            next_critic_output_dict = self.critic(state_next, next_state_actions)
            next_critic_value = next_critic_output_dict["output"]
            if self.num_bins > 1:
                next_critic_value = self.hl_gauss_loss(next_critic_value).view(-1)
            else:
                next_critic_value = next_critic_value.view(-1)
            curr_reward = data.rewards[:, -1].flatten()
            curr_continue = 1 - data.dones[:, -1].flatten()
            target_value = curr_reward + curr_continue * self.gamma * next_critic_value

        curr_critic_output_dict = self.critic(state_curr, data.actions[:, -1])

        if self.num_bins > 1:
            curr_critic_value = self.hl_gauss_loss(curr_critic_output_dict["output"]).view(-1)
            critic_loss = self.hl_gauss_loss(curr_critic_output_dict["output"], target_value)
        else:
            curr_critic_value = curr_critic_output_dict["output"].view(-1)
            critic_loss = F.mse_loss(curr_critic_value, target_value)

        delta = target_value - curr_critic_value

        activations_dict = {}

        info_dict = {
            "delta": delta.mean().item(),
            "critic_loss": critic_loss.item(),
            "curr_critic_value": curr_critic_value.mean().item(),
            "next_critic_value": next_critic_value.mean().item(),
            "target_value": target_value.mean().item(),
        }

        return critic_loss, activations_dict, info_dict

    def compute_actor_loss(self, state_curr):
        if self.detach_actor:
            state_curr = state_curr.detach()
        pi, log_pi = self.actor.get_action(state_curr)

        for param in self.critic.parameters():
            param.requires_grad_(False)

        critic_pi_output_dict = self.critic(state_curr, pi)
        critic_pi = critic_pi_output_dict["output"]
        if self.num_bins > 1:
            critic_pi = self.hl_gauss_loss(critic_pi).unsqueeze(-1)
        else:
            critic_pi = critic_pi.unsqueeze(-1)
        actor_loss = -critic_pi.mean()

        for param in self.critic.parameters():
            param.requires_grad_(True)

        # DACER2 loss (https://arxiv.org/abs/2505.23426)
        actions = pi.clone().detach()
        actions.requires_grad = True
        eps = 1e-4
        device = pi.device
        batch_size = pi.shape[0]
        t = (torch.rand((batch_size, 1), device=device)) * (1 - eps) + eps
        c = 0.4
        d = -1.8
        w_t = torch.exp(c * t + d)

        def calc_target(q_network, actions):
            q_output_dict = q_network(state_curr, actions)
            q_values = q_output_dict["output"]
            if self.num_bins > 1:
                q_values = self.hl_gauss_loss(q_values).unsqueeze(-1)
            else:
                q_values = q_values.unsqueeze(-1)
            q_grad = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=actions,
                create_graph=True,
            )[0]
            with torch.no_grad():
                target = (1 - t) / t * q_grad + 1 / t * actions
                target /= target.norm(dim=1, keepdim=True) + 1e-8
                return w_t * target

        target = calc_target(self.critic, actions)
        noise = torch.randn_like(actions)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        actor_output_dict = self.actor.forward(a_t, t.squeeze(1), state_curr)
        v = actor_output_dict["output"]
        dacer_loss = F.mse_loss(v, target)

        # Combine actor losses
        total_actor_loss = actor_loss + dacer_loss * 0.05

        activations_dict = {
            "actor": actor_output_dict["activation"],
            "critic": critic_pi_output_dict["activation"],
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "log_pi": log_pi.mean().item(),
        }

        return total_actor_loss, activations_dict, info_dict

    def compute_sequence_loss(self, data, state_curr):
        if self.detach_predictor:
            state_curr = state_curr.detach()

        # 最後のactionを取得 (actions[:, -1]がcurrent_stateに対応するaction)
        action_curr = data.actions[:, -1]  # (B, action_dim)

        # 次のstateをencodeする
        with torch.no_grad():
            last_obs = data.observations[:, -1]  # (B, C, H, W)
            target_state_next = self.encoder.ae.encode(last_obs).latents  # (B, C', H', W')
            B, C, H, W = target_state_next.shape
            target_state_next = target_state_next.flatten(2).permute(0, 2, 1)  # (B, H'*W', C')

        reward_next = data.rewards[:, -1]  # (B, 1)
        target_reward_next = self.reward_encoder.encode(reward_next)  # (B, C')
        x1 = torch.cat([target_state_next, target_reward_next], dim=1)  # (B, H'*W'+1, C')

        # Flow Matching for state prediction
        x0 = torch.randn_like(x1)
        shape_t = (x0.shape[0],) + (1,) * (len(x0.shape) - 1)
        t = torch.rand(shape_t, device=x1.device)

        # Sample from interpolation path for state
        xt = (1.0 - t) * x0 + t * x1

        # Convert tensors
        state_curr = state_curr.view(B, -1, C)

        # Predict velocity for state
        pred_dict = self.state_predictor.forward(xt, t, state_curr, action_curr)
        pred_vt = pred_dict["output"]  # (B, H*W, C)

        # Flow Matching loss
        vt = x1 - x0
        pred_loss = F.mse_loss(pred_vt, vt)

        activations_dict = {"state_predictor": pred_dict["activation"]}

        info_dict = {"seq_loss": pred_loss.item()}

        return pred_loss, activations_dict, info_dict

    @torch.inference_mode()
    def predict_next_state(self, state_curr, action_curr) -> tuple[np.ndarray, float]:
        device = state_curr.device
        B = state_curr.size(0)
        C = 4
        H = 12
        W = 12
        state_curr = state_curr.view(B, -1, C)
        noise_shape = (B, (H * W) + 1, C)
        normal = torch.distributions.Normal(
            torch.zeros(noise_shape, device=device),
            torch.ones(noise_shape, device=device),
        )
        next_hidden_state = normal.sample().to(device)
        next_hidden_state = torch.clamp(next_hidden_state, -3.0, 3.0)
        dt = 1.0 / self.predictor_step_num

        curr_time = torch.zeros((B), device=device)

        for _ in range(self.predictor_step_num):
            tmp_dict = self.state_predictor.forward(
                next_hidden_state, curr_time, state_curr, action_curr
            )
            vt = tmp_dict["output"]  # (B, H*W + 1, C)
            next_hidden_state = next_hidden_state + dt * vt
            curr_time += dt

        image_part = next_hidden_state[:, :-1, :]  # (B, H*W, C)
        image_part = image_part.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        next_image = self.encoder.decode(image_part)
        next_image = next_image.detach().cpu().numpy()
        next_image = next_image.squeeze(0)
        next_image = next_image.transpose(1, 2, 0)

        reward_part = next_hidden_state[:, -1, :]  # (B, 1, C)
        next_reward = self.reward_encoder.decode(reward_part)  # (B, 1)

        return next_image, next_reward.item()
