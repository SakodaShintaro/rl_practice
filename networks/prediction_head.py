import numpy as np
import torch
import torch.nn as nn

from .epona.flux_dit import FluxDiT
from .epona.layers import LinearEmbedder


class StatePredictionHead(nn.Module):
    def __init__(
        self,
        hidden_image_dim: int,
        action_dim: int,
        predictor_hidden_dim: int,
        predictor_block_num: int,
    ):
        super().__init__()
        self.reward_encoder = LinearEmbedder(hidden_image_dim)
        self.state_predictor = FluxDiT(
            in_channels=hidden_image_dim,
            out_channels=hidden_image_dim,
            vec_in_dim=action_dim,
            context_in_dim=hidden_image_dim,
            hidden_size=predictor_hidden_dim,
            mlp_ratio=4.0,
            num_heads=8,
            depth_double_blocks=predictor_block_num,
            depth_single_blocks=predictor_block_num,
            axes_dim=[64],
            theta=10000,
            qkv_bias=True,
        )

    @torch.inference_mode()
    def predict_next_state(
        self,
        state_curr: torch.Tensor,
        action_curr: torch.Tensor,
        image_processor,
        observation_space_shape: tuple[int, ...],
        predictor_step_num: int,
        disable_state_predictor: bool,
    ) -> tuple[np.ndarray, float]:
        if disable_state_predictor:
            H = observation_space_shape[1]
            W = observation_space_shape[2]
            next_image = np.zeros((H, W, 3), dtype=np.float32)
            next_reward = 0.0
            return next_image, next_reward

        device = state_curr.device
        B = state_curr.size(0)
        C = image_processor.output_shape[0]
        H = image_processor.output_shape[1]
        W = image_processor.output_shape[2]
        state_curr = state_curr.view(B, -1, C)
        noise_shape = (B, (H * W) + 1, C)
        normal = torch.distributions.Normal(
            torch.zeros(noise_shape, device=device),
            torch.ones(noise_shape, device=device),
        )
        next_hidden_state = normal.sample().to(device)
        next_hidden_state = torch.clamp(next_hidden_state, -3.0, 3.0)
        dt = 1.0 / predictor_step_num

        curr_time = torch.zeros((B), device=device)

        for _ in range(predictor_step_num):
            tmp_dict = self.state_predictor.forward(
                next_hidden_state, curr_time, state_curr, action_curr
            )
            vt = tmp_dict["output"]
            next_hidden_state = next_hidden_state + dt * vt
            curr_time += dt

        image_part = next_hidden_state[:, :-1, :]
        image_part = image_part.permute(0, 2, 1).view(B, C, H, W)
        next_image = image_processor.decode(image_part)
        next_image = next_image.detach().cpu().numpy()
        next_image = next_image.squeeze(0)
        next_image = next_image.transpose(1, 2, 0)

        reward_part = next_hidden_state[:, -1, :]
        next_reward = self.reward_encoder.decode(reward_part)

        return next_image, next_reward.item()
