# SPDX-License-Identifier: MIT
import torch
from torch import nn
from torch.nn import functional as F

from .value_head import _vp_sigma


def euler_denoise(
    noise: torch.Tensor,
    denoising_time: float,
    denoising_steps: int,
    predict_fn,
) -> torch.Tensor:
    """Euler denoising via flow matching (forward: 0 -> denoising_time).

    Args:
        predict_fn: callable(x_t, t) -> clean data x_1 (x-prediction).
            Velocity is derived as v = (x_1_pred - x_t) / (1 - t).

    Returns:
        tanh(x_T), same shape as noise
    """
    x_t = noise
    dt = denoising_time / denoising_steps
    time_val = 0.0
    for _ in range(denoising_steps):
        t = torch.full((noise.shape[0],), time_val, device=noise.device)
        pred = predict_fn(x_t, t)
        denom = max(1e-4, 1.0 - time_val)
        v = (pred - x_t) / denom
        x_t = x_t + dt * v
        time_val += dt
    return torch.tanh(x_t)


def compute_actor_loss_with_dacer(
    state: torch.Tensor,
    action: torch.Tensor,
    value_head,
    hl_gauss_loss,
    num_bins: int,
    dacer_loss_weight: float,
    predict_fn,
) -> tuple[torch.Tensor, dict, dict]:
    """Advantage-based actor loss + DACER2 loss (https://arxiv.org/abs/2505.23426).

    Args:
        state: (B, state_dim)
        action: (B, horizon, action_dim)
        value_head: Q-function network
        hl_gauss_loss: HL-Gauss loss (used when num_bins > 1)
        num_bins: number of distributional bins
        dacer_loss_weight: weight for DACER2 loss
        predict_fn: callable(a_t: (B, H, A), t: (B,)) -> (B, H, A) returning clean x_1 (x-prediction)

    Returns:
        (total_loss, advantage_dict, info_dict)
    """
    # Advantage-based loss
    for param in value_head.parameters():
        param.requires_grad_(False)

    advantage_dict = value_head.get_advantage(state, action)
    advantage = advantage_dict["output"]
    if num_bins > 1:
        advantage = hl_gauss_loss(advantage)
    advantage = advantage.view(-1, 1)
    actor_loss = -advantage.mean()

    for param in value_head.parameters():
        param.requires_grad_(True)

    # DACER2 loss
    B, horizon, action_dim = action.shape
    device = action.device
    action_flat = action.view(B, -1)
    actions = action_flat.clone().detach()
    actions.requires_grad = True
    eps = 1e-4
    t = (torch.rand((B, 1), device=device)) * (1 - eps) + eps
    c = 0.4
    d = -1.8
    w_t = torch.exp(c * t + d)

    actions_chunk = actions.view(B, horizon, action_dim)
    q_output_dict = value_head(state, actions_chunk)
    q_values = q_output_dict["output"]
    if num_bins > 1:
        q_values = hl_gauss_loss(q_values).unsqueeze(-1)
    else:
        q_values = q_values.unsqueeze(-1)
    q_grad = torch.autograd.grad(
        outputs=q_values.sum(),
        inputs=actions,
        create_graph=True,
    )[0]

    noise = torch.randn_like(actions)
    noise = torch.clamp(noise, -3.0, 3.0)
    a_t = (1.0 - t) * noise + t * actions

    with torch.no_grad():
        # DACER velocity target
        v_target = (1 - t) / t * q_grad + 1 / t * actions
        v_target /= v_target.norm(dim=1, keepdim=True) + 1e-8
        v_target = w_t * v_target
        # Convert to x-space: x_1_target = a_t + (1 - t) * v_target
        target = a_t + (1 - t) * v_target

    a_t_chunk = a_t.view(B, horizon, action_dim)
    pred_chunk = predict_fn(a_t_chunk, t.squeeze(1))
    pred = pred_chunk.view(B, -1)
    dacer_loss = F.mse_loss(pred, target)

    total_loss = actor_loss + dacer_loss * dacer_loss_weight

    info_dict = {
        "actor_loss": actor_loss.item(),
        "dacer_loss": dacer_loss.item(),
        "advantage": advantage.mean().item(),
    }

    return total_loss, advantage_dict, info_dict


def compute_actor_loss_with_energy_flow(
    state: torch.Tensor,
    action: torch.Tensor,
    value_head,
    hl_gauss_loss: nn.Module | None,
    num_bins: int,
) -> tuple[torch.Tensor, dict, dict]:
    """Denoising score matching loss applied to the action-value head used as
    a scalar energy field (``E = -Q``).

    Following EnergyFlow (Eq. 15 with α = 1), the score of the buffer-action
    distribution is matched against +∇_a Q at randomly sampled noise levels.
    With λ(t) = σ²(t), the σ²-weighted DSM objective collapses into

        L = E_{t, a_0, ε} ||σ(t) · ∇_a Q(a_t, s, t) + ε||²,

    where a_t = a_0 + σ(t)·ε with the geometric VE schedule
    σ(t) = σ_min^(1 - t/T) · σ_max^(t/T). The clean buffer action ``action``
    serves as a_0 (the role of expert demonstrations in the paper).

    Args:
        state: (B, state_dim) state embedding.
        action: (B, horizon, action_dim) clean buffer action chunk.
        value_head: ActionValueHead with use_energy_flow=True.
        hl_gauss_loss: HL-Gauss readout when num_bins > 1, else None.
        num_bins: distributional bin count of the Q head.

    Returns:
        (loss, activations_dict, info_dict)
    """
    assert getattr(value_head, "use_energy_flow", False), (
        "compute_actor_loss_with_energy_flow requires value_head.use_energy_flow=True"
    )
    B = action.size(0)
    device = action.device

    sigma_min = value_head.energy_sigma_min
    sigma_max = value_head.energy_sigma_max
    sigma_t_max = value_head.energy_sigma_t_max
    eps = 1e-4

    # Diffusion timestep t ~ U(eps, T).
    t = torch.rand(B, device=device) * (sigma_t_max - eps) + eps
    sigma_t = _vp_sigma(t, sigma_min, sigma_max, sigma_t_max)  # (B,)
    sigma_t_view = sigma_t.view(B, 1, 1)

    # Forward perturbation: a_t = a_0 + σ(t) ε.
    noise = torch.randn_like(action)
    a_t = (action + sigma_t_view * noise).detach().requires_grad_(True)

    # Score = ∇_a Q via autograd through the advantage stream.
    grad = value_head.compute_score(state, a_t, t, hl_gauss_loss, create_graph=True)

    # || σ(t) · ∇_a Q + ε ||² (per-element MSE).
    diff = sigma_t_view * grad + noise
    loss = diff.pow(2).mean()

    info_dict = {
        "actor_loss": loss.item(),
        "dacer_loss": 0.0,
        "energy_flow_score_norm": grad.detach().pow(2).mean().sqrt().item(),
        "energy_flow_sigma_mean": sigma_t.mean().item(),
    }
    activations_dict: dict = {}

    return loss, activations_dict, info_dict
