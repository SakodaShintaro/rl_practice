# SPDX-License-Identifier: MIT
import math

import torch
from hl_gauss_pytorch import HLGaussLoss
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from .blocks import SimbaBlock
from .image_processor import ImageProcessor
from .sparse_utils import apply_one_shot_pruning


class _MishResidualBlock(nn.Module):
    """C^2-differentiable residual MLP block used as energy-flow backbone."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels, elementwise_affine=False),
            nn.Linear(channels, channels),
            nn.Mish(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _EnergyTimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 128) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def _vp_sigma(
    t: torch.Tensor | float,
    sigma_min: float,
    sigma_max: float,
    sigma_t_max: float,
) -> torch.Tensor | float:
    """Geometric VE noise schedule: sigma(t) = sigma_min^(1 - t/T) * sigma_max^(t/T)."""
    if isinstance(t, torch.Tensor):
        return sigma_min ** (1.0 - t / sigma_t_max) * sigma_max ** (t / sigma_t_max)
    return sigma_min ** (1.0 - t / sigma_t_max) * sigma_max ** (t / sigma_t_max)


def _dsigma2_dt(
    t: torch.Tensor | float,
    sigma_min: float,
    sigma_max: float,
    sigma_t_max: float,
) -> torch.Tensor | float:
    """d/dt of sigma(t)^2 for the geometric VE schedule."""
    log_ratio = math.log(sigma_max) - math.log(sigma_min)
    sigma2 = _vp_sigma(t, sigma_min, sigma_max, sigma_t_max) ** 2
    return sigma2 * (2.0 / sigma_t_max) * log_ratio


def maybe_update_hl_gauss_range(
    module: nn.Module,
    target_value: torch.Tensor,
) -> None:
    observed_max = target_value.abs().max().item()
    if observed_max <= module.value_range:
        return
    module.value_range = observed_max
    device = module.hl_gauss_loss.support.device
    module.hl_gauss_loss = HLGaussLoss(
        min_value=-module.value_range,
        max_value=+module.value_range,
        num_bins=module.num_bins,
        clamp_to_range=True,
    ).to(device)


def weights_init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias, 0)


class StateValueHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_channels, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result_dict = {}

        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict


class ActionValueHead(nn.Module):
    """Dueling Architecture: Q(s,a) = V(s) + A(s,a).

    When ``use_energy_flow=True`` the advantage stream is also conditioned on a
    diffusion timestep t, uses Mish activations (C^2-differentiable as required
    by EnergyFlow score matching), and applies spectral normalization to its
    output linear layer. In that mode Q(s,a,t) plays the role of a soft energy
    field: its action gradient defines a denoising score, and probability-flow
    ODE rollouts on that score generate actions (Algorithm 2 of "Recovering
    Hidden Reward in Diffusion-Based Policies").
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        block_num: int,
        num_bins: int,
        sparsity: float,
        use_energy_flow: bool,
        energy_sigma_min: float,
        energy_sigma_max: float,
        energy_sigma_t_max: float,
        energy_ode_steps: int,
        energy_ode_endpoint: float,
        energy_time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        total_action_dim = action_dim * horizon
        self.num_bins = num_bins
        self.use_energy_flow = use_energy_flow
        self.energy_sigma_min = energy_sigma_min
        self.energy_sigma_max = energy_sigma_max
        self.energy_sigma_t_max = energy_sigma_t_max
        self.energy_ode_steps = energy_ode_steps
        self.energy_ode_endpoint = energy_ode_endpoint

        # Value stream: V(s) - depends only on state
        self.v_fc_in = nn.Linear(in_channels, hidden_dim)
        self.v_fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.v_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.v_fc_out = nn.Linear(hidden_dim, num_bins)

        # Advantage stream: A(s,a[,t])
        if use_energy_flow:
            self.t_embedder = _EnergyTimeEmbedder(energy_time_embed_dim)
            adv_in_dim = in_channels + total_action_dim + energy_time_embed_dim
            block_cls: type[nn.Module] = _MishResidualBlock
        else:
            self.t_embedder = None
            adv_in_dim = in_channels + total_action_dim
            block_cls = SimbaBlock

        self.a_fc_in = nn.Linear(adv_in_dim, hidden_dim)
        self.a_fc_mid = nn.Sequential(*[block_cls(hidden_dim) for _ in range(block_num)])
        self.a_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.a_fc_out = nn.Linear(hidden_dim, num_bins)

        self.apply(weights_init_)

        # Spectral normalization on the scalar energy readout (paper §C.2).
        # Apply after weight init since spectral_norm replaces the weight with
        # a parametrization that cannot be assigned to in-place.
        if use_energy_flow:
            self.a_fc_out = spectral_norm(self.a_fc_out)

        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def _build_advantage_input(
        self,
        x: torch.Tensor,
        a_flat: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_energy_flow:
            assert self.t_embedder is not None
            if t is None:
                t = torch.zeros(a_flat.size(0), device=a_flat.device)
            t_emb = self.t_embedder(t)
            return torch.cat([x, a_flat, t_emb], dim=1)
        return torch.cat([x, a_flat], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
            t: diffusion timestep (B,) or None. Required only when
                ``use_energy_flow`` is True; ignored otherwise.
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        # Value stream: V(s)
        v = self.v_fc_in(x)
        v = self.v_fc_mid(v)
        v = self.v_norm(v)
        v_out = self.v_fc_out(v)  # (B, num_bins)

        # Advantage stream: A(s,a[,t])
        xa = self._build_advantage_input(x, a_flat, t)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        adv_out = self.a_fc_out(adv)  # (B, num_bins)

        result_dict["activation"] = torch.cat([v, adv], dim=1)

        # Q(s,a) = V(s) + A(s,a) in logit space
        output = v_out + adv_out
        result_dict["output"] = output

        return result_dict

    def get_advantage(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: state embedding (B, state_dim)
            a: action chunk (B, horizon, action_dim)
            t: diffusion timestep (B,) or None.
        """
        result_dict = {}
        bs = a.size(0)
        a_flat = a.view(bs, -1)  # (B, horizon * action_dim)

        xa = self._build_advantage_input(x, a_flat, t)
        adv = self.a_fc_in(xa)
        adv = self.a_fc_mid(adv)
        adv = self.a_norm(adv)
        result_dict["activation"] = adv
        adv_out = self.a_fc_out(adv)  # (B, num_bins)
        result_dict["output"] = adv_out

        return result_dict

    ###########################
    # Energy-flow extensions  #
    ###########################

    def _q_scalar(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        hl_gauss_loss: nn.Module | None,
    ) -> torch.Tensor:
        """Forward returning Q as a scalar (B,) tensor (differentiable in a)."""
        # Only the advantage stream depends on a, so it suffices for ∇_a Q.
        out = self.get_advantage(x, a, t)["output"]  # (B, num_bins) or (B, 1)
        if self.num_bins > 1:
            assert hl_gauss_loss is not None
            return hl_gauss_loss(out)  # (B,)
        return out.view(-1)

    def compute_score(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
        hl_gauss_loss: nn.Module | None,
        create_graph: bool,
    ) -> torch.Tensor:
        """Return ∇_a Q(s, a, t).

        With Q ∝ log π_E (max-entropy), this is the score of the expert action
        distribution. Works in eval (create_graph=False) and train.
        """
        if not a.requires_grad:
            a = a.detach().requires_grad_(True)
        with torch.enable_grad():
            q = self._q_scalar(x, a, t, hl_gauss_loss)
            grad = torch.autograd.grad(q.sum(), a, create_graph=create_graph)[0]
        return grad

    def get_action(
        self,
        x: torch.Tensor,
        hl_gauss_loss: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action chunks via probability-flow ODE on +∇_a Q.

        Mirrors Algorithm 2 of "Recovering Hidden Reward in Diffusion-Based
        Policies" with the sign convention E = -Q, so the update direction is
        +∇_a Q (gradient ascent on Q):
            a_{k+1} = a_k + (Δt / 2) · d[σ²]/dt|_{t_k} · ∇_a Q(a_k, s, t_k)
        Returns ``(action, dummy_log_p)`` to match the policy_head interface.
        """
        assert self.use_energy_flow, "ActionValueHead.get_action requires use_energy_flow=True"
        bs = x.size(0)
        device = x.device
        T = self.energy_sigma_t_max
        gamma = self.energy_ode_endpoint
        K = self.energy_ode_steps
        sigma_T = float(
            _vp_sigma(T, self.energy_sigma_min, self.energy_sigma_max, T)
        )
        a = sigma_T * torch.randn(bs, self.horizon, self.action_dim, device=device)
        dt = (T - gamma) / K
        for k in range(K):
            t_val = T - k * dt
            t_tensor = torch.full((bs,), t_val, device=device)
            grad = self.compute_score(x, a, t_tensor, hl_gauss_loss, create_graph=False)
            dsig2 = float(
                _dsigma2_dt(t_val, self.energy_sigma_min, self.energy_sigma_max, T)
            )
            a = a.detach() + dt * 0.5 * dsig2 * grad.detach()
        a = torch.tanh(a)
        dummy_log_p = torch.zeros((bs, 1), device=device)
        return a, dummy_log_p


class SeparateCritic(nn.Module):
    """Separate critic network with its own ImageProcessor and MLP."""

    def __init__(
        self,
        observation_space_shape: tuple[int],
        hidden_dim: int,
        block_num: int,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.image_processor = ImageProcessor(observation_space_shape)
        output_shape = self.image_processor.output_shape
        flat_dim = output_shape[0] * output_shape[1] * output_shape[2]

        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(flat_dim, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, num_bins)
        self.apply(weights_init_)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for separate critic.

        Args:
            obs: (B, C, H, W) observation image

        Returns:
            Dictionary with 'output' and 'activation' keys
        """
        result_dict = {}

        x = self.image_processor.encode(obs)
        x = self.flatten(x)
        x = self.fc_in(x)
        x = self.fc_mid(x)
        x = self.norm(x)
        result_dict["activation"] = x

        output = self.fc_out(x)
        result_dict["output"] = output

        return result_dict
