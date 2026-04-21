# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.nn import functional as F


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,  # (B,)
    dim: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:  # (B, dim)
    """Sinusoidal positional embedding for diffusion timestep."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, device=time.device, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(time.dtype)


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with adaptive modulation (scale, shift, gate) from conditioning signal."""

    def __init__(self, dim: int, eps: float, cond_dim: int) -> None:
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(
        self,
        x: torch.Tensor,  # (B, seq_len, dim)
        cond: torch.Tensor,  # (B, cond_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # normed: (B, seq_len, dim), gate: (B, 1, dim)
        dtype = x.dtype
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        modulation = self.dense(cond).unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed.float() * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: query from action tokens, key/value from VLM last_hidden.

    - AdaRMSNorm with timestep/bias conditioning (cond_dim == hidden_size)
    - SwiGLU MLP (Qwen3 style)
    - QK-norm
    """

    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        intermediate_size = hidden_size * 4

        self.input_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        self.post_attn_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # (B, tok_len, hidden_size)
        context: torch.Tensor,  # (B, ctx_len, context_dim)
        adarms_cond: torch.Tensor,  # (B, hidden_size)
    ) -> torch.Tensor:  # (B, tok_len, hidden_size)
        residual = x
        normed, gate1 = self.input_layernorm(x, adarms_cond)

        B, tok_len, _ = normed.shape
        ctx_len = context.shape[1]

        q = self.q_proj(normed).view(B, tok_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(B, ctx_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(B, ctx_len, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, tok_len, -1)
        x = residual + gate1 * self.o_proj(attn_out)

        residual2 = x
        normed2, gate2 = self.post_attn_layernorm(x, adarms_cond)
        x = residual2 + gate2 * self.down_proj(
            F.silu(self.gate_proj(normed2)) * self.up_proj(normed2)
        )
        return x


class CrossAttentionActionExpert(nn.Module):
    """Stack of cross-attention blocks for action denoising (GR00T / ABot_M0 style).

    Action tokens cross-attend to VLM last_hidden at every layer.
    AdaRMSNorm conditioned on the diffusion timestep embedding.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, context_dim, num_heads, head_dim, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,  # (B, horizon, hidden_size) action tokens (with time/pos injected outside)
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim) VLM last_hidden
        adarms_cond: torch.Tensor,  # (B, hidden_size) timestep embedding
    ) -> torch.Tensor:  # (B, horizon, hidden_size)
        for layer in self.layers:
            x = layer(x, context, adarms_cond)
        return self.norm(x)


class CrossAttentionActionValueHead(nn.Module):
    """Q(s, a) via cross-attention to VLM last_hidden.

    Dueling architecture:
      - V stream: learnable value_query → cross-attn → V(s)
      - A stream: [adv_query, action_tokens] → cross-attn → A(s, a)
    Output: Q = V + A in logit space (matches ActionValueHead interface).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        action_dim: int,
        horizon: int,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.action_dim = action_dim

        self.action_in_proj = nn.Linear(action_dim, hidden_size)
        self.action_pos_emb = nn.Parameter(torch.randn(1, horizon, hidden_size) * 0.02)

        self.v_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.a_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.cond_bias = nn.Parameter(torch.zeros(1, hidden_size))

        self.v_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, context_dim, num_heads, head_dim, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.a_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_size, context_dim, num_heads, head_dim, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.v_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.a_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        self.v_out = nn.Linear(hidden_size, num_bins)
        self.a_out = nn.Linear(hidden_size, num_bins)

    def _encode(
        self,
        tokens: torch.Tensor,  # (B, N, hidden_size)
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim)
        layers: nn.ModuleList,
        norm: nn.Module,
    ) -> torch.Tensor:  # (B, N, hidden_size)
        B = tokens.shape[0]
        cond = self.cond_bias.expand(B, -1)
        for layer in layers:
            tokens = layer(tokens, context, cond)
        return norm(tokens)

    def _v_stream(
        self,
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim)
    ) -> torch.Tensor:  # (B, hidden_size)  output of value_query token
        B = context.shape[0]
        v_q = self.v_query.expand(B, -1, -1)
        return self._encode(v_q, context, self.v_layers, self.v_norm)[:, 0]

    def _a_stream(
        self,
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim)
        action: torch.Tensor,  # (B, horizon, action_dim)
    ) -> torch.Tensor:  # (B, hidden_size)  output of adv_query token
        B = context.shape[0]
        action_tokens = self.action_in_proj(action) + self.action_pos_emb
        a_q = self.a_query.expand(B, -1, -1)
        tokens = torch.cat([a_q, action_tokens], dim=1)
        return self._encode(tokens, context, self.a_layers, self.a_norm)[:, 0]

    def forward(
        self,
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim) VLM last_hidden
        action: torch.Tensor,  # (B, horizon, action_dim)
    ) -> dict[str, torch.Tensor]:  # activation: (B, 2*hidden_size), output: (B, num_bins)
        v_feat = self._v_stream(context)
        a_feat = self._a_stream(context, action)
        v_out = self.v_out(v_feat)
        a_out = self.a_out(a_feat)
        return {
            "activation": torch.cat([v_feat, a_feat], dim=-1),
            "output": v_out + a_out,
        }

    def get_advantage(
        self,
        context: torch.Tensor,  # (B, vlm_seq_len, context_dim) VLM last_hidden
        action: torch.Tensor,  # (B, horizon, action_dim)
    ) -> dict[str, torch.Tensor]:  # activation: (B, hidden_size), output: (B, num_bins)
        a_feat = self._a_stream(context, action)
        return {
            "activation": a_feat,
            "output": self.a_out(a_feat),
        }
