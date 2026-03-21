# SPDX-License-Identifier: MIT
import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. q/k: (B, heads, seq, head_dim), cos/sin: (B, seq, rope_dim).

    Supports partial rotary embedding: if rope_dim < head_dim, RoPE is applied
    only to the first rope_dim dimensions and the rest pass through unchanged.
    """
    cos = cos.unsqueeze(1)  # (B, 1, seq, rope_dim)
    sin = sin.unsqueeze(1)
    rope_dim = cos.shape[-1]
    if rope_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
        k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dim: int, min_period: float, max_period: float
) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timestep. time: (B,), returns (B, dim)."""
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        modulation = self.dense(cond)
        # cond is (B, cond_dim), expand for (B, seq, dim)
        modulation = modulation.unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed.float() * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class ActionExpertLayer(nn.Module):
    """Single transformer layer: combined self+cross attention to VLM, then MLP.

    Uses GQA matching VLM config. Cross-attention uses VLM's own k/v projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_kv_heads
        intermediate_size = hidden_size * 4

        # Pre-attention norm (adaptive)
        self.input_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        # Self-attention projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # QK-norm (like Qwen3)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        # Post-attention norm (adaptive)
        self.post_attn_layernorm = AdaptiveRMSNorm(hidden_size, rms_norm_eps, hidden_size)

        # MLP (SwiGLU like Qwen3)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA. (B, num_kv_heads, S, head_dim) -> (B, num_heads, S, head_dim)"""
        return kv.repeat_interleave(self.num_kv_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, action_len, hidden_size)
        vlm_k: torch.Tensor,  # (B, num_kv_heads, vlm_len, head_dim) with RoPE
        vlm_v: torch.Tensor,  # (B, num_kv_heads, vlm_len, head_dim)
        cos: torch.Tensor,  # (B, action_len, head_dim)
        sin: torch.Tensor,  # (B, action_len, head_dim)
        adarms_cond: torch.Tensor,  # (B, cond_dim)
    ) -> torch.Tensor:
        residual = hidden_states
        normed, gate1 = self.input_layernorm(hidden_states, adarms_cond)

        B, action_len, _ = normed.shape

        # Expert Q/K/V
        q = self.q_proj(normed).view(B, action_len, self.num_heads, self.head_dim)
        k_self = self.k_proj(normed).view(B, action_len, self.num_kv_heads, self.head_dim)
        v_self = self.v_proj(normed).view(B, action_len, self.num_kv_heads, self.head_dim)

        # QK-norm
        q = self.q_norm(q)
        k_self = self.k_norm(k_self)

        q = q.transpose(1, 2)  # (B, num_heads, action_len, head_dim)
        k_self = k_self.transpose(1, 2)  # (B, num_kv_heads, action_len, head_dim)
        v_self = v_self.transpose(1, 2)  # (B, num_kv_heads, action_len, head_dim)

        # RoPE on Q and self K to match VLM K
        q, k_self = apply_rotary_pos_emb(q, k_self, cos, sin)

        # Concat VLM K/V with self K/V: [vlm_tokens, action_tokens]
        k = torch.cat([vlm_k, k_self], dim=2)  # (B, num_kv_heads, vlm_len+action_len, head_dim)
        v = torch.cat([vlm_v, v_self], dim=2)

        # GQA: expand KV heads
        k = self._repeat_kv(k)  # (B, num_heads, vlm_len+action_len, head_dim)
        v = self._repeat_kv(v)  # (B, num_heads, vlm_len+action_len, head_dim)

        # Scaled dot-product attention (no mask: action tokens attend to everything)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, action_len, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, action_len, -1)

        # Output projection + gated residual
        hidden_states = residual + gate1 * self.o_proj(attn_out)  # (B, action_len, hidden_size)

        # MLP
        residual2 = hidden_states
        normed2, gate2 = self.post_attn_layernorm(hidden_states, adarms_cond)
        hidden_states = residual2 + gate2 * self.down_proj(
            F.silu(self.gate_proj(normed2)) * self.up_proj(normed2)
        )

        return hidden_states


class ActionExpert(nn.Module):
    """Stack of ActionExpertLayers that cross-attend to VLM hidden states."""

    def __init__(
        self,
        num_layers: int,
        expert_hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rotary_emb: nn.Module,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(
            [
                ActionExpertLayer(
                    expert_hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(expert_hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_kv_list: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_seq_len: int,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        B, action_len, _ = hidden_states.shape

        # RoPE positions for action tokens: right after VLM sequence
        action_pos = torch.arange(
            vlm_seq_len, vlm_seq_len + action_len, device=hidden_states.device
        )
        action_pos_ids = action_pos.unsqueeze(0).expand(B, -1)  # (B, action_len)
        cos, sin = self.rotary_emb(hidden_states, position_ids=action_pos_ids)

        for j in range(self.num_layers):
            k, v = vlm_kv_list[j]
            hidden_states = self.layers[j](hidden_states, k, v, cos, sin, adarms_cond)
        return self.norm(hidden_states)
