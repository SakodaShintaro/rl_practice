# Copyright (c) The mmMamba team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fused_norm_gate import FusedRMSNormSwishGate

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from timm.models.layers import DropPath

compute_ARank = False # [ARank] Set this to True to compute attention rank

from .configuration_mmMamba_embedding import mmMambaEmbeddingConfig

from .configuration_mmMamba import mmMambaConfig

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

import torch.nn.functional as F

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "mmMambaEmbeddingConfig"

flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None
def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import flash_attn_func as _flash_attn_func, flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input as _pad_input, index_first_axis as _index_first_axis, unpad_input as _unpad_input
        flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
        pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    except ImportError:
        raise ImportError("flash_attn is not installed.")

_import_flash_attn()

def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->mmMamba
class mmMambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        mmMambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.model.llama.modeling_llama.LlamaRotaryEmbedding with Llama->mmMamba
class mmMambaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        #self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        #self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=torch.float32)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.model.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->mmMamba
class mmMambaLinearScalingRotaryEmbedding(mmMambaRotaryEmbedding):
    """mmMambaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.model.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->mmMamba
class mmMambaDynamicNTKScalingRotaryEmbedding(mmMambaRotaryEmbedding):
    """mmMambaRotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class mmMambaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj


# Copied from transformers.model.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None,  :].expand(batch, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, head_dim)

class MHA_LM(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: mmMambaEmbeddingConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.rotary_emb_dim = self.head_dim
        self.softmax_scale = None
        self.causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        assert RotaryEmbedding is not None, "rotary requires flash_attn to be installed"
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            base=self.config.rope_theta,
            interleaved=False,
            device=self.wo.weight.device, 
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
        kv_cache = kv_cache[:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        assert flash_attn_with_kvcache is not None, "flash_attn must be installed"
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        )
        return context 

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            k, v = kv.unbind(dim=-3)
            #k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_key_value_heads)
            #v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_key_value_heads)
            attn_output = flash_attn_func(
                q, k, v, 0.0, softmax_scale=None, causal=self.causal
            )
            return attn_output
        else:
            batch = q.shape[0]
            kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,#------------------------------------------------------------------------
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                hidden_states.shape[0], inference_params.max_seqlen, dtype=hidden_states.dtype
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )

        bsz, q_len, _ = hidden_states.size()
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None

        qkv = self.wqkv(hidden_states)
        qkv = rearrange(
            qkv,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        q = qkv[..., : self.num_key_value_groups, :]
        q = rearrange(q, "b q h gs d -> b q (h gs) d")
        kv = qkv[..., self.num_key_value_groups:, :].transpose(2,3)
        #kv = rearrange(kv, "b q h gs d -> b q (h gs) d")
        #kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset[:bsz,...], max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                k, v = kv.unbind(dim=-3)
                k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_key_value_heads)
                v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_key_value_heads)
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=None
                ).transpose(1, 2)
            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        out = self.wo(context)
        return out
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.wo.weight.dtype if dtype is None else dtype
        device = self.wo.weight.device
        kv_cache = torch.empty(
            batch_size, max_seqlen, 2, self.num_key_value_heads, self.head_dim, dtype=dtype, device=device,
        )
        return kv_cache, None

class Mamba2_LM(nn.Module):
    """
    LoLCATs attention implementation initialized from a 
    `LlamaAttention` or `MistralAttention` object (base_attn)

    Most of the arguments are directly tied to argparse args
    - For now we don't support padding.
    """
    def __init__(self, config: mmMambaConfig, layer_idx: Optional[int] = None,
                 elementwise_affine: Optional[bool] = True,
                 norm_eps: float = 1e-5,
                 ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx
        self.bias = False
        self.chunk_size = 128
        conv_bias = True
        self.conv_bias = conv_bias
        self.d_conv = 2
        self.activation="silu"
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.wvkqgdt = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads + self.num_heads) * self.head_dim + self.num_heads,
            bias=self.bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.device = self.wvkqgdt.weight.device
        self.dtype = self.wvkqgdt.weight.dtype

        conv_dim = self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim
            
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv - 1,
            device=self.device, 
            dtype=self.dtype
        )
        with torch.no_grad():
            self.conv1d.weight.zero_()  
            self.conv1d.weight[:, 0, 1] = 1 
            self.conv1d.bias.zero_()  

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")
        
        self.g_norm_swish_gate = FusedRMSNormSwishGate(hidden_size=self.head_dim, elementwise_affine=elementwise_affine, eps=norm_eps).to(self.dtype).to(self.device)

        dt = torch.exp(
            torch.rand(self.num_heads, dtype=self.dtype, device=self.device) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=0.001)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        A_log_bias = torch.zeros(self.num_heads, dtype=self.dtype, device=self.device)
        self.A_log_bias = nn.Parameter(A_log_bias)
        self.A_log_bias._no_weight_decay = True

    def forward(self,
                hidden_states: torch.Tensor,
                inference_params = None,
                output_attentions: bool = False,
                use_cache: bool = True,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        hidden_states = hidden_states.to(self.dtype)
        vkqgdt = self.wvkqgdt(hidden_states)
        vkq, g, dt = torch.split(
                vkqgdt,
                [
                    (2*self.num_key_value_heads+self.num_heads) * self.head_dim,
                    self.num_heads * self.head_dim,
                    self.num_heads,
                ],
                dim=2,
            )
        batch, seqlen, _ = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        conv_state = conv_state[:batch, ...]
        ssm_state = ssm_state[:batch, ...]
        
        if use_cache and inference_params.seqlen_offset==0:
            vkq, new_conv_states = causal_conv1d_fn(
                vkq.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                initial_states=None,
                return_final_states=True,
                activation=None if self.activation == "identity" else self.activation,
            )

            v, k, q = torch.split(
                vkq,
                [
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                    self.num_heads * self.head_dim,
                ],
                dim=1,
            )

            v = rearrange(v, "b (h n) l -> b h l n", h=self.num_key_value_heads)
            k = rearrange(k, "b (h n) l -> b h l n", h=self.num_key_value_heads)
            q = rearrange(q, "b (h n) l -> b l h n", h=self.num_heads)
            k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
            v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
                
            A = -torch.exp(self.A_log_bias.float())

            y, new_ssm_states = mamba_chunk_scan_combined(
                x = v,
                #x = v / F.softplus(A_log).to(v.dtype).unsqueeze(-1),
                dt=dt,
                dt_softplus=True,
                A=A,
                B=k,
                C=q,
                chunk_size=self.chunk_size,
                dt_bias=self.dt_bias,
                initial_states=None, # currently not supported by mamba_ssm.utils.generation
                return_final_states=True,
            )

            conv_state.copy_(new_conv_states)
            ssm_state.copy_(new_ssm_states)

        elif use_cache and inference_params.seqlen_offset>0:
            
            vkq = causal_conv1d_update(
                vkq.transpose(1, 2).squeeze(-1),
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            v, k, q = torch.split(
                vkq,
                [
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                    self.num_heads * self.head_dim,
                ],
                dim=1,
            )

            v = rearrange(v, "b (h n) -> b h n", h=self.num_key_value_heads)
            k = rearrange(k, "b (h n) -> b h n", h=self.num_key_value_heads)
            q = rearrange(q, "b (h n) -> b h n", h=self.num_heads)
            k = repeat_kv2(k, self.num_key_value_groups)
            v = repeat_kv2(v, self.num_key_value_groups)

            dt = dt.transpose(1, 2).squeeze(-1)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            A = -torch.exp(self.A_log_bias.float())
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.head_dim).to(dtype=torch.float32)
            D = torch.zeros((self.num_heads, self.head_dim), dtype=A.dtype, device=A.device)

            y = selective_state_update(
                ssm_state,
                v,
                dt,
                A=A,
                B=k,
                C=q,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            
        else:
            vkq = causal_conv1d_fn(
                vkq.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                initial_states=None,
                return_final_states=False,
                activation=None if self.activation == "identity" else self.activation,
            )

            v, k, q = torch.split(
                vkq,
                [
                    self.num_key_value_heads * self.head_dim,
                    self.num_key_value_heads * self.head_dim,
                    self.num_heads * self.head_dim,
                ],
                dim=1,
            )

            v = rearrange(v, "b (h n) l -> b h l n", h=self.num_key_value_heads)
            k = rearrange(k, "b (h n) l -> b h l n", h=self.num_key_value_heads)
            q = rearrange(q, "b (h n) l -> b l h n", h=self.num_heads)
            k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
            v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
                
            A = -torch.exp(self.A_log_bias.float())

            y = mamba_chunk_scan_combined(
                x = v,
                dt=dt,
                dt_softplus=True,
                A=A,
                B=k,
                C=q,
                chunk_size=self.chunk_size,
                dt_bias=self.dt_bias,
                initial_states=None, # currently not supported by mamba_ssm.utils.generation
                return_final_states=False,
            )
        
        g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
        y_true = self.g_norm_swish_gate(y, g)
        y_true = y_true.view(batch, seqlen, self.hidden_size)
        y_true = self.o_proj(y_true)

        return y_true
        
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        device = self.conv1d.weight.device
        dtype = self.conv1d.weight.dtype
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size, 2*self.hidden_size, self.d_conv-1, device=device, dtype=dtype
            )
            ssm_state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.conv1d.weight.device
        dtype = self.conv1d.weight.dtype
        conv_state = torch.zeros(
            batch_size, 2*self.hidden_size, self.d_conv-1, device=device, dtype=dtype
        )

        ssm_state = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype
        )
        return conv_state, ssm_state
    

mmMamba_ATTENTION_CLASSES = {
    'mha': MHA_LM,
    "mamba2":Mamba2_LM
}

# Modified from transformers.model.llama.modeling_llama.LlamaDecoderLayer
class mmMambaDecoderLayer(nn.Module):
    def __init__(self, config: mmMambaEmbeddingConfig, layer_idx: int, drop_path_rate=0.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention = mmMamba_ATTENTION_CLASSES[config.layers_block_type[layer_idx]](config=config, layer_idx=layer_idx)

        self.feed_forward = mmMambaMLP(config)
        self.attention_norm = mmMambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = mmMambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*)
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states = self.attention(
            hidden_states=hidden_states,
            inference_params=inference_params,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + self.drop_path1(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + self.drop_path2(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.attention.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: mmMambaEmbeddingConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
        
        self.post_init()
    
    def post_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor, 
                use_cls_token=False,
                ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(target_dtype)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if use_cls_token:
            class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            assert not self.config.use_2d_sincos_pos_embed, '2D SinCos pos embed is not supported with use_cls_token'
            position_embedding = torch.cat([
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
            ], dim=1)
            embeddings = embeddings + position_embedding
        else:
            position_embedding = self._get_pos_embed(self.position_embedding[:, 1:, :], height, width).to(target_dtype)
            embeddings = patch_embeds + position_embedding

        return embeddings


class mmMambaEmbedding(PreTrainedModel):
    config_class = mmMambaEmbeddingConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: mmMambaEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.gradient_checkpointing = True

        self.vision_embeddings = VisionEmbeddings(config)
        self.llm_text_embeddings = nn.Embedding(self.config.llm_vocab_size, self.config.llm_hidden_size)
        self.special_token_maps = config.special_token_maps
        if len(self.special_token_maps) > 0:
            self.special_text_embeddings = nn.Embedding(len(config.special_token_maps), self.config.llm_hidden_size)

        assert self.config.use_ls is False, 'LS is not supported in mmMamba'
        if hasattr(config, 'drop_path_rate'):
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        else:
            dpr = [0.0] * config.num_hidden_layers
        self.encoder = nn.ModuleList([
            mmMambaDecoderLayer(config, idx, dpr[idx]) for idx in range(config.num_hidden_layers)
        ])
        
        if self.config.use_pixel_shuffle_proj:
            self.pixel_shuffle_proj = nn.Sequential(
                nn.Linear(int(config.hidden_size / (config.downsample_ratio * config.downsample_ratio)), config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        
        self.num_img_tokens = (self.config.image_size // self.config.patch_size) ** 2
        
    def set_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for layer in self.encoder:
            layer.gradient_checkpointing = True

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.vision_embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.vision_embeddings.position_embedding = nn.Parameter(pos_emb)
        self.vision_embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))
    
    def replace_img_tokens(self, input_ids, hidden_states, vision_hidden_states):
        img_context_token_mask = (input_ids == self.config.img_context_token_id)
        hidden_states[img_context_token_mask] = hidden_states[img_context_token_mask] * 0.0 + vision_hidden_states.flatten(0, 1)

        return hidden_states
    
    def get_ignore_mask(self, input_ids):
        ignore_ids = torch.tensor(
            [self.special_token_maps[token] for token in [IMG_START_TOKEN, IMG_END_TOKEN]], 
            device=input_ids.device)
        ignore_mask = torch.isin(input_ids, ignore_ids)

        return ignore_mask
    
    def get_text_mask(self, input_ids):
        txt_mask = (input_ids != self.config.img_context_token_id)

        return txt_mask
    
    def get_input_embeddings(self, input_ids):
        special_mask = input_ids > self.llm_text_embeddings.weight.shape[0] - 1
        llm_embeddings = self.llm_text_embeddings(input_ids * (~special_mask).to(input_ids))

        if len(self.special_token_maps) > 0:
            special_embeddings = self.special_text_embeddings((input_ids - self.llm_text_embeddings.weight.shape[0]) * special_mask.to(input_ids))
            special_mask = special_mask.unsqueeze(-1)
            text_embeddings = llm_embeddings * (~special_mask).to(llm_embeddings) + \
                                special_embeddings * special_mask.to(llm_embeddings)
        else:
            text_embeddings = llm_embeddings

        return text_embeddings
    
    def get_txt_embeddings(self, input_ids):
        B, L = input_ids.shape
        txt_mask = (input_ids != self.config.img_context_token_id)
        txt_embeddings = self.llm_text_embeddings(input_ids[txt_mask])
        txt_embeddings = txt_embeddings.reshape(-1, txt_embeddings.shape[-1])

        return txt_embeddings
    
    def get_txt_feature(self, input_ids, feature):
        B, L, C = feature.shape
        txt_mask = (input_ids != self.config.img_context_token_id)
        txt_feature = feature[txt_mask].reshape(-1, C)

        return txt_feature
    
    def get_img_feature(self, input_ids, feature):
        B, L, C = feature.shape
        img_mask = (input_ids == self.config.img_context_token_id)
        img_feature = feature[img_mask].reshape(-1, C)

        return img_feature
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        if getattr(self.config, 'pixel_shuffle_loc', 'pre') == 'post':
            x = x.view(x.shape[0]//self.num_img_tokens, self.num_img_tokens, -1)

        n, l, c = x.size()
        h = w = int(l ** 0.5)
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).reshape(n, int(l * scale_factor * scale_factor), int(c / (scale_factor * scale_factor))).contiguous()
        
        if getattr(self.config, 'pixel_shuffle_loc', 'pre') == 'post':
            x = x.view(int(x.shape[0]*self.num_img_tokens*(self.config.downsample_ratio**2)), -1)
        return x

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inference_params = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is not None:
            if len(pixel_values.shape) == 4:
                if self.gradient_checkpointing and self.training:
                    vision_hidden_states = torch.utils.checkpoint.checkpoint(self.vision_embeddings, pixel_values)
                else:
                    vision_hidden_states = self.vision_embeddings(pixel_values)

                if self.config.use_pixel_shuffle_proj and getattr(self.config, 'pixel_shuffle_loc', 'pre') == 'pre':
                    vision_hidden_states = self.pixel_shuffle(vision_hidden_states, scale_factor=self.config.downsample_ratio)
                    if self.gradient_checkpointing and self.training:
                        vision_hidden_states = torch.utils.checkpoint.checkpoint(self.pixel_shuffle_proj, vision_hidden_states)
                    else:
                        vision_hidden_states = self.pixel_shuffle_proj(vision_hidden_states)

                hidden_states = self.get_input_embeddings(input_ids)
                hidden_states = self.replace_img_tokens(input_ids, hidden_states, vision_hidden_states)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        for layer_idx, layer_module in enumerate(self.encoder):
            if self.gradient_checkpointing and self.training:
                assert use_cache is None, 'Gradient checkpointing is not compatible with cache'
                outputs = torch.utils.checkpoint.checkpoint(layer_module, 
                                                            hidden_states,
                                                            inference_params,
                                                            None, False, False,
                                                            )
                hidden_states = outputs[0]
            else:
                outputs = layer_module(
                    hidden_states=hidden_states,
                    inference_params=inference_params,
                    use_cache=use_cache,
                )
                hidden_states = outputs[0]


        img_feature = self.get_img_feature(input_ids, hidden_states)
        
        if self.config.use_pixel_shuffle_proj and getattr(self.config, 'pixel_shuffle_loc', 'pre') == 'post':
            img_feature = self.pixel_shuffle(img_feature, scale_factor=self.config.downsample_ratio)
            img_feature = self.pixel_shuffle_proj(img_feature)
        
        return img_feature, hidden_states
            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            layer.layer_idx: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for layer in self.encoder
        }
        