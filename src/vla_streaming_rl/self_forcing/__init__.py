"""Self-Forcing (vendored).

Vendored from https://github.com/guandeh17/Self-Forcing, stripped down to a
single-GPU LoRA fine-tune + KV-cached causal inference path on Wan 2.1 T2V-1.3B.

Public entry points:
    Trainer (training-time)            -- vla_streaming_rl.self_forcing.train (script)
    CausalDiffusion (training model)   -- model.training_model
    CausalInferencePipeline (infer)    -- model.inference_model
"""

from .model.inference_model import CausalInferencePipeline
from .model.training_model import CausalDiffusion

__all__ = ["CausalDiffusion", "CausalInferencePipeline"]
