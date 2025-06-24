import math

import torch.nn as nn


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute gradient norm for a model.

    Args:
        model: Neural network model

    Returns:
        Gradient norm value
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm()
            total_norm += param_norm.item() ** 2.0

    total_norm = math.sqrt(total_norm)
    return total_norm
