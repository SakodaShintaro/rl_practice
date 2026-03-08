# SPDX-License-Identifier: MIT
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
        if param.grad is not None and param.grad.dtype.is_floating_point:
            param_norm = param.grad.data.norm()
            total_norm += param_norm.item() ** 2.0

    total_norm = math.sqrt(total_norm)
    return total_norm


def compute_parameter_norm(model: nn.Module) -> float:
    """
    Compute parameter norm for a model.

    Args:
        model: Neural network model

    Returns:
        Parameter norm value
    """
    total_norm = 0.0

    for param in model.parameters():
        if param.dtype.is_floating_point:
            param_norm = param.data.norm()
            total_norm += param_norm.item() ** 2.0

    total_norm = math.sqrt(total_norm)
    return total_norm
