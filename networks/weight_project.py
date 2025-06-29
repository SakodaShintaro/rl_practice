import torch
import torch.nn as nn


def get_initial_norms(module: nn.Module) -> dict[str, float]:
    """
    Get initial parameter norms for weight projection.

    Args:
        module: Neural network module

    Returns:
        Dictionary of parameter names to their initial norms
    """
    initial_norms = {}
    for name, param in module.named_parameters():
        if param.requires_grad and _is_weight_parameter(name, param):
            initial_norms[name] = param.data.norm().item()
    return initial_norms


def weight_project(module: nn.Module, initial_norms: dict[str, float]) -> None:
    """
    Apply weight projection to maintain initial parameter norms.

    This should be called after every gradient step to maintain constant
    parameter norms as described in the NaP paper.

    Args:
        module: Neural network module
        initial_norms: Dictionary of parameter names to their target norms
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.requires_grad and name in initial_norms:
                current_norm = param.data.norm().item()
                if current_norm > 1e-8:  # Avoid division by zero
                    target_norm = initial_norms[name]
                    scaling_factor = target_norm / current_norm
                    param.data.mul_(scaling_factor)


def _is_weight_parameter(name: str, param: torch.Tensor) -> bool:
    """Check if parameter is a weight matrix (not bias or scale/offset)."""
    return "weight" in name and param.dim() >= 2
