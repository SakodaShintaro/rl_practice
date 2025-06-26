"""
Sparse network utilities for implementing one-shot random pruning
based on "Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning"
"""

import torch
import torch.nn as nn


def calculate_erdos_renyi_sparsity(
    layer_shape: tuple, overall_sparsity: float, is_conv: bool = False
) -> float:
    """
    Calculate layer-wise sparsity using Erdős-Rényi initialization

    Args:
        layer_shape: Shape of the layer (input_dim, output_dim) for FC or (out_ch, in_ch, h, w) for conv
        overall_sparsity: Overall target sparsity level
        is_conv: Whether this is a convolutional layer

    Returns:
        Layer-specific sparsity ratio
    """
    if is_conv:
        # For conv layers: shape is (out_channels, in_channels, height, width)
        n_out, n_in, h, w = layer_shape
        # Erdős-Rényi formula for conv: 1 - (n_in + n_out + w*h) / (n_in * n_out * w * h)
        sparsity = 1.0 - (n_in + n_out + w * h) / (n_in * n_out * w * h)
    else:
        # For FC layers: shape is (output_dim, input_dim)
        n_out, n_in = layer_shape
        # Erdős-Rényi formula for FC: 1 - (n_in + n_out) / (n_in * n_out)
        sparsity = 1.0 - (n_in + n_out) / (n_in * n_out)

    return max(0.0, min(sparsity, 0.99))  # Clamp between 0 and 0.99


def create_random_mask(shape: tuple, sparsity: float, device: torch.device) -> torch.Tensor:
    """
    Create a random binary mask for pruning

    Args:
        shape: Shape of the tensor to mask
        sparsity: Fraction of weights to zero out (0.0 to 1.0)
        device: Device to create mask on

    Returns:
        Binary mask tensor
    """
    mask = torch.ones(shape, device=device, dtype=torch.float32)

    if sparsity > 0.0:
        # Create random mask
        rand_tensor = torch.rand(shape, device=device)
        # Set weights to 0 where random values are below sparsity threshold
        mask = (rand_tensor >= sparsity).float()

    return mask


def apply_one_shot_pruning(
    module: nn.Module,
    overall_sparsity: float = 0.9,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Apply one-shot random pruning to a neural network module

    Args:
        module: Neural network module to prune
        overall_sparsity: Overall target sparsity (0.0 to 1.0)
        device: Device for computations

    Returns:
        SparseMask object containing all layer masks
    """
    use_erdos_renyi = False
    if device is None:
        device = next(module.parameters()).device

    masks = {}

    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:  # Only prune weight parameters with 2+ dimensions
            shape = param.shape

            if use_erdos_renyi:
                # Determine if this is a conv layer based on dimensions
                is_conv = param.dim() == 4
                layer_sparsity = calculate_erdos_renyi_sparsity(shape, overall_sparsity, is_conv)
            else:
                # Use uniform sparsity
                layer_sparsity = overall_sparsity

            # Create and apply mask
            mask = create_random_mask(shape, layer_sparsity, device)
            masks[name] = mask

            # Apply mask immediately
            param.data *= mask

            # Calculate actual sparsity
            actual_sparsity = (mask == 0).float().mean().item()
            print(
                f"Layer {name}: target sparsity={layer_sparsity:.3f}, actual sparsity={actual_sparsity:.3f}"
            )

    return masks


def apply_masks_during_training(module: nn.Module) -> None:
    """
    Apply masks to ensure pruned weights stay zero during training
    This should be called after each optimizer step

    Args:
        module: Neural network module
    """
    if not hasattr(module, "sparse_mask") or module.sparse_mask is None:
        return

    masks = module.sparse_mask
    for name, param in module.named_parameters():
        if name in masks:
            param.data *= masks[name].to(param.device)
