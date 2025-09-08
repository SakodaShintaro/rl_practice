#!/usr/bin/env python3

import torch

from networks.spatial_temporal_transformer import get_fourier_embeds_from_coordinates


def inverse_get_fourier_embeds_from_coordinates(embeddings: torch.Tensor) -> torch.Tensor:
    """
    フーリエ埋め込みから解析的に座標を復元

    Args:
        embeddings: [B, T, coord_dim, embed_dim] のフーリエ埋め込み

    Returns:
        復元された座標 [B, T, coord_dim]
    """
    embed_dim = embeddings.shape[-1]
    half_dim = embed_dim // 2

    # 最低周波数のsin/cos成分を取得
    sin_comp = embeddings[..., 0]  # [..., 0] は最初のsin成分
    cos_comp = embeddings[..., half_dim]  # [..., half_dim] は最初のcos成分

    # atan2で位相復元
    phase = torch.atan2(sin_comp, cos_comp)

    return phase


if __name__ == "__main__":
    embed_dim = 64
    num_samples = 100
    coord_range = (-5.0, 5.0)

    coords = (
        torch.linspace(coord_range[0], coord_range[1], num_samples).unsqueeze(0).unsqueeze(-1)
    )  # [1, num_samples, 1]

    embeddings = get_fourier_embeds_from_coordinates(embed_dim, coords)

    recovered_coords = inverse_get_fourier_embeds_from_coordinates(embeddings)

    print(f"{coords.shape=}")
    print(f"{embeddings.shape=}")
    print(f"{recovered_coords.shape=}")
    assert coords.shape == recovered_coords.shape

    # [-pi, pi]の範囲内だけOK
    for i in range(num_samples):
        original = coords[0, i, 0].item()
        recovered = recovered_coords[0, i, 0].item()
        diff = original - recovered
        print(
            f"Sample {i:04d}: Original={original:.3f}, Recovered={recovered:.3f}, Diff={diff:.3f}"
        )
