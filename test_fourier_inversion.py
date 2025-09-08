#!/usr/bin/env python3

import torch
import torch.nn as nn

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


class LinearEmbedder(nn.Module):
    """
    LinearでスカラーをEmbedして、同じ重みの転置で戻すLayer
    """

    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bias = bias
        # 1次元→embed_dim次元へのLinear層
        self.encoder = nn.Linear(1, embed_dim, bias=bias)

    def embed(self, x):
        """
        スカラー値を埋め込みベクトルに変換
        Args:
            x: [B, T, coord_dim] のスカラー値
        Returns:
            [B, T, coord_dim, embed_dim] の埋め込みベクトル
        """
        # x: [B, T, coord_dim] -> [B, T, coord_dim, 1]
        x = x.unsqueeze(-1)
        # Linear変換: [B, T, coord_dim, 1] -> [B, T, coord_dim, embed_dim]
        embedded = self.encoder(x)
        return embedded

    def decode(self, embedded):
        """
        埋め込みベクトルからスカラー値を復元（全要素の平均を使用）
        Args:
            embedded: [B, T, coord_dim, embed_dim] の埋め込みベクトル
        Returns:
            [B, T, coord_dim] の復元されたスカラー値
        """
        if self.use_bias:
            # バイアス有りの場合: y = x * W + b => x = (y - b) / W
            bias_expanded = self.encoder.bias.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
            decoded_values = (embedded - bias_expanded) / self.encoder.weight[:, 0]
        else:
            # バイアス無しの場合: y = x * W => x = y / W
            decoded_values = embedded / self.encoder.weight[:, 0]

        # 全要素の平均を取る
        decoded = decoded_values.mean(dim=-1)  # [B, T, coord_dim]

        return decoded


if __name__ == "__main__":
    embed_dim = 64
    num_samples = 100
    coord_range = (-5.0, 5.0)

    coords = (
        torch.linspace(coord_range[0], coord_range[1], num_samples).unsqueeze(0).unsqueeze(-1)
    )  # [1, num_samples, 1]

    # embeddings = get_fourier_embeds_from_coordinates(embed_dim, coords)
    # recovered_coords = inverse_get_fourier_embeds_from_coordinates(embeddings)

    embedder = LinearEmbedder(embed_dim)
    embeddings = embedder.embed(coords.squeeze(-1))
    recovered_coords = embedder.decode(embeddings).unsqueeze(-1)

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
