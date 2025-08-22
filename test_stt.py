import torch

from networks.spatial_temporal_transformer import SpatialTemporalTransformer

if __name__ == "__main__":
    B = 2
    T = 3
    S = 1024
    C = 512

    model = SpatialTemporalTransformer(
        n_layer=2,
        space_len=S,
        tempo_len=T,
        hidden_dim=C,
        n_head=8,
        res_drop_prob=0.1,
        attn_drop_prob=0.1,
    )

    feature_total = torch.randn(B, T, S, C)
    print(f"{feature_total.shape=}")

    output = model(feature_total)
    print(f"{output.shape=}")
