import torch

from networks.spatial_temporal_transformer import SpatialTemporalTransformer

if __name__ == "__main__":
    batch_size = 2
    condition_frames = 3
    img_tokens_size = 1024
    n_embd = 512

    model = SpatialTemporalTransformer(
        n_layer=2,
        n_head=8,
        n_embd=n_embd,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        condition_frames=condition_frames,
    )

    feature_total = torch.randn(batch_size, condition_frames, img_tokens_size, n_embd)
    print(f"{feature_total.shape=}")

    output = model(feature_total)
    print(f"{output.shape=}")
