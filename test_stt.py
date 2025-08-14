import torch

from networks.spatial_temporal_transformer import SpatialTemporalTransformer


def test_spatial_temporal_transformer():
    batch_size = 2
    condition_frames = 3
    img_tokens_size = 1024
    pose_tokens_size = 2
    yaw_token_size = 1
    total_tokens_size = img_tokens_size + pose_tokens_size + yaw_token_size

    token_size_dict = {
        "img_tokens_size": img_tokens_size,
        "pose_tokens_size": pose_tokens_size,
        "yaw_token_size": yaw_token_size,
        "total_tokens_size": total_tokens_size,
    }

    model = SpatialTemporalTransformer(
        block_size=total_tokens_size,
        n_layer=[2, 1],
        n_head=8,
        n_embd=512,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        n_unmasked=0,
        local_rank=0,
        condition_frames=condition_frames,
        latent_size=(32, 32),
        token_size_dict=token_size_dict,
        vae_emb_dim=8,
        temporal_block=1,
        pose_x_vocab_size=512,
        pose_y_vocab_size=512,
        yaw_vocab_size=512,
    )

    feature_total = torch.randn(batch_size, condition_frames + 1, img_tokens_size, 8)
    pose_indices_total = torch.randint(0, 512, (batch_size, (condition_frames + 1) * 1, 2))
    yaw_indices_total = torch.randint(0, 512, (batch_size, (condition_frames + 1) * 1, 1))

    print("入力サイズ:")
    print(f"feature_total: {feature_total.shape}")
    print(f"pose_indices_total: {pose_indices_total.shape}")
    print(f"yaw_indices_total: {yaw_indices_total.shape}")

    output = model(feature_total, pose_indices_total, yaw_indices_total)

    print("\n出力サイズ:")
    print(f"logits: {output['logits'].shape}")
    print(f"pose_emb: {output['pose_emb'].shape}")

    print("\nテスト成功！")


if __name__ == "__main__":
    test_spatial_temporal_transformer()
