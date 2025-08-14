import torch

from networks.spatial_temporal_transformer import SpatialTemporalTransformer


def test_spatial_temporal_transformer():
    batch_size = 2
    condition_frames = 3
    img_tokens_size = 1024
    action_tokens_size = 3  # 3つの行動に対応
    total_tokens_size = img_tokens_size + action_tokens_size

    token_size_dict = {
        "img_tokens_size": img_tokens_size,
        "action_tokens_size": action_tokens_size,
        "total_tokens_size": total_tokens_size,
    }

    # 3つの行動のvocab size（例：x座標、y座標、角度）
    action_vocab_sizes = [512, 512, 360]  # 最後は角度なので360

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
        action_vocab_sizes=action_vocab_sizes,
    )

    feature_total = torch.randn(batch_size, condition_frames + 1, img_tokens_size, 8)
    # 3つの行動値 (action_0, action_1, action_2)
    action_indices_total = torch.stack(
        [
            torch.randint(0, 512, (batch_size, (condition_frames + 1) * 1)),  # action_0
            torch.randint(0, 512, (batch_size, (condition_frames + 1) * 1)),  # action_1
            torch.randint(0, 360, (batch_size, (condition_frames + 1) * 1)),  # action_2 (角度)
        ],
        dim=-1,
    )

    print("入力サイズ:")
    print(f"feature_total: {feature_total.shape}")
    print(f"action_indices_total: {action_indices_total.shape}")

    output = model(feature_total, action_indices_total)

    print("\n出力サイズ:")
    print(f"logits: {output['logits'].shape}")
    print(f"action_emb: {output['action_emb'].shape}")

    print("\nテスト成功！")


if __name__ == "__main__":
    test_spatial_temporal_transformer()
