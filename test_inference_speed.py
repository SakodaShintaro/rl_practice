import time

import torch
from torch import nn

from networks.backbone import AE, VAE, SmolVLMEncoder


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def test_inference_speed(model, name, batch_size, dtype):
    device = torch.device("cuda")
    x = torch.rand(batch_size, 3, 96, 96, device=device, dtype=dtype)

    COUNT = 10

    model.to(device)
    model.eval()
    model = nn.DataParallel(model)
    p_count = parameter_count(model)

    _ = model(x)

    start_time = time.time()
    for _ in range(COUNT):
        _ = model(x)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / COUNT * 1000
    print(f"{name},{p_count},{elapsed_time:.1f}")


if __name__ == "__main__":
    device = torch.device("cuda")

    model_ae = AE().to(device)
    model_vae = VAE().to(device)
    model_smolvlm = SmolVLMEncoder(device=device)
    model_smolvlm_image_encoder = model_smolvlm.model.model.vision_model

    for batch_size in [4, 8, 16]:
        print(f"{batch_size=}")
        test_inference_speed(model_ae, "AE", batch_size, torch.float32)

        test_inference_speed(model_vae, "VAE", batch_size, torch.float32)

        test_inference_speed(
            model_smolvlm_image_encoder, "SmolVLMEncoderImageEncoder", batch_size, torch.bfloat16
        )

        test_inference_speed(model_smolvlm, "SmolVLMEncoder", batch_size, torch.float32)
