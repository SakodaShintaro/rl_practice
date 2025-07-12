import time

import torch

from networks.backbone import AE, VAE, BaseCNN, SmolVLAEncoder, SmolVLMEncoder


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def test_inference_speed(model, name):
    device = torch.device("cuda")
    x = torch.rand(32, 3, 96, 96, device=device)
    _ = model.encode(x)

    COUNT = 10

    model.to(device)
    print(f"{name} parameter count: {parameter_count(model):,}")
    start_time = time.time()
    for _ in range(COUNT):
        _ = model.encode(x)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / COUNT * 1000
    print(f"{name} encode time: {elapsed_time:.1f} ms")


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.rand(32, 3, 96, 96, device=device)

    model_ae = AE().to(device)
    test_inference_speed(model_ae, "AE")

    model_vae = VAE().to(device)
    test_inference_speed(model_vae, "VAE")

    model_cnn = BaseCNN(in_channels=3).to(device)
    test_inference_speed(model_cnn, "BaseCNN")

    model_smolvlm = SmolVLMEncoder(device=device)
    test_inference_speed(model_smolvlm, "SmolVLMEncoder")

    model_smolvla = SmolVLAEncoder(device=device)
    test_inference_speed(model_smolvla, "SmolVLABackbone")
