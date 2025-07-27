import argparse
import time
from pathlib import Path

import cv2
import torch
from mamba_ssm.utils.generation import InferenceParams
from torchvision import transforms

from networks.backbone import MMMambaEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default="./local/image/ep_00000001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = args.images_dir

    image_path_list = sorted(images_dir.glob("*.png"))
    print(f"{len(image_path_list)=}")

    NUM = 40
    image_path_list = image_path_list[:NUM]
    image_list = []

    # 画像の前処理用のtransform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )

    print(f"Loading {len(image_path_list)} images...")

    for i, image_path in enumerate(image_path_list):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
        image_tensor = transform(image)
        image_list.append(image_tensor)
        print(f"Loaded image {i + 1}/{len(image_path_list)}: {image_path.name}")

    # 画像をシーケンスとして結合し、GPUに移動
    images_sequence = torch.stack(image_list)  # shape: (sequence_length, 3, height, width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_sequence = images_sequence.to(device)
    print(f"Images sequence shape: {images_sequence.shape}")
    print(f"Using device: {device}")

    # VLMエンコーダーの初期化
    print("Initializing MMMambaEncoder...")
    encoder = MMMambaEncoder(device=device)

    # Test step
    encoder.reset_inference_params()
    encoder.encode(images_sequence[0:1])
    encoder.reset_inference_params()

    start = time.time()

    for i, image_tensor in enumerate(images_sequence):
        print(f"start {i=}")
        image_tensor = image_tensor.unsqueeze(0)
        encoder.encode(image_tensor)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        average_msec = elapsed_msec / (i + 1)
        print(f"Step {i + 1}/{len(images_sequence)}: {average_msec=:.1f} ms")
