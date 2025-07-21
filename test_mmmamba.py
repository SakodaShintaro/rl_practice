import argparse
from pathlib import Path

import cv2
import torch
from mamba_ssm.utils.generation import InferenceParams
from torchvision import transforms

from networks.backbone import MMMambaEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = args.images_dir

    image_path_list = sorted(images_dir.glob("*.png"))
    print(f"{len(image_path_list)=}")

    NUM = 10
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

    # 説明を生成
    prompt = (
        "This is a video of Gymnasium's CarRacing-v3. "
        "You are the red car, and your goal is to follow the grey road. "
        "You must not go off the road indicated by the green. "
        "Choose your action from turn right, go straight, or turn left."
    )
    descriptions = encoder.describe(images_sequence[0:1])

    for i, (description) in enumerate(descriptions):
        image_names = [path.name for path in image_path_list[: i + 1]]
        print(f"Description {i + 1}: {description}")
        print("-" * 30)

    # Test step
    inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)

    for i, image_tensor in enumerate(images_sequence):
        print(f"start {i=}")
        image_tensor = image_tensor.unsqueeze(0)
        encoder.step(image_tensor, inference_params)
