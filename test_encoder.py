import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from networks.backbone import (
    SingleFrameEncoder,
    STTEncoder,
)
from networks.vlm import (
    MMMambaEncoder,
    QwenVLEncoder,
    SmolVLMEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["single_frame", "mmmamba", "smolvlm", "qwenvl", "stt", "all"],
        default="all",
    )
    parser.add_argument("--images_dir", type=Path, default="./local/image/ep_00000001")
    parser.add_argument("--num_images", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = args.images_dir
    num_images = args.num_images

    # 画像の前処理用のtransform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )

    image_list = []

    if images_dir.is_file():
        cap = cv2.VideoCapture(str(images_dir))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_list.append(frame)
            if len(image_list) >= num_images:
                break
        cap.release()
    else:
        image_path_list = sorted(images_dir.glob("*.png"))
        image_path_list = image_path_list[:num_images]
        for i, image_path in enumerate(image_path_list):
            image = cv2.imread(str(image_path))
            image_list.append(image)

    print(f"Loaded {len(image_list)} images from {images_dir}")

    # 画像をシーケンスとして結合し、GPUに移動
    image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
    image_list = [transform(image) for image in image_list]
    seq_len = len(image_list)
    images_sequence = torch.stack(image_list)  # shape: (sequence_length, 3, height, width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_sequence = images_sequence.to(device)
    # バッチ方向にコピー -> shape: (batch_size, sequence_length, 3, height, width)
    batch_size = 2
    images_sequence = images_sequence.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    print(f"Images sequence shape: {images_sequence.shape}")
    print(f"Using device: {device}")

    action_sequence = torch.randn((batch_size, seq_len, 3), device=device)
    reward_sequence = torch.randn((batch_size, seq_len, 1), device=device)

    # エンコーダーの初期化
    print("Initializing Encoder")
    target_encoder_list = []
    if args.encoder == "single_frame":
        target_encoder_list.append(SingleFrameEncoder(num_images, device))
    elif args.encoder == "mmmamba":
        target_encoder_list.append(MMMambaEncoder(num_images, device))
    elif args.encoder == "smolvlm":
        target_encoder_list.append(SmolVLMEncoder(num_images, device))
    elif args.encoder == "qwenvl":
        target_encoder_list.append(QwenVLEncoder(num_images, device))
    elif args.encoder == "stt":
        target_encoder_list.append(STTEncoder(num_images, device, "transformer"))
    else:  # all
        target_encoder_list.append(SingleFrameEncoder(num_images, device))
        target_encoder_list.append(MMMambaEncoder(num_images, device))
        target_encoder_list.append(SmolVLMEncoder(num_images, device))
        target_encoder_list.append(QwenVLEncoder(num_images, device))
        target_encoder_list.append(STTEncoder(num_images, device, "transformer"))

    for encoder in target_encoder_list:
        start = time.time()
        representation = encoder(images_sequence, action_sequence, reward_sequence)
        end = time.time()
        elapsed_msec = (end - start) * 1000
        print(
            f"{encoder.__class__.__name__}, {representation.shape=}, {elapsed_msec=:.1f}"
        )
