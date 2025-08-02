import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from networks.backbone import AE, MMMambaEncoder, QwenVLEncoder, SmolVLMEncoder, parse_action_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", type=str, choices=["ae", "mmmamba", "smolvlm", "qwenvl"])
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
    images_sequence = torch.stack(image_list)  # shape: (sequence_length, 3, height, width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_sequence = images_sequence.to(device)
    print(f"Images sequence shape: {images_sequence.shape}")
    print(f"Using device: {device}")

    # VLMエンコーダーの初期化
    print("Initializing Encoder")
    if args.encoder == "ae":
        encoder = AE(device=device)
    elif args.encoder == "mmmamba":
        encoder = MMMambaEncoder(device=device)
    elif args.encoder == "smolvlm":
        encoder = SmolVLMEncoder(device=device)
    elif args.encoder == "qwenvl":
        encoder = QwenVLEncoder(device=device)
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder}")

    # Test step
    encoder.reset_inference_params()
    encoder(images_sequence[0:1])
    encoder.reset_inference_params()

    start = time.time()

    for i, image_tensor in enumerate(images_sequence):
        image_tensor = image_tensor.unsqueeze(0)
        representation, action_text = encoder(image_tensor)

        # Parse action text to get numeric values
        steering, gas, braking = parse_action_text(action_text)

        end = time.time()
        elapsed_msec = (end - start) * 1000
        average_msec = elapsed_msec / (i + 1)
        print(f"Step {i + 1}/{len(images_sequence)}: {average_msec=:.1f} ms")
        print(f"  Action text: {action_text}")
        print(f"  Parsed values: steering={steering:.3f}, gas={gas:.3f}, braking={braking:.3f}")
