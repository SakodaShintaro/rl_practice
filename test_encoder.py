import argparse
import time
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from networks.backbone import SpatialTemporalEncoder, TemporalOnlyEncoder
from networks.vlm import MMMambaEncoder, QwenVLEncoder, SmolVLMEncoder, parse_action_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder",
        type=str,
        choices=[
            "spatial_temporal",
            "temporal_only",
            "mmmamba",
            "smolvlm",
            "qwenvl",
        ],
    )
    parser.add_argument("--images_dir", type=Path, default="./local/image/ep_00000001")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def test_encoder(encoder, images_sequence, device, batch_size):
    """Test any encoder type with unified interface"""
    print(f"\n{encoder.__class__.__name__}")

    batched_images = images_sequence.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    seq_len = images_sequence.shape[0]

    obs_z_shape = encoder.image_processor.output_shape
    obs_z = torch.zeros(batch_size, seq_len, *obs_z_shape, device=device)
    rnn_state = encoder.init_state().to(device).repeat(1, batch_size, 1)
    action_sequence = torch.randn((batch_size, seq_len, 3), device=device)
    reward_sequence = torch.randn((batch_size, seq_len, 1), device=device)

    start = time.time()
    representation, rnn_state, action_text = encoder(
        batched_images, obs_z, action_sequence, reward_sequence, rnn_state
    )
    end = time.time()
    elapsed_msec = (end - start) * 1000

    print(f"  Batch size: {batch_size}")
    print(f"  Representation shape: {representation.shape}")
    print(f"  Elapsed time: {elapsed_msec:.1f} ms")
    print(f"  Action text: '{action_text}'")
    action_values = parse_action_text(action_text)
    print(
        f"  Parsed action: steering={action_values[0]:.3f}, gas={action_values[1]:.3f}, braking={action_values[2]:.3f}"
    )


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

    # エンコーダーの初期化
    print("\nInitializing Encoder...")
    observation_space_shape = (3, 96, 96)
    seq_len = images_sequence.shape[0]

    if args.encoder == "spatial_temporal":
        encoder = SpatialTemporalEncoder(
            observation_space_shape,
            seq_len,
            1,
            3,
            "transformer",
            "simple_cnn",
            False,
        )
        encoder = encoder.to(device)

    elif args.encoder == "temporal_only":
        encoder = TemporalOnlyEncoder(
            observation_space_shape,
            seq_len,
            1,
            3,
            "gru",
            "simple_cnn",
            True,
        )
        encoder = encoder.to(device)

    elif args.encoder == "mmmamba":
        encoder = MMMambaEncoder(device)

    elif args.encoder == "smolvlm":
        encoder = SmolVLMEncoder(device)

    elif args.encoder == "qwenvl":
        encoder = QwenVLEncoder(device)

    # 統一されたテスト関数で実行
    test_encoder(encoder, images_sequence, device, args.batch_size)
