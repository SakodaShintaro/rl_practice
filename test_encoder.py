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
    return parser.parse_args()


def test_standard_encoder(encoder, images_sequence, action_sequence, reward_sequence, device):
    """Test standard encoders (SpatialTemporal, TemporalOnly)"""
    batch_size = images_sequence.shape[0]
    obs_z_shape = encoder.image_processor.output_shape
    obs_z = torch.zeros(batch_size, images_sequence.shape[1], *obs_z_shape, device=device)
    rnn_state = encoder.init_state().to(device).repeat(1, batch_size, 1)

    start = time.time()
    representation, _ = encoder(images_sequence, obs_z, action_sequence, reward_sequence, rnn_state)
    end = time.time()
    elapsed_msec = (end - start) * 1000

    print(f"{encoder.__class__.__name__}")
    print(f"  Representation shape: {representation.shape}")
    print(f"  Elapsed time: {elapsed_msec:.1f} ms")


def test_vlm_encoder(encoder, images_sequence):
    """Test VLM encoders (MMMamba, SmolVLM, QwenVL)"""
    # VLMエンコーダは1フレームずつ処理
    print(f"{encoder.__class__.__name__}")

    # Test step
    encoder(images_sequence[0, 0:1])

    start = time.time()
    for i in range(images_sequence.shape[1]):
        image_tensor = images_sequence[0, i : i + 1]
        representation, action_text = encoder(image_tensor)

        # Parse action text to get numeric values
        action_values = parse_action_text(action_text)

        end = time.time()
        elapsed_msec = (end - start) * 1000
        average_msec = elapsed_msec / (i + 1)
        print(f"  Step {i + 1}/{images_sequence.shape[1]}: {average_msec:.1f} ms")
        print(f"    Action text: {action_text}")
        print(
            f"    Parsed values: steering={action_values[0]:.3f}, gas={action_values[1]:.3f}, braking={action_values[2]:.3f}"
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
    seq_len = len(image_list)
    images_sequence = torch.stack(image_list)  # shape: (sequence_length, 3, height, width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_sequence = images_sequence.to(device)
    print(f"Images sequence shape (single batch): {images_sequence.shape}")
    print(f"Using device: {device}")

    # エンコーダーの初期化
    print("\nInitializing Encoder...")
    observation_space_shape = (3, 96, 96)

    if args.encoder == "spatial_temporal":
        # バッチサイズ2で標準エンコーダをテスト
        batch_size = 2
        batched_images = images_sequence.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        action_sequence = torch.randn((batch_size, seq_len, 3), device=device)
        reward_sequence = torch.randn((batch_size, seq_len, 1), device=device)

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
        test_standard_encoder(encoder, batched_images, action_sequence, reward_sequence, device)

    elif args.encoder == "temporal_only":
        # バッチサイズ2で標準エンコーダをテスト
        batch_size = 2
        batched_images = images_sequence.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        action_sequence = torch.randn((batch_size, seq_len, 3), device=device)
        reward_sequence = torch.randn((batch_size, seq_len, 1), device=device)

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
        test_standard_encoder(encoder, batched_images, action_sequence, reward_sequence, device)

    elif args.encoder == "mmmamba":
        # VLMエンコーダはバッチサイズ1で実行
        vlm_images = images_sequence.unsqueeze(0)
        encoder = MMMambaEncoder(device)
        test_vlm_encoder(encoder, vlm_images)

    elif args.encoder == "smolvlm":
        # VLMエンコーダはバッチサイズ1で実行
        vlm_images = images_sequence.unsqueeze(0)
        encoder = SmolVLMEncoder(device)
        test_vlm_encoder(encoder, vlm_images)

    elif args.encoder == "qwenvl":
        # VLMエンコーダはバッチサイズ1で実行
        vlm_images = images_sequence.unsqueeze(0)
        encoder = QwenVLEncoder(device)
        test_vlm_encoder(encoder, vlm_images)
