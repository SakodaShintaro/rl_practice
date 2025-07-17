from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from networks.backbone import SmolVLMEncoder

images_dir = Path(
    "/home/sakoda/work/rl_practice/results/20250717_115527_SAC_HLGaussLoss(-70,+70)/image/ep_00000350"
)
image_path_list = sorted(images_dir.glob("*.png"))

NUM = 5  # より多くの画像を処理
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
print("Initializing SmolVLMEncoder...")
encoder = SmolVLMEncoder(device=device)

# 個別画像の説明を生成
print("\nGenerating descriptions for individual images...")
individual_descriptions = encoder.describe_image_sequence(images_sequence)

# 結果を表示
print("\n" + "=" * 50)
print("Individual Image Descriptions:")
print("=" * 50)
for i, (image_path, description) in enumerate(zip(image_path_list, individual_descriptions)):
    print(f"\nImage {i + 1}: {image_path.name}")
    print(f"Description: {description}")
    print("-" * 30)

# 累積シーケンスの説明を生成
print("\n" + "=" * 50)
print("Cumulative Sequence Descriptions:")
print("=" * 50)
cumulative_prompt = "Analyze these driving game screenshots and describe what happens"
cumulative_descriptions = encoder.describe_cumulative_sequence(images_sequence, cumulative_prompt)

for i, (description) in enumerate(cumulative_descriptions):
    images_in_sequence = i + 1
    image_names = [path.name for path in image_path_list[:images_in_sequence]]
    print(f"\nCumulative sequence {i + 1} (Images: {', '.join(image_names)}):")
    print(f"Description: {description}")
    print("-" * 30)

# カスタムプロンプトでの説明生成も試してみる
print("\n" + "=" * 50)
print("Custom Prompt Descriptions (Individual):")
print("=" * 50)
custom_prompt = "What is happening in this driving scene? <image>"
custom_descriptions = encoder.describe_image_sequence(images_sequence, custom_prompt)

for i, (image_path, description) in enumerate(zip(image_path_list, custom_descriptions)):
    print(f"\nImage {i + 1}: {image_path.name}")
    print(f"Custom Description: {description}")
    print("-" * 30)
