# SPDX-License-Identifier: MIT
"""Download and extract STL-10 dataset."""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

STL10_URL = "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

LABEL_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        images = np.fromfile(f, dtype=np.uint8)
    images = images.reshape(-1, 3, 96, 96)
    return np.transpose(images, (0, 3, 2, 1))


def load_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.fromfile(f, dtype=np.uint8)


def save_images(images: np.ndarray, labels: np.ndarray, split_dir: Path):
    for i, (img, label) in enumerate(zip(images, labels)):
        label_name = LABEL_NAMES[label - 1]
        label_dir = split_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(label_dir / f"{i:05d}.png")


if __name__ == "__main__":
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "stl10_binary.tar.gz"
    if tar_path.exists():
        print(f"Already downloaded: {tar_path}")
    else:
        print(f"Downloading STL-10 to {tar_path} ...")
        urllib.request.urlretrieve(STL10_URL, tar_path)

    print("Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir, filter="data")

    bin_dir = output_dir / "stl10_binary"

    print("Converting train images ...")
    train_images = load_images(bin_dir / "train_X.bin")
    train_labels = load_labels(bin_dir / "train_y.bin")
    save_images(train_images, train_labels, output_dir / "train")

    print("Converting test images ...")
    test_images = load_images(bin_dir / "test_X.bin")
    test_labels = load_labels(bin_dir / "test_y.bin")
    save_images(test_images, test_labels, output_dir / "test")

    print(f"Done. train={len(train_images)}, test={len(test_images)}")
