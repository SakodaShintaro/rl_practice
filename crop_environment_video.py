"""
Script to crop only the left side (environment part) of videos

Due to concat_labeled_images processing, images are arranged as follows:
[environment | observation | prediction | reward]

A label is attached at the top of each image.
- env.render() size: 400x600
- Label height: 28 pixels (added by add_text_label_on_top)
- Environment part size (including label): 428x600
"""

import argparse
from pathlib import Path

import cv2
import imageio

# Crop region constants (environment part excluding label)
CROP_X = 0
CROP_Y = 28
CROP_WIDTH = 600
CROP_HEIGHT = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    return parser.parse_args()


def crop_video(input_path, output_path):
    """
    Crop only the environment part of the video

    Args:
        input_path: Path to input video
        output_path: Path to output video
    """
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cropped_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_frame = frame_rgb[CROP_Y : CROP_Y + CROP_HEIGHT, CROP_X : CROP_X + CROP_WIDTH]
        cropped_frames.append(cropped_frame)

    cap.release()

    if cropped_frames:
        print(f"Size after cropping: {cropped_frames[0].shape[1]}x{cropped_frames[0].shape[0]}")
        print(f"Total frame count: {len(cropped_frames)}")
        imageio.mimsave(str(output_path), cropped_frames, fps=fps, macro_block_size=1)
        print(f"Save complete: {output_path}")
    else:
        print("No frames found")


def main():
    args = parse_args()
    input_dir = args.input_dir

    output_dir = input_dir.parent / "video_cropped"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video files matching the pattern
    video_files = sorted(list(input_dir.glob("*.mp4")))
    if not video_files:
        print(f"No video files found: {input_dir}/*.mp4")
        return

    print(f"Number of videos to process: {len(video_files)}")

    for i, video_path in enumerate(video_files):
        print(f"\n[{i + 1}/{len(video_files)}] Processing: {video_path.name}")
        output_path = output_dir / video_path.name
        crop_video(video_path, output_path)


if __name__ == "__main__":
    main()
