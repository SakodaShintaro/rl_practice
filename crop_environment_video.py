"""
動画の左側（environment部分）だけをクロップするスクリプト

concat_labeled_imagesの処理により、画像は以下のように並んでいる：
[environment | observation | prediction | reward]

各画像の上部にはラベルが付与されている。
- env.render()のサイズ: 400x600
- ラベルの高さ: 28ピクセル（add_text_label_on_topで追加）
- environment部分のサイズ（ラベル込み）: 428x600
"""

import argparse
from pathlib import Path

import cv2
import imageio

# クロップ領域の定数（ラベルを除外したenvironment部分）
CROP_X = 0
CROP_Y = 28
CROP_WIDTH = 600
CROP_HEIGHT = 400


def crop_video(input_path, output_path):
    """
    動画のenvironment部分だけをクロップ

    Args:
        input_path: 入力動画のパス
        output_path: 出力動画のパス
    """
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cropped_frames = []
    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if first_frame:
            print(f"元の動画サイズ: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}")
            print(f"クロップ領域: x={CROP_X}, y={CROP_Y}, width={CROP_WIDTH}, height={CROP_HEIGHT}")
            first_frame = False

        cropped_frame = frame_rgb[CROP_Y : CROP_Y + CROP_HEIGHT, CROP_X : CROP_X + CROP_WIDTH]
        cropped_frames.append(cropped_frame)

    cap.release()

    if cropped_frames:
        print(f"クロップ後のサイズ: {cropped_frames[0].shape[1]}x{cropped_frames[0].shape[0]}")
        print(f"総フレーム数: {len(cropped_frames)}")
        imageio.mimsave(str(output_path), cropped_frames, fps=fps, macro_block_size=1)
        print(f"保存完了: {output_path}")
    else:
        print("フレームが見つかりませんでした")


def main():
    parser = argparse.ArgumentParser(description="動画のenvironment部分だけをクロップ")
    parser.add_argument("--input_dir", type=str, required=True, help="入力動画ディレクトリ")
    parser.add_argument("--output_dir", type=str, required=True, help="出力動画ディレクトリ")
    parser.add_argument(
        "--pattern", type=str, required=True, help="処理する動画のパターン（例: *.mp4）"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # パターンに一致する動画ファイルを取得
    video_files = sorted(list(input_dir.glob(args.pattern)))

    if not video_files:
        print(f"動画ファイルが見つかりません: {input_dir}/{args.pattern}")
        return

    print(f"処理する動画数: {len(video_files)}")

    for i, video_path in enumerate(video_files):
        print(f"\n[{i + 1}/{len(video_files)}] 処理中: {video_path.name}")
        output_path = output_dir / video_path.name

        try:
            crop_video(video_path, output_path)
        except Exception as e:
            print(f"エラー: {video_path.name} - {str(e)}")


if __name__ == "__main__":
    main()
