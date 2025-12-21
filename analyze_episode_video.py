"""
QwenVLを使用してエピソード動画を解析し、失敗原因や反省点をテキスト生成するスクリプト
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--max_videos", type=int, default=3)
    return parser.parse_args()


def load_episode_scores(video_dir):
    """log_episode.tsvからエピソードスコアを読み込む

    Args:
        video_dir: 動画ディレクトリのパス

    Returns:
        エピソード番号をキーとするスコアの辞書、最高得点
    """
    tsv_path = video_dir.parent / "log_episode.tsv"
    best_score_path = video_dir.parent / "best_score.txt"

    scores = {}
    max_score = 0.0

    if tsv_path.exists():
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            episode_num = 1
            for row in reader:
                score = float(row["episodic_return"])
                scores[episode_num] = score
                max_score = max(max_score, score)
                episode_num += 1

    best_score = max_score
    if best_score_path.exists():
        with open(best_score_path, "r") as f:
            parts = f.read().strip().split("\t")
            if len(parts) == 2:
                best_score = float(parts[1])
                max_score = max(max_score, best_score)

    return scores, max_score


def get_episode_info(video_filename, scores, max_score):
    """動画ファイル名からエピソード情報を取得

    Args:
        video_filename: 動画ファイル名
        scores: エピソードスコアの辞書
        max_score: 最高得点

    Returns:
        エピソード番号、スコア、最高得点
    """
    if video_filename == "best_episode.mp4":
        return "best", max_score, max_score

    match = re.search(r"ep_(\d+)", video_filename)
    if match:
        episode_num = int(match.group(1))
        score = scores.get(episode_num, 0.0)
        return episode_num, score, max_score

    return None, 0.0, max_score


def load_video_frames(video_path):
    """動画からフレームを抽出

    Args:
        video_path: 動画ファイルのパス

    Returns:
        フレームのリスト（PIL Image形式）、FPS
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    cap.release()

    return frames, fps


def analyze_episode_with_qwen(
    video_path, best_video_path, model, processor, device, episode_score, max_score
):
    """QwenVLで動画を解析してエピソードの反省点を生成

    Args:
        video_path: 動画ファイルのパス
        best_video_path: ベストエピソードの動画ファイルのパス
        model: QwenVLモデル
        processor: QwenVLプロセッサ
        device: 使用するデバイス
        episode_score: このエピソードのスコア
        max_score: 最高得点

    Returns:
        生成されたテキスト
    """
    # 動画からフレームを抽出
    frames, fps = load_video_frames(video_path)
    best_frames, best_fps = load_video_frames(best_video_path)

    print(f"対象エピソード - フレーム数: {len(frames)}, FPS: {fps}")
    print(f"ベストエピソード - フレーム数: {len(best_frames)}, FPS: {best_fps}")

    prompt = f"""You are shown TWO videos of a reinforcement learning agent in the CarRacing-v3 environment.
The red car is the learning agent, which must drive on the gray road and avoid going onto the green grass.

VIDEO 1 (BEST EPISODE): Score = {max_score:.2f} points (Best performance)
VIDEO 2 (CURRENT EPISODE): Score = {episode_score:.2f} points ({episode_score / max_score * 100:.1f}% of best)

TASK: Compare these two episodes and analyze why Video 2 performed worse than Video 1.

Please provide insights on the following points:

1. Performance Comparison
   - What are the key differences in driving behavior between the best episode and current episode?
   - Speed control: How does the current episode's speed management compare?

2. Failure Analysis (Current Episode)
   - What specific mistakes did the agent make in the current episode that the best episode avoided?
   - Where did the agent go off-track or onto the grass?
   - What were the critical errors that led to the lower score?

3. Learning Improvements
   - Based on the comparison, what should the agent learn from the best episode?
   - What specific behaviors need to be corrected to reach best episode performance?
   - What patterns from the best episode should be replicated?

Please provide a detailed comparison-based analysis."""

    # チャットメッセージ形式で構築
    content = [
        {"type": "text", "text": "Video 1 (Best Episode):"},
        {"type": "video", "video": best_frames, "fps": best_fps},
        {"type": "text", "text": "Video 2 (Current Episode):"},
        {"type": "video", "video": frames, "fps": fps},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": content}]

    # テキストテンプレート適用
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ビジョン情報処理
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        video_metadata = None

    # プロセッサで入力準備
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )

    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }

    # テキスト生成
    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return decoded[0].strip() if decoded else ""


def main():
    args = parse_args()
    video_dir = args.video_dir

    device = "cuda"

    # エピソードスコアを読み込む
    scores, max_score = load_episode_scores(video_dir)
    print(f"最高得点: {max_score:.2f}")

    # モデルとプロセッサのロード
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    # model_id = "Qwen/Qwen3-VL-8B-Instruct"
    # model_id = "Qwen/Qwen3-VL-8B-Thinking"
    # model_id = "Qwen/Qwen3-VL-32B-Instruct"
    print(f"モデルをロード中: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        cache_dir="./cache",
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    print("モデルのロードが完了しました")

    # 動画ファイルの取得
    video_files = sorted(list(video_dir.glob("*.mp4")))[: args.max_videos]

    if not video_files:
        print(f"動画ファイルが見つかりません: {video_dir}")
        return

    # best_episode.mp4のパスを取得
    best_video_path = video_dir / "best_episode.mp4"
    if not best_video_path.exists():
        print(f"best_episode.mp4が見つかりません: {best_video_path}")
        return

    # best_episode.mp4以外のファイルのみを解析対象とする
    video_files = [v for v in video_files if v.name != "best_episode.mp4"]

    if not video_files:
        print("解析対象の動画ファイルが見つかりません（best_episode.mp4以外）")
        return

    print(f"\n解析する動画数: {len(video_files)}")

    # 結果を保存
    results = []

    for i, video_path in enumerate(video_files):
        print(f"\n{'=' * 80}")
        print(f"[{i + 1}/{len(video_files)}] 解析中: {video_path.name}")
        print(f"{'=' * 80}")

        episode_num, episode_score, max_score_val = get_episode_info(
            video_path.name, scores, max_score
        )
        print(f"エピソード番号: {episode_num}, スコア: {episode_score:.2f} / {max_score_val:.2f}")

        analysis = analyze_episode_with_qwen(
            str(video_path),
            str(best_video_path),
            model,
            processor,
            device,
            episode_score,
            max_score_val,
        )

        result = f"""
エピソード: {video_path.name} (Score: {episode_score:.2f} / {max_score_val:.2f})
{"=" * 80}
{analysis}
{"=" * 80}
"""
        print(result)
        results.append(result)

    # 結果をファイルに保存（動画ディレクトリと同じ階層に固定）
    output_path = video_dir.parent / "episode_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results))

    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
