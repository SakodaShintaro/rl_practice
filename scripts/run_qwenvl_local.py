# SPDX-License-Identifier: MIT
import argparse
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info

from rl_practice.networks.vlm_backbone import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default=Path("local/image/ep_00000001"))
    parser.add_argument("--seq_len", type=int, default=5, help="Number of images per inference step")
    parser.add_argument("--tokens_per_step", type=int, default=5, help="Tokens to generate per step")
    parser.add_argument("--max_context_words", type=int, default=100, help="Max words to keep in context")
    parser.add_argument("--prompt", type=str, default="Please describe the scene.")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    return parser.parse_args()


def build_messages(
    args: argparse.Namespace,
    image_paths: list[Path],
    previous_text: str = "",
) -> list[dict]:
    user_content: list[dict[str, object]] = []

    # Task prompt
    user_content.append({"type": "text", "text": args.prompt})

    # Images
    if args.mode == "video":
        video_sources = [f"file://{path.resolve()}" for path in image_paths]
        user_content.append({"type": "video", "video": video_sources})
    else:
        for path in image_paths:
            user_content.append({"type": "image", "image": f"file://{path.resolve()}"})

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # Add previous text as assistant's partial response to continue from
    if previous_text:
        messages.append({
            "role": "assistant",
            "content": previous_text,
        })

    return messages


def truncate_context(text: str, max_words: int) -> str:
    """Keep only the last max_words words from the text."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def process_and_generate(
    model,
    processor,
    args: argparse.Namespace,
    image_paths: list[Path],
    previous_text: str,
) -> tuple[str, float]:
    """Process images and generate tokens, returning generated text and elapsed time."""
    messages = build_messages(args, image_paths, previous_text)

    # If we have previous_text, continue from that assistant message
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not previous_text,
        continue_final_message=bool(previous_text),
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    ).to(model.device)

    start = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=args.tokens_per_step)
    elapsed_msec = (time.time() - start) * 1000

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text, elapsed_msec


def main() -> None:
    args = parse_args()

    # Load all images
    all_image_paths = sorted(args.images_dir.glob("*"))
    all_image_paths = [p for p in all_image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    if not all_image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    print(f"Found {len(all_image_paths)} images")
    print(f"Sliding window: seq_len={args.seq_len}, tokens_per_step={args.tokens_per_step}")

    # Calculate number of steps
    num_steps = len(all_image_paths) - args.seq_len + 1
    if num_steps <= 0:
        print(f"Not enough images. Need at least {args.seq_len}, got {len(all_image_paths)}")
        num_steps = 1
        args.seq_len = len(all_image_paths)

    print(f"Will process {num_steps} steps")

    model, processor = load_model(
        "Qwen/Qwen3-VL-2B-Instruct",
        use_quantization=True,
        use_lora=False,
        device="cuda",
    )

    accumulated_text = ""
    total_time = 0.0

    with torch.inference_mode():
        for step in range(num_steps):
            # Get sliding window of images
            window_start = step
            window_end = step + args.seq_len
            current_images = all_image_paths[window_start:window_end]

            # Truncate context to limit words
            context = truncate_context(accumulated_text, args.max_context_words)

            # Generate
            new_text, elapsed_msec = process_and_generate(
                model, processor, args, current_images, context
            )
            total_time += elapsed_msec

            # Accumulate text
            accumulated_text = (accumulated_text + " " + new_text).strip()

            # Print progress
            print(f"\n--- Step {step}/{num_steps - 1} (images {window_start}-{window_end - 1}) ---")
            print(f"Generated: {new_text}")
            print(f"Time: {elapsed_msec:.2f} ms")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(accumulated_text)
    print(f"\nTotal time: {total_time:.2f} ms")
    print(f"Average time per step: {total_time / num_steps:.2f} ms")


if __name__ == "__main__":
    main()
