# SPDX-License-Identifier: MIT
import argparse
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument(
        "--seq_len", type=int, required=True, help="Number of images per inference step"
    )
    parser.add_argument(
        "--tokens_per_step", type=int, required=True, help="Tokens to generate per step"
    )
    parser.add_argument(
        "--max_context_words", type=int, required=True, help="Max words to keep in context"
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--use_quantization", action="store_true")
    return parser.parse_args()


def load_model(model_id: str, use_quantization: bool, device: torch.device) -> tuple:
    """Load Qwen3.5 model and processor."""
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir="./cache",
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="./cache",
    )

    return model, processor


def build_messages(
    args: argparse.Namespace,
    image_paths: list[Path],
    previous_text: str,
) -> list[dict]:
    user_content: list[dict[str, object]] = []

    user_content.append({"type": "text", "text": args.prompt})

    if args.mode == "video":
        video_sources = [f"file://{path.resolve()}" for path in image_paths]
        user_content.append({"type": "video", "video": video_sources})
    else:
        for path in image_paths:
            user_content.append({"type": "image", "image": f"file://{path.resolve()}"})

    messages = [{"role": "user", "content": user_content}]

    if previous_text:
        messages.append({"role": "assistant", "content": previous_text})

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not previous_text,
        continue_final_message=bool(previous_text),
    )

    images, videos = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        return_tensors="pt",
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

    all_image_paths = sorted(args.images_dir.glob("*"))
    all_image_paths = [p for p in all_image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    assert all_image_paths, f"No images found in {args.images_dir}"

    print(f"Found {len(all_image_paths)} images")
    print(f"Sliding window: seq_len={args.seq_len}, tokens_per_step={args.tokens_per_step}")

    num_steps = max(1, len(all_image_paths) - args.seq_len + 1)
    seq_len = min(args.seq_len, len(all_image_paths))

    print(f"Will process {num_steps} steps")

    model, processor = load_model(args.model_id, args.use_quantization, "cuda")

    accumulated_text = ""
    total_time = 0.0

    with torch.inference_mode():
        for step in range(num_steps):
            window_start = step
            window_end = step + seq_len
            current_images = all_image_paths[window_start:window_end]

            context = truncate_context(accumulated_text, args.max_context_words)

            new_text, elapsed_msec = process_and_generate(
                model, processor, args, current_images, context
            )
            total_time += elapsed_msec

            accumulated_text = (accumulated_text + " " + new_text).strip()

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
