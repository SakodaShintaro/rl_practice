import argparse
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default=Path("local/image/ep_00000001"))
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="Please describe the scene.")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    return parser.parse_args()


def build_messages(args: argparse.Namespace, image_paths: list[Path]) -> list[dict]:
    content: list[dict[str, object]] = []
    if args.mode == "video":
        video_sources = [f"file://{path.resolve()}" for path in image_paths]
        content.append({"type": "video", "video": video_sources})
    else:
        for path in image_paths:
            content.append({"type": "image", "image": f"file://{path.resolve()}"} )
    content.append({"type": "text", "text": args.prompt})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.images_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    image_paths = image_paths[: args.num_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    messages = build_messages(args, image_paths)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
