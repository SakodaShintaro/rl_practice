import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default=Path("local/image/ep_00000001"))
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="Please describe the image(s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.images_dir.glob("*.png"))[: args.num_images]
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {args.images_dir}")

    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-2B-Instruct",
    #     dtype="auto",
    #     device_map="auto",
    # )
    # processor = AutoProcessor.from_pretrained(
    #     "Qwen/Qwen3-VL-2B-Instruct",
    #     cache_dir="./cache",
    #     dtype=torch.bfloat16,
    # )

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        cache_dir="./cache",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": args.prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
