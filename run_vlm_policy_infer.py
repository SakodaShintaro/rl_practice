import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from networks.vlm_policy_network import VLMPolicyNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("local/20260112_084706_OFF_POLICY_qwenvl"),
    )
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--action_hidden_dim", type=int, default=256)
    parser.add_argument("--value_hidden_dim", type=int, default=256)
    parser.add_argument("--target_layer_idx", type=int, default=-1)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--task_prompt", type=str, default="")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache"))
    parser.add_argument("--local_files_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--value_bins", type=int, default=51)
    parser.add_argument("--value_min", type=float, default=-10.0)
    parser.add_argument("--value_max", type=float, default=10.0)
    parser.add_argument("--euler_steps", type=int, default=5)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--use_quantization", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--image_processor_type", type=str, default="simple_cnn", choices=["simple_cnn", "ae"]
    )
    parser.add_argument("--dacer_loss_weight", type=float, default=0.05)
    return parser.parse_args()


def load_log(log_path: Path) -> list[dict[str, float]]:
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "action0": float(row["action0"]),
                    "action1": float(row["action1"]),
                    "action2": float(row["action2"]),
                    "reward": float(row["reward"]),
                }
            )
    return rows


def load_images(image_paths: list[Path]) -> torch.Tensor:
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        images.append(img_tensor)
    return torch.stack(images, dim=0)


def main() -> None:
    args = parse_args()
    best_dir = args.data_dir / "images" / "best_episode_frames"
    log_path = best_dir / "log.tsv"

    image_paths = sorted(best_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {best_dir}")

    rows = load_log(log_path)
    if len(rows) < args.seq_len or len(image_paths) < args.seq_len:
        raise ValueError("Not enough frames or log rows for the requested seq_len.")

    image_paths = image_paths[: args.seq_len]
    rows = rows[: args.seq_len]

    s_seq = load_images(image_paths).unsqueeze(0)
    action_seq = torch.tensor(
        [[row["action0"], row["action1"], row["action2"]] for row in rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    rewards = torch.tensor([[row["reward"]] for row in rows], dtype=torch.float32).unsqueeze(0)

    if args.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        cache_dir=str(args.cache_dir),
        local_files_only=bool(args.local_files_only),
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir),
        local_files_only=bool(args.local_files_only),
    )

    policy = VLMPolicyNetwork(
        action_dim=3,
        seq_len=args.seq_len,
        action_horizon=args.action_horizon,
        observation_space_shape=tuple(s_seq.shape[2:]),
        image_processor_type=args.image_processor_type,
        target_layer_idx=args.target_layer_idx,
        model=model,
        processor=processor,
        use_lora=True,
        task_prompt=args.task_prompt,
        action_hidden_dim=args.action_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        value_bins=args.value_bins,
        value_min=args.value_min,
        value_max=args.value_max,
        euler_steps=args.euler_steps,
        gamma=args.gamma,
        dacer_loss_weight=args.dacer_loss_weight,
    )

    device = model.device
    policy = policy.to(device)
    s_seq = s_seq.to(device)
    action_seq = action_seq.to(device)
    obs_z_seq = torch.zeros((1, args.seq_len, 1, 1, 1), dtype=torch.float32, device=device)
    rnn_state = policy.init_state().to(device)
    infer_dict = policy.infer(
        s_seq,
        obs_z_seq,
        action_seq,
        rewards.to(device),
        rnn_state,
    )
    pred_action = infer_dict["action"]
    pred_value = infer_dict["value"]

    print("pred_action:", pred_action.detach().cpu().numpy())
    print("pred_value:", pred_value.detach().cpu().numpy())


if __name__ == "__main__":
    main()
