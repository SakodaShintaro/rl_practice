import argparse
import time
from pathlib import Path

import torch
import unsloth
from PIL import Image
from torchvision import transforms

from networks.vlm_actor_critic_with_state_value import VLMActorCriticWithStateValue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe max context length and batch size for VLM core training step."
    )
    parser.add_argument(
        "--images_root",
        type=Path,
        default=Path("local/20260112_084706_OFF_POLICY_qwenvl/images"),
    )
    parser.add_argument("--episode_dir", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--seq_lens", type=str, default="8,16,32,64")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8")
    parser.add_argument("--with_backward", type=int, default=0, choices=[0, 1])

    parser.add_argument("--vlm_model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--use_quantization", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_lora", type=int, default=0, choices=[0, 1])
    parser.add_argument("--target_layer_idx", type=int, default=-1)
    parser.add_argument("--use_unsloth", type=int, default=0, choices=[0, 1])

    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=1)
    parser.add_argument("--value_range", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--image_processor_type", type=str, default="simple_cnn")

    return parser.parse_args()


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _select_episode_dir(images_root: Path, episode_dir: str) -> Path:
    if episode_dir:
        return images_root / episode_dir
    candidates = sorted(p for p in images_root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No episode dirs found under {images_root}")
    return candidates[0]


def _load_images(images_dir: Path, num_frames: int, image_size: int) -> torch.Tensor:
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    image_paths = image_paths[:num_frames]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))

    return torch.stack(images, dim=0)


def _ensure_seq_len(images: torch.Tensor, seq_len: int) -> torch.Tensor:
    if images.size(0) >= seq_len:
        return images[:seq_len]
    repeat = (seq_len + images.size(0) - 1) // images.size(0)
    tiled = images.repeat(repeat, 1, 1, 1)
    return tiled[:seq_len]


def _try_run(
    actor_critic: VLMActorCriticWithStateValue,
    images_seq: torch.Tensor,
    seq_len: int,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[bool, float, float]:
    device = images_seq.device
    images = _ensure_seq_len(images_seq, seq_len)
    images = images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    rewards = torch.zeros(batch_size, seq_len, 1, device=device)
    actions = torch.randn(batch_size, 2, device=device).clamp(-1.0, 1.0)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()

    value, _, log_prob = actor_critic._compute_value_and_log_prob(images, rewards, actions)

    loss = value.mean() + log_prob.mean()
    if args.with_backward:
        loss.backward()

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    del images, rewards, actions, value, log_prob, loss
    torch.cuda.empty_cache()

    return True, elapsed_ms, peak_mem


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")

    episode_dir = _select_episode_dir(args.images_root, args.episode_dir)
    images_seq = _load_images(episode_dir, args.num_frames, args.image_size).to(device)

    observation_space_shape = (3, args.image_size, args.image_size)
    action_space_shape = (2,)
    seq_lens = _parse_int_list(args.seq_lens)
    if seq_lens:
        args.seq_len = max(seq_lens)

    actor_critic = VLMActorCriticWithStateValue(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        args=args,
    )
    actor_critic.train(mode=bool(args.with_backward))

    batch_sizes = _parse_int_list(args.batch_sizes)

    print(f"Episode dir: {episode_dir}")
    print(f"Loaded frames: {images_seq.shape[0]}")
    print(f"Seq lens: {seq_lens}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Backward: {bool(args.with_backward)}")

    results = []
    for seq_len in seq_lens:
        max_ok = 0
        for batch_size in batch_sizes:
            try:
                ok, elapsed_ms, peak_mem = _try_run(
                    actor_critic, images_seq, seq_len, batch_size, args
                )
                if ok:
                    max_ok = batch_size
                results.append((seq_len, batch_size, True, elapsed_ms, peak_mem))
                print(
                    f"OK seq={seq_len} batch={batch_size} time={elapsed_ms:.1f}ms peak={peak_mem:.2f}GB"
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    results.append((seq_len, batch_size, False, 0.0, 0.0))
                    print(f"OOM seq={seq_len} batch={batch_size}")
                    break
                raise
        print(f"Max batch for seq={seq_len}: {max_ok}")

    ok_rows = [r for r in results if r[2]]
    if ok_rows:
        best = max(ok_rows, key=lambda x: (x[0], x[1]))
        print(
            f"Best OK (largest seq,batch in test list): seq={best[0]} batch={best[1]}"
        )


if __name__ == "__main__":
    main()
