# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

import wandb
from agents.avg import AvgAgent
from agents.ppo import PpoAgent
from agents.sac import SacAgent
from utils import create_full_image_with_reward
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--trial_num", type=int, default=1)
    parser.add_argument(
        "--env_id",
        type=str,
        default="MiniGrid-MemoryS9-v0",
        choices=[
            "CarRacing-v3",
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-MemoryS9-v0",
            "MiniGrid-MemoryS11-v0",
        ],
    )
    parser.add_argument("--partial_obs", type=int, default=1, choices=[0, 1])
    parser.add_argument("--agent_type", type=str, default="ppo", choices=["sac", "avg", "ppo"])
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--target_score", type=float, default=0.95)
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument(
        "--encoder",
        type=str,
        default="stt",
        choices=["single_frame", "smolvlm", "mmmamba", "stt", "simple", "recurrent"],
    )
    parser.add_argument("--encoder_block_num", type=int, default=1)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_block_num", type=int, default=1)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_block_num", type=int, default=1)
    parser.add_argument("--predictor_hidden_dim", type=int, default=512)
    parser.add_argument("--predictor_block_num", type=int, default=2)
    parser.add_argument("--predictor_step_num", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--step_limit", type=int, default=1_000_000)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--action_norm_penalty", type=float, default=0.0)
    parser.add_argument("--render_reconstruction", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--tempo_block_type", type=str, default="transformer", choices=["transformer", "mamba"]
    )
    parser.add_argument("--debug", action="store_true")

    # for SAC
    parser.add_argument("--buffer_size", type=int, default=int(2e4))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--apply_masks_during_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_weight_projection", action="store_true")
    parser.add_argument("--detach_actor", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detach_critic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--detach_predictor", type=int, default=0, choices=[0, 1])

    # for AVG
    parser.add_argument("--use_eligibility_trace", action="store_true")
    parser.add_argument("--et_lambda", default=0.8, type=float)

    # for PPO/Recurrent PPO
    parser.add_argument("--buffer_capacity", type=int, default=4096)
    parser.add_argument(
        "--model_name", type=str, default="default", choices=["default", "paligemma"]
    )

    return parser.parse_args()


def main(args, exp_name: str, seed: int) -> None:
    wandb.init(
        project=f"rl_practice_{args.env_id}",
        config=vars(args),
        name=exp_name,
        save_code=True,
        settings=wandb.Settings(quiet=True),
    )

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    # Create result directories and save files only if not in debug mode
    if not args.debug:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # save seed to file
        with open(result_dir / "seed.txt", "w") as f:
            f.write(str(seed))

        # git show -sの結果とgit diffの結果を保存
        with open(result_dir / "git_info.txt", "w") as f:
            f.write(f"git show -s:\n{os.popen('git show -s').read()}\n")
            f.write(f"git diff:\n{os.popen('git diff').read()}\n")

        video_dir = result_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = None
        video_dir = None
    image_save_interval = 500
    log_episode_path = result_dir / "log_episode.tsv" if result_dir is not None else None
    log_episode_file = None
    log_episode_writer = None

    # env setup
    env = make_env(args.env_id, args.partial_obs)
    env.action_space.seed(seed)

    target_score = args.target_score if args.target_score is not None else env.spec.reward_threshold

    start_time = time.time()

    # start the game
    global_step = 0
    score_list = []
    best_score = -float("inf")
    obs, _ = env.reset(seed=seed)
    step_limit = args.step_limit

    if args.agent_type == "sac":
        agent = SacAgent(args, env.observation_space, env.action_space)
    elif args.agent_type == "avg":
        agent = AvgAgent(args, env.observation_space, env.action_space)
    elif args.agent_type == "ppo":
        agent = PpoAgent(args, env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    parameter_count = sum(p.numel() for p in agent.network.parameters())
    print(f"Parameter count: {parameter_count:,}")

    episode_id = 0
    while True:
        # initialize episode
        obs, _ = env.reset()
        obs_for_render = obs.copy().transpose(1, 2, 0)
        pred_image = np.zeros_like(obs_for_render)
        pred_reward = 0.0
        pred_image_loss = None
        pred_reward_loss = None
        initial_rgb_image = create_full_image_with_reward(
            env.render(), obs_for_render, np.zeros_like(obs_for_render), pred_image, 0.0, 0.0
        )
        bgr_image_list = [cv2.cvtColor(initial_rgb_image, cv2.COLOR_RGB2BGR)]
        agent.initialize_for_episode()
        action, agent_info = agent.select_action(global_step, obs, 0.0)

        while True:
            global_step += 1

            # step
            obs, reward, termination, truncation, env_info = env.step(action)
            action, agent_info = agent.step(global_step, obs, reward, termination, truncation)

            # log
            elapsed_time_sec = time.time() - start_time
            elapsed_time_min = elapsed_time_sec / 60
            data_dict = {
                "global_step": global_step,
                "elapsed_time_min": elapsed_time_min,
                "SPS": global_step / elapsed_time_sec,
                "reward": reward,
                **agent_info,
            }
            if pred_image_loss is not None:
                data_dict["losses/pred_image_loss"] = pred_image_loss
            if pred_reward_loss is not None:
                data_dict["losses/pred_reward_loss"] = pred_reward_loss
            wandb.log(data_dict)

            # render
            obs_for_render = obs.copy().transpose(1, 2, 0)
            if args.render_reconstruction:
                with torch.inference_mode():
                    net = agent.network.encoder.ae
                    x = torch.tensor(obs.copy(), device=net.device).unsqueeze(0)
                    x = net.encode(x).latents
                    x = net.decode(x).sample
                    curr_reconstruction = x.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            else:
                curr_reconstruction = np.zeros_like(obs_for_render)

            pred_image = agent_info.get("next_image", np.zeros_like(obs_for_render))
            pred_image_loss = np.mean(np.abs(pred_image - obs_for_render))
            pred_reward = agent_info.get("next_reward", 0.0)
            pred_reward_loss = np.abs(pred_reward - reward)
            rgb_image = create_full_image_with_reward(
                env.render(), obs_for_render, curr_reconstruction, pred_image, pred_reward, reward
            )
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image_list.append(bgr_image)
            if args.render:
                cv2.imshow("CarRacing", bgr_image)
                cv2.waitKey(1)

            if termination or truncation:
                break

            if global_step >= step_limit:
                break

        if global_step >= step_limit:
            break

        score = env_info["episode"]["r"]
        score_list.append(score)
        EVAL_RANGE = 100
        score_list = score_list[-EVAL_RANGE:]
        recent_average_score = np.mean(score_list)

        data_dict = {
            "global_step": global_step,
            "episodic_return": env_info["episode"]["r"],
            "episodic_length": env_info["episode"]["l"],
            "recent_average_score": recent_average_score,
        }
        wandb.log(data_dict)

        if result_dir is not None:
            if log_episode_writer is None:
                log_episode_file = open(log_episode_path, "w", newline="")
                fieldnames = list(data_dict.keys())
                log_episode_writer = csv.DictWriter(
                    log_episode_file, fieldnames=fieldnames, delimiter="\t"
                )
                log_episode_writer.writeheader()
            log_episode_writer.writerow(data_dict)
            log_episode_file.flush()

        is_solved = recent_average_score > target_score and episode_id >= EVAL_RANGE

        if episode_id % 5 == 0 or is_solved:
            print(
                f"Ep: {episode_id}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {env_info['episode']['l']:.2f}"
            )

        is_best = score > best_score

        if is_best and result_dir is not None:
            with open(result_dir / "best_score.txt", "w") as f:
                f.write(f"{episode_id + 1}\t{score:.2f}")
            best_score = score
            video_path = video_dir / f"best_episode.mp4"
            if bgr_image_list:
                rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
                imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

        if (
            episode_id == 0 or (episode_id + 1) % image_save_interval == 0
        ) and result_dir is not None:
            video_path = video_dir / f"ep_{episode_id + 1:08d}.mp4"
            if bgr_image_list:
                rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
                imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

        episode_id += 1

        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break

    env.close()
    if log_episode_file is not None:
        log_episode_file.close()
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.off_wandb = True
        args.learning_starts = 10
        args.buffer_capacity = 50
        args.render = 0
        args.step_limit = 100

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if not os.environ.get("DISPLAY"):
        print("Because a headless environment is detected, rendering is automatically disabled.")
        args.render = 0

    exp_name = f"{args.agent_type.upper()}_{args.exp_name}"
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)

    for i in range(args.trial_num):
        suffix = f"_{i:02d}" if args.trial_num > 1 else ""
        main(args, exp_name + suffix, seed + i)
