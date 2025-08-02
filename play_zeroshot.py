import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch

from networks.backbone import AE, MMMambaEncoder, QwenVLEncoder, SmolVLMEncoder, parse_action_text
from utils import concat_images
from wrappers import make_env


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        return self.action_space.sample()

    def initialize_for_episode(self):
        pass


class VLMAgent:
    def __init__(self, encoder_type, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoder_type == "ae":
            self.encoder = AE(device=self.device)
        elif encoder_type == "smolvlm":
            self.encoder = SmolVLMEncoder(device=self.device)
        elif encoder_type == "qwenvl":
            self.encoder = QwenVLEncoder(device=self.device)
        elif encoder_type == "mmmamba":
            self.encoder = MMMambaEncoder(device=self.device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    @torch.inference_mode()
    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, action_text = self.encoder(obs_tensor)
        action_array = parse_action_text(action_text)
        return action_array

    def initialize_for_episode(self):
        self.encoder.reset_inference_params()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="CarRacing-v3",
        choices=["CarRacing-v3", "MiniGrid-Empty-5x5-v0", "MiniGrid-MemoryS11-v0"],
    )
    parser.add_argument("--partial_obs", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--agent_type",
        type=str,
        default="random",
        choices=["random", "ae", "smolvlm", "qwenvl", "mmmamba"],
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--save_video", action="store_true")

    return parser.parse_args()


def run_episode(env, agent, render=False, save_video=False):
    obs, _ = env.reset()
    agent.initialize_for_episode()

    total_reward = 0
    step_count = 0
    bgr_image_list = []

    if save_video:
        obs_for_render = obs.copy().transpose(1, 2, 0)
        bgr_image_list.append(concat_images(env.render(), [obs_for_render]))

    while True:
        action = agent.select_action(obs)
        obs, reward, termination, truncation, env_info = env.step(action)

        total_reward += reward
        step_count += 1

        if save_video:
            obs_for_render = obs.copy().transpose(1, 2, 0)
            bgr_image = concat_images(env.render(), [obs_for_render])
            bgr_image_list.append(bgr_image)

        if render:
            bgr_image = concat_images(env.render(), [obs.copy().transpose(1, 2, 0)])
            cv2.imshow("CarRacing", bgr_image)
            cv2.waitKey(1)

        if termination or truncation:
            break

    episode_length = env_info.get("episode", {}).get("l", step_count)
    episode_reward = env_info.get("episode", {}).get("r", total_reward)

    return episode_reward, episode_length, bgr_image_list


if __name__ == "__main__":
    args = parse_args()

    exp_name = f"ZEROSHOT_{args.agent_type.upper()}"

    # seeding
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # save seed to file
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    video_dir = result_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # env setup
    env = make_env(args.env_id, args.partial_obs)
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    if args.agent_type == "random":
        agent = RandomAgent(env.action_space)
    else:
        agent = VLMAgent(args.agent_type)

    print(f"Running {args.num_episodes} episodes with {args.agent_type} agent...")

    episode_rewards = []
    episode_lengths = []
    best_reward = -float("inf")
    best_video = None

    start_time = time.time()

    for episode in range(args.num_episodes):
        episode_reward, episode_length, bgr_image_list = run_episode(
            env,
            agent,
            render=args.render and episode == 0,  # Only render first episode
            save_video=args.save_video,
        )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Save best episode video
        if episode_reward > best_reward:
            best_reward = episode_reward
            if bgr_image_list:
                best_video = bgr_image_list

        # Log episode data
        data_dict = {
            "episode": episode + 1,
            "episodic_return": episode_reward,
            "episodic_length": episode_length,
            "average_return": np.mean(episode_rewards),
            "best_return": best_reward,
        }

        print(
            f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:3d}"
        )

        # Save episode video
        if args.save_video and bgr_image_list:
            video_path = video_dir / f"ep_{episode + 1:03d}.mp4"
            rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_image_list]
            imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    # Save best episode video
    if best_video:
        video_path = video_dir / "best_episode.mp4"
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in best_video]
        imageio.mimsave(str(video_path), rgb_images, fps=10, macro_block_size=1)

    # Calculate and display statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    elapsed_time = time.time() - start_time

    print(f"\n=== Results after {args.num_episodes} episodes ===")
    print(f"Agent type: {args.agent_type}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Total time: {elapsed_time:.1f} seconds")

    # Final statistics
    final_stats = {
        "final/average_return": avg_reward,
        "final/std_return": std_reward,
        "final/best_return": best_reward,
        "final/average_length": avg_length,
        "final/total_time": elapsed_time,
    }

    # Save results to file
    with open(result_dir / "results.txt", "w") as f:
        f.write(f"Agent type: {args.agent_type}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Best reward: {best_reward:.2f}\n")
        f.write(f"Average episode length: {avg_length:.1f}\n")
        f.write(f"Total time: {elapsed_time:.1f} seconds\n")
        f.write(f"Episode rewards: {episode_rewards}\n")

    env.close()
    cv2.destroyAllWindows()
