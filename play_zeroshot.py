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

from networks.vlm_backbone import MMMambaEncoder, QwenVLEncoder, parse_action_text
from utils import concat_images, convert_to_uint8
from wrappers import make_env

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="CarRacing-v3",
        choices=["CarRacing-v3", "MiniGrid-Empty-5x5-v0"],
    )
    parser.add_argument(
        "--agent_type", type=str, default="qwenvl", choices=["random", "qwenvl", "mmmamba"]
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_episodes", type=int, default=10)

    return parser.parse_args()


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        return self.action_space.sample()


class VLMAgent:
    def __init__(self, encoder_type, observation_space_shape, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoder_type == "qwenvl":
            self.encoder = QwenVLEncoder(
                observation_space_shape=observation_space_shape, output_text=True
            )
        elif encoder_type == "mmmamba":
            self.encoder = MMMambaEncoder(
                observation_space_shape=observation_space_shape, device=self.device
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    @torch.inference_mode()
    def select_action(self, obs, prev_reward):
        # obs: (C, H, W) -> (B=1, T=1, C, H, W)
        obs_tensor = (
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Create dummy obs_z, actions, rewards, rnn_state
        obs_z = torch.zeros(1, 1, 1, device=self.device)  # dummy
        actions = torch.zeros(1, 1, 3, device=self.device)  # dummy actions
        rewards = torch.tensor([[[prev_reward]]], device=self.device)  # use previous reward
        rnn_state = self.encoder.init_state().to(self.device)

        _, _, action_text = self.encoder(obs_tensor, obs_z, actions, rewards, rnn_state)
        print(f"{action_text=}")
        action_array = parse_action_text(action_text)
        return action_array


def run_episode(env, agent, render=False):
    obs, _ = env.reset()

    total_reward = 0
    step_count = 0
    bgr_image_list = []

    obs_for_render = convert_to_uint8(obs.copy().transpose(1, 2, 0))
    bgr_image_list.append(concat_images([env.render(), obs_for_render]))
    prev_reward = 0.0

    while True:
        action = agent.select_action(obs, prev_reward)
        obs, reward, termination, truncation, env_info = env.step(action)

        prev_reward = reward

        total_reward += reward
        step_count += 1

        obs_for_render = convert_to_uint8(obs.copy().transpose(1, 2, 0))
        bgr_image = concat_images([env.render(), obs_for_render])
        bgr_image_list.append(bgr_image)

        if render:
            bgr_image = concat_images(
                [env.render(), convert_to_uint8(obs.copy().transpose(1, 2, 0))]
            )
            cv2.imshow("CarRacing", bgr_image)
            cv2.waitKey(1)

        if termination or truncation:
            break

    episode_length = env_info.get("episode", {}).get("l", step_count)
    episode_reward = env_info.get("episode", {}).get("r", total_reward)

    return episode_reward, episode_length, bgr_image_list, step_count


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
    env = make_env(args.env_id)
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # agent setup
    if args.agent_type == "random":
        agent = RandomAgent(env.action_space)
    else:
        agent = VLMAgent(args.agent_type, env.observation_space.shape)

    print(f"Running {args.num_episodes} episodes with {args.agent_type} agent...")

    episode_rewards = []
    episode_lengths = []
    best_reward = -float("inf")
    best_video = None
    global_step = 0

    start_time = time.time()

    for episode in range(args.num_episodes):
        episode_reward, episode_length, bgr_image_list, step_count = run_episode(
            env, agent, render=args.render
        )

        global_step += step_count
        elapsed_time = time.time() - start_time
        sps = global_step / elapsed_time if elapsed_time > 0 else 0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Save best episode video
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_video = bgr_image_list

        # Log episode data
        data_dict = {
            "episode": episode + 1,
            "global_step": global_step,
            "episodic_return": episode_reward,
            "episodic_length": episode_length,
            "average_return": np.mean(episode_rewards),
            "best_return": best_reward,
            "sps": sps,
            "elapsed_time": elapsed_time,
        }

        print(
            f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Length = {episode_length:3d}, "
            f"Global Step = {global_step:5d}, SPS = {sps:6.1f}"
        )

        # Save episode video
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
    final_sps = global_step / elapsed_time if elapsed_time > 0 else 0

    print(f"\n=== Results after {args.num_episodes} episodes ===")
    print(f"Agent type: {args.agent_type}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Steps per second: {final_sps:.1f}")

    # Final statistics
    final_stats = {
        "final/average_return": avg_reward,
        "final/std_return": std_reward,
        "final/best_return": best_reward,
        "final/average_length": avg_length,
        "final/global_step": global_step,
        "final/total_time": elapsed_time,
        "final/sps": final_sps,
    }

    # Save results to file
    with open(result_dir / "results.txt", "w") as f:
        f.write(f"Agent type: {args.agent_type}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Best reward: {best_reward:.2f}\n")
        f.write(f"Average episode length: {avg_length:.1f}\n")
        f.write(f"Total steps: {global_step}\n")
        f.write(f"Total time: {elapsed_time:.1f} seconds\n")
        f.write(f"Steps per second: {final_sps:.1f}\n")
        f.write(f"Episode rewards: {episode_rewards}\n")

    env.close()
    cv2.destroyAllWindows()
