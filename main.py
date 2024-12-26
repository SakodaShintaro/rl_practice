import argparse
from pathlib import Path

import cv2
import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=Path, default="result")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir

    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialise the environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for i in range(1000):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"{i=:08d}\t{reward=}")

        r = env.render()  # (400, 600, 3)
        cv2.imwrite(str(save_dir / f"{i:08d}.png"), r)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
