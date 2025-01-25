import argparse
from pathlib import Path
from shutil import rmtree

import cv2
import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Pendulum-v1", type=str)
    parser.add_argument("--save_dir", type=Path, default="result")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_name = args.env_name
    save_dir = args.save_dir / env_name

    rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialise the environment
    env = gym.make(env_name, render_mode="rgb_array")

    print(f"{env.observation_space=}")
    print(f"{env.action_space=}")

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
