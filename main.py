import argparse
from pathlib import Path
from shutil import rmtree

import cv2
import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def check_sample(env_name: str) -> None:
    save_dir = Path("./result") / env_name

    rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialise the environment
    env = gym.make(env_name, render_mode="rgb_array")

    print(f"{env_name=}")
    print(f"{env.observation_space=}")
    print(f"{env.action_space=}")

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for i in range(100):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"{i=:08d}\t{reward=}", end="\r")

        r = env.render()  # (400, 600, 3)
        r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_dir / f"{i:08d}.png"), r)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
    print()

    env.close()
    print()


if __name__ == "__main__":
    args = parse_args()

    env_name_list = [
        # Classic control
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCarContinuous-v0",
        "MountainCar-v0",
        "Pendulum-v1",
        # Box2D
        "BipedalWalker-v3",
        "CarRacing-v3",
        "LunarLander-v3",
        # MuJoCo
        "Ant-v5",
    ]

    for env_name in env_name_list:
        check_sample(env_name)
