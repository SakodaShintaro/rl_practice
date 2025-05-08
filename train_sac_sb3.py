import argparse
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wrappers import (
    REPEAT,
    ActionRepeatWrapper,
    AverageRewardEarlyStopWrapper,
    DieStateRewardWrapper,
)


def make_env(video_dir):
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = ActionRepeatWrapper(env, repeat=REPEAT)
    env = AverageRewardEarlyStopWrapper(env)
    env = DieStateRewardWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, episode_trigger=lambda x: x % 100 == 0
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off_wandb", action="store_true")
    return parser.parse_args()


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.logger.record("rollout/ep_rew_mean", ep_rew)
                self.logger.record("rollout/ep_len_mean", ep_len)
                wandb.log(
                    {
                        "global_step": self.num_timesteps,
                        "episodic_return": info["episode"]["r"],
                        "episodic_length": info["episode"]["l"],
                    }
                )

        model = self.model
        if hasattr(model, "actor_loss") and hasattr(model, "critic_loss"):
            wandb.log(
                {
                    "global_step": self.num_timesteps,
                    "losses/actor_loss": float(model.actor_loss.detach().cpu()),
                    "losses/qf_loss": float(model.critic_loss.detach().cpu()),
                    "losses/alpha": float(model.log_ent_coef.exp().detach().cpu()),
                }
            )
        return True


if __name__ == "__main__":
    args = parse_args()

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project="cleanRL", config=vars(args), name="SAC_sb3", monitor_gym=True, save_code=True
    )

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_SAC_sb3"
    result_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(result_dir / "video")

    model = SAC("CnnPolicy", env, verbose=1, buffer_size=1000000 // 2)

    model.learn(total_timesteps=1000000, callback=CustomCallback())
