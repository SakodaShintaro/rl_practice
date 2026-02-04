# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch


class RewardProcessor:
    """Reward Processor."""

    def __init__(self, processing_type: str, reward_scale: float) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.type = processing_type
        self.reward_scale = reward_scale
        assert self.reward_scale > 0.0

    def update(self, reward: float) -> None:
        """Update the running mean and std with the new reward."""
        self.return_rms.update(np.array([reward]))

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward * self.reward_scale
        elif self.type == "scaling":
            result = reward / np.sqrt(self.return_rms.var + self.epsilon)
            result *= self.reward_scale
        elif self.type == "centering":
            result = (reward - self.return_rms.mean) / np.sqrt(self.return_rms.var + self.epsilon)
            result *= self.reward_scale
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        MAX_VALUE = 10.0
        result = torch.clamp(result, -MAX_VALUE, MAX_VALUE)
        return result

    def inverse(self, reward: torch.Tensor) -> torch.Tensor:
        """Inverse normalization of the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward / self.reward_scale
        elif self.type == "scaling":
            reward /= self.reward_scale
            result = reward * np.sqrt(self.return_rms.var + self.epsilon)
        elif self.type == "centering":
            reward /= self.reward_scale
            result = reward * np.sqrt(self.return_rms.var + self.epsilon) + self.return_rms.mean
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        return result


if __name__ == "__main__":
    rp_scaling = RewardProcessor("scaling", 1.0)
    rp_centering = RewardProcessor("centering", 1.0)
    rewards = [0.5, 10.0, 2.0, 3.0, 4.0, 5.0, -4.0, -10.0, 0.0, 1.0, -1.0]
    for r in rewards:
        rp_scaling.update(r)
        rp_centering.update(r)
        r_tensor = torch.tensor(r)
        norm_r_scaling_tensor = rp_scaling.normalize(r_tensor)
        norm_r_centering_tensor = rp_centering.normalize(r_tensor)
        inv_r_scaling = rp_scaling.inverse(norm_r_scaling_tensor).item()
        inv_r_centering = rp_centering.inverse(norm_r_centering_tensor).item()
        norm_r_scaling = norm_r_scaling_tensor.item()
        norm_r_centering = norm_r_centering_tensor.item()
        print(
            f"{r=:+6.2f}, {norm_r_scaling=:+6.2f}, {norm_r_centering=:+6.2f} -> {inv_r_scaling=:+6.2f}, {inv_r_centering=:+6.2f}"
        )
