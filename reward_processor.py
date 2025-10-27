import gymnasium as gym
import numpy as np


class RewardProcessor:
    """Reward Processor."""

    def __init__(self, processing_type: str, constant: float) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.type = processing_type
        self.constant = constant

    def update(self, reward: float) -> None:
        """Update the running mean and std with the new reward."""
        self.return_rms.update(np.array([reward]))

    def normalize(self, reward: float) -> float:
        """Normalize the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward / self.constant
        elif self.type == "symlog":
            result = self._symlog(reward)
        elif self.type == "scaling":
            result = reward / np.sqrt(self.return_rms.var + self.epsilon)
        elif self.type == "centering":
            result = (reward - self.return_rms.mean) / np.sqrt(self.return_rms.var + self.epsilon)
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        MAX_VALUE = 10.0
        result = np.clip(result, -MAX_VALUE, MAX_VALUE)
        return result

    def inverse(self, reward: float) -> float:
        """Inverse normalization of the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward * self.constant
        elif self.type == "scaling":
            result = reward * np.sqrt(self.return_rms.var + self.epsilon)
        elif self.type == "centering":
            result = reward * np.sqrt(self.return_rms.var + self.epsilon) + self.return_rms.mean
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        return result


if __name__ == "__main__":
    rp_scaling = RewardProcessor("scaling", 0.0)
    rp_centering = RewardProcessor("centering", 0.0)
    rewards = [0.5, 10.0, 2.0, 3.0, 4.0, 5.0, -4.0, -10.0, 0.0, 1.0, -1.0]
    for r in rewards:
        rp_scaling.update(r)
        rp_centering.update(r)
        norm_r_scaling = rp_scaling.normalize(r)
        norm_r_centering = rp_centering.normalize(r)
        inv_r_scaling = rp_scaling.inverse(norm_r_scaling)
        inv_r_centering = rp_centering.inverse(norm_r_centering)
        print(
            f"{r=:+6.2f}, {norm_r_scaling=:+6.2f}, {norm_r_centering=:+6.2f} -> {inv_r_scaling=:+6.2f}, {inv_r_centering=:+6.2f}"
        )
