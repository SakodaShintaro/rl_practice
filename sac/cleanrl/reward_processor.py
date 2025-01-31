import gymnasium as gym
import numpy as np


class RewardProcessor:
    """Reward Processor."""

    def __init__(self, processing_type: str) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.type = processing_type

    def symlog(self, x: float) -> float:
        """Symmetric log."""
        return np.sign(x) * np.log(1 + np.abs(x))

    def normalize(self, reward: float) -> float:
        """Normalize the reward."""
        if self.type == "none":
            return reward
        elif self.type == "const":
            return reward / 500.0
        elif self.type == "symlog":
            return self.symlog(reward)
        elif self.type == "scaling":
            self.return_rms.update(np.array([reward]))
            return reward / np.sqrt(self.return_rms.var + self.epsilon)
        elif self.type == "centering":
            self.return_rms.update(np.array([reward]))
            return (reward - self.return_rms.mean) / np.sqrt(self.return_rms.var + self.epsilon)
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)
