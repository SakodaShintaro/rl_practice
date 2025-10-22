import gymnasium as gym
import numpy as np


class RewardProcessor:
    """Reward Processor."""

    def __init__(self, processing_type: str) -> None:
        self.return_rms = gym.wrappers.utils.RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.type = processing_type

    def _symlog(self, x: float) -> float:
        """Symmetric log."""
        return np.sign(x) * np.log(1 + np.abs(x))

    def normalize(self, reward: float) -> float:
        """Normalize the reward."""
        if self.type == "none":
            result = reward
        elif self.type == "const":
            result = reward / 500.0
        elif self.type == "symlog":
            result = self._symlog(reward)
        elif self.type == "scaling":
            self.return_rms.update(np.array([reward]))
            result = reward / np.sqrt(self.return_rms.var + self.epsilon)
        elif self.type == "centering":
            self.return_rms.update(np.array([reward]))
            result = (reward - self.return_rms.mean) / np.sqrt(self.return_rms.var + self.epsilon)
        else:
            msg = "Invalid normalizer type"
            raise ValueError(msg)

        MAX_VALUE = 10.0
        result = np.clip(result, -MAX_VALUE, MAX_VALUE)
        return result


if __name__ == "__main__":
    rp_symlog = RewardProcessor("symlog")
    rp_scaling = RewardProcessor("scaling")
    rp_centering = RewardProcessor("centering")
    rewards = [10.0, 2.0, 3.0, 4.0, 5.0, -4.0, -10.0, 0.0, 1.0, -1.0]
    for r in rewards:
        norm_r_symlog = rp_symlog.normalize(r)
        norm_r_scaling = rp_scaling.normalize(r)
        norm_r_centering = rp_centering.normalize(r)
        print(f"{r=:.3f}, {norm_r_symlog=:.3f}, {norm_r_scaling=:.3f}, {norm_r_centering=:.3f}")
