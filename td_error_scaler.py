# from: https://github.com/gauthamvasan/avg/blob/main/incremental_rl/td_error_scaler.py
import numpy as np


def make_np(x):
    return np.array(x).copy().astype("float32")


class RunningStats:
    """Computes running mean and standard deviation
    Adapted from:
        * http://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
        * http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html
        * https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n: float = 0.0, m: float | None = None, s: float | None = None) -> None:
        self.n = n
        self.m = m
        self.s = s

    def clear(self) -> None:
        self.n = 0.0

    def update(self, x: float) -> None:
        x = make_np(x)
        self.update_params(x)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.0
            return RunningStats(
                sum_ns,
                (self.m * self.n + other.m * other.n) / sum_ns,
                self.s + other.s + delta2 * prod_ns / sum_ns,
            )
        else:
            self.update(other)
            return self

    @property
    def mean(self) -> float:
        return self.m if self.n else 0.0

    @property
    def variance(self) -> float:
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def __repr__(self) -> str:
        return f"<RunningMean(mean={self.mean: 2.4f}, std={self.std: 2.4f}, n={self.n: 2f}, m={self.m: 2.4f}, s={self.s: 2.4f})>"

    def __str__(self) -> str:
        return f"mean={self.mean: 2.4f}, std={self.std: 2.4f}"

    def zscore(self, x: float) -> float:
        return (x - self.mean) / (self.std + 1e-5)


class TDErrorScaler:
    """A simple implementation of:
    Schaul, T., Ostrovski, G., Kemaev, I., & Borsa, D. (2021).
    Return-based scaling: Yet another normalisation trick for deep RL.
    arXiv preprint arXiv:2105.05347.
    - URL:  https://arxiv.org/pdf/2105.05347.pdf

    Usage: Push the statistics online _before_ a learning update. Scale TD error by sigma.
    """

    def __init__(self):
        self.gamma_rms = RunningStats()
        self.return_sq_rms = RunningStats()
        self.reward_rms = RunningStats()
        self.return_rms = RunningStats()

    def update(self, reward, gamma, G):
        if G is not None:
            self.return_sq_rms.update(G**2)
            self.return_rms.update(G)
        self.reward_rms.update(reward)
        self.gamma_rms.update(gamma)

    @property
    def sigma(self):
        variance = max(
            self.reward_rms.variance + self.gamma_rms.variance * self.return_sq_rms.mean, 1e-4
        )

        # N.B: Do not scale until the first return is seen
        if variance <= 0.01 and self.return_sq_rms.n == 0:
            return 1

        return np.sqrt(variance)
