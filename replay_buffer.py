from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBufferData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        seq_len: int,
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device: torch.device,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.action_shape = action_shape
        self.device = device

        self.observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        curr_size = self.size if self.full else self.idx
        assert curr_size >= self.seq_len, "Not enough data to sample a sequence."
        indices = np.random.randint(0, curr_size - self.seq_len, size=batch_size)
        observations = []
        actions = []
        rewards = []
        dones = []
        for idx in indices:
            observations.append(self.observations[idx : idx + self.seq_len])
            actions.append(self.actions[idx : idx + self.seq_len])
            rewards.append(self.rewards[idx : idx + self.seq_len])
            dones.append(self.dones[idx : idx + self.seq_len])
        observations = np.stack(observations)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        return ReplayBufferData(
            torch.tensor(observations).to(self.device),
            torch.tensor(actions).to(self.device),
            torch.tensor(rewards).to(self.device),
            torch.tensor(dones).to(self.device),
        )
