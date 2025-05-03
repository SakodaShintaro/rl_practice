from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBufferData:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device: torch.device,
    ) -> None:
        self.size = size
        self.action_shape = action_shape
        self.device = device

        self.observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.idx] = obs
        self.next_observations[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        idx = np.random.randint(0, self.size if self.full else self.idx, size=batch_size)
        return ReplayBufferData(
            torch.Tensor(self.observations[idx]).to(self.device),
            torch.Tensor(self.next_observations[idx]).to(self.device),
            torch.Tensor(self.actions[idx]).to(self.device),
            torch.Tensor(self.rewards[idx]).to(self.device),
            torch.Tensor(self.dones[idx]).to(self.device),
        )
